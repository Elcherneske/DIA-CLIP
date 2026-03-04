import torch
import os
import pickle
import numpy as np
from tqdm import tqdm
import pandas as pd
from torch.utils.tensorboard import SummaryWriter
from torch.nn import functional as F
from .FDR import FDRUtils
from MolecularUtils import ModificationUtils

# 配置日志
import logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

class ModelInfer():
    def __init__(self, args, model):
        self.model = model
        self.device = args.get_config('General', 'device')

        # evaluation parameters
        self.FDR = args.get_config('Infer', 'FDR')

        # result
        self.save_result_dir = args.get_config('General', 'out_dir')
        self.score_column = args.get_config('Infer', 'score_column', default='score')
        # self.feature_info = args.get_config('Infer', 'feature_info')

        # parse model path
        model_path = args.get_config('Infer', 'checkpoint_path')
        if os.path.exists(model_path) and model_path.endswith('.pt'):
            self.model.load(model_path)
            if torch.cuda.device_count() > 1:
                logging.info(f"Found {torch.cuda.device_count()} GPUs, use DataParallel")
                self.model = torch.nn.DataParallel(self.model)
            self.model.to(self.device)
            logging.info(f"Loaded model from {model_path}")
        else:
            raise ValueError(f"Invalid model path: {model_path}")

    def infer(self, data_loader, xic_datas: list, basename: str):
        results_df = self.infer_model(data_loader)

        # build a map for xic data
        xic_data_map = {}
        for xic_data in xic_datas:
            xic_data_map[ModificationUtils.format_modified_sequence(xic_data['pre']['peptide'], xic_data['pre']['modification']) + '_' + str(xic_data['pre']['charge'])] = xic_data

        quant_results = []
        for index, row in results_df.iterrows():
            modified_peptide = row['modified_peptide']
            charge = row['charge']
            precursor_key = modified_peptide + '_' + str(charge)
            if precursor_key not in xic_data_map:
                logging.warning(f"precursor key {precursor_key} not found in xic data map, skip this sample")
                quant_results.append(None)
            else:
                quant_result = self.quant_precursor(xic_data_map[precursor_key], topk=6)
                quant_results.append(quant_result)
        results_df["quant_result"] = quant_results
            
        fdr_df = self.evaluate_results(results_df, self.score_column)

        logging.info(f"Identified {len(fdr_df)} peptides with FDR {self.FDR:.2%}")

        # Save results to files
        if self.save_result_dir:
            results_df.to_csv(f"{self.save_result_dir}/{basename}.all.tsv", sep='\t', index=False)
            fdr_df.to_csv(f"{self.save_result_dir}/{basename}.fdr.tsv", sep='\t', index=False)
        else:
            raise ValueError(f"Invalid save result directory: {self.save_result_dir}")

    def infer_model(self, data_loader):
        self.model.eval()
        result = {
            'label': [],
            'score': [],
            'feature_distance': [],
            'cos_similarity': [],
            'modified_peptide': [],
            'charge': [],
            'peptide_feature': [],
            'spec_feature': [],
            # 'original_index': [],
        }
        with torch.no_grad():
            with tqdm(data_loader, desc=f"Inference:", total=len(data_loader)) as pbar:
                for n_step, data in enumerate(pbar):
                    output = self.model(data)
                    result['label'].extend(data["label"].cpu().detach().numpy().tolist())
                    if 'score' in output:   # if score is not None
                        result['score'].extend(output['score'].cpu().detach().numpy().tolist())
                    else:
                        result['score'].extend([-1] * len(data["label"]))
                    result['feature_distance'].extend((-1 * torch.norm(output["peptide_feature"] - output["spec_feature"], dim=-1, keepdim=False)).cpu().detach().numpy().tolist())
                    result['cos_similarity'].extend(F.cosine_similarity(output["peptide_feature"], output["spec_feature"], dim=-1).cpu().detach().numpy().tolist() )
                    result['modified_peptide'].extend(data["modified_peptide"])
                    result['charge'].extend(data["charge"].cpu().detach().numpy().tolist())
                    result['peptide_feature'].extend(output["peptide_feature"].cpu().detach().numpy().tolist())
                    result['spec_feature'].extend(output["spec_feature"].cpu().detach().numpy().tolist())
                    # result['original_index'].extend([i for i in range(n_step * len(data["label"]), (n_step + 1) * len(data["label"]))])

            results_df = pd.DataFrame(result)
            results_df = results_df.drop(columns=['peptide_feature', 'spec_feature'])
        return results_df

    def evaluate_results(self, results_df, colume = 'score'):
        # Calculate FDR and filter results
        fdr_utils = FDRUtils()
        recall_count, fdr_threshold = fdr_utils.calculate_fdr(results_df[colume], results_df['label'], self.FDR)
        fdr_ratio = recall_count / len([1 for item in results_df['label'] if item == 1])
        filtered_df = results_df[(results_df[colume] >= fdr_threshold) & (results_df['label'] == 1)].copy()
        return filtered_df

    def quant_precursor(self, data, topk=6):
        """
        计算单个样本的定量值，方法为：对每个fragment_chrom，结合rt间隔计算其面积（对rt维度做加权求和），
        取topk个fragment面积的均值作为该样本的定量值。

        参数:
            data: dict, fragment_chrom为[ion_num, rt_dim]的list, fragment_rt为[rt_dim]的list
            topk: int, 选取面积最大的topk个fragment

        返回:
            quant_result: 定量值（float）
        """
        fragment_chrom = np.array(data['frag']["chrom"])  # [ion_num, rt_dim]
        fragment_rt = np.array(data['frag']["RT"])        # [rt_dim]

        # 对fragment_rt做维度拓展，变为[1, rt_dim]，以便后续广播
        fragment_rt = np.expand_dims(fragment_rt, axis=0)  # [1, rt_dim]

        # 计算每个fragment的面积
        # 1. 计算rt间隔 [1, rt_dim-1]
        rt_interval = fragment_rt[:, 1:] - fragment_rt[:, :-1]  # [1, rt_dim-1]
        # 2. 取每个区间的信号强度，采用梯形法则近似面积
        chrom_left = fragment_chrom[:, :-1]   # [ion_num, rt_dim-1]
        chrom_right = fragment_chrom[:, 1:]   # [ion_num, rt_dim-1]
        # 3. 梯形法则面积
        fragment_area = np.sum((chrom_left + chrom_right) / 2 * rt_interval, axis=1)  # [ion_num]

        # 取面积最大的topk个fragment，若topk存在面积为0的，只考虑不为0的
        nonzero_areas = fragment_area[fragment_area > 0]
        if nonzero_areas.size == 0:
            quant_result = 0.0
        else:
            topk_num = min(topk, nonzero_areas.shape[0])
            topk_area = np.sort(nonzero_areas)[-topk_num:]  # 取最大的topk个非零面积
            quant_result = np.mean(topk_area)
        return float(quant_result)