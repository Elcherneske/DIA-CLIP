import os
import pickle
from tqdm import tqdm
import numpy as np
from MolecularUtils import ModificationUtils
from .SpectraUtils import XICExtractor
from .DIANNInfoReader import DIANNInfoReader
from .DIANNOutReader import DIANNOutReader

import logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

class XICManager:
    def __init__(self, args, mzml_path):
        self.args = args
        self.out_dir = args.get_config('General', 'out_dir')
        self.meta_file_path = os.path.join(self.out_dir, 'diann', 'diann_info.tsv')
        self.mzml_path = mzml_path
        self.mzml_name = os.path.basename(mzml_path).replace('.mzML', '')
        self.threads = args.get_config('General', 'threads', default=1)
        self.filter = args.get_config('XIC Parameter', 'filter', default='all')
        self.ppm_threadhold = args.get_config('XIC Parameter', 'ppm_threadhold', default=30)

    def process(self):
        if not os.path.exists(self.out_dir):
            raise ValueError(f"Output directory {self.out_dir} does not exist")
        if not os.path.exists(self.meta_file_path):
            raise ValueError(f"Meta file {self.meta_file_path} does not exist")
        if not os.path.exists(self.mzml_path):
            raise ValueError(f"MzML file {self.mzml_path} does not exist")
        
        reader = DIANNInfoReader()
        reader.read(self.meta_file_path)
        peptide_infos = reader.get_all_peptide_info()
        logging.info(f"read {len(peptide_infos)} results from {self.meta_file_path}")

        xic_extractor = XICExtractor(num_threads=self.threads, mode='rt_range')

        xics = xic_extractor.extract_xics(self.mzml_path, peptide_infos)
        
        datas = []
        for id, (index, peptide_info) in tqdm(enumerate(peptide_infos.iterrows()), total=len(peptide_infos), desc="Processing peptides"):
            precursor_xics, fragment_xics = xics[id]
            modified_peptide = peptide_info['peptide']
            sequence, modification = ModificationUtils.parse_modified_sequence(modified_peptide)
            modification = self._convert_uniimod_to_name(modification)
                    
            datas.append({
                'pre': {
                    'chrom': [xic.intensity_array for xic in precursor_xics],
                    'ppm': [xic.ppm_array for xic in precursor_xics],
                    'RT': precursor_xics[0].rt_array,
                    'mz': [precursor_xic.mz for precursor_xic in precursor_xics],
                    'peptide': sequence,
                    'charge': peptide_info['charge'],
                    'modification': modification,
                },
                'frag': {
                    'chrom': [xic.intensity_array for xic in fragment_xics],
                    'ppm': [xic.ppm_array for xic in fragment_xics],
                    'RT': fragment_xics[0].rt_array,
                    'mz': [fragment_xic.mz for fragment_xic in fragment_xics]
                },
                'label': int(peptide_info['label'])
            })
        
        # pkl_path = os.path.join(self.out_dir, f"{self.mzml_name}.info.pkl")
        # with open(pkl_path, 'wb') as f:
        #     pickle.dump(datas, f)

        logging.info(f"XIC Extraction completed, {len(datas)} peptides extracted")
        return datas

    def _convert_uniimod_to_name(self, modification: dict) -> dict:
        if not modification == {}:
            for key, value in modification.items():
                if value == 'UniMod:4':
                    modification[key] = 'Carbamidomethyl'
                elif value == 'UniMod:21':
                    modification[key] = 'Phosphorylation'
                elif value == 'UniMod:35' or value == 'Oxidation (P)':
                    modification[key] = 'Oxidation'
                elif value == 'UniMod:27':
                    modification[key] = 'Pyroglu'
                elif value == 'UniMod:26':
                    modification[key] = 'Pyrocarbamidomethyl'
                elif value == 'UniMod:7':
                    modification[key] = 'Deamidation'
                elif value == 'UniMod:1':
                    modification[key] = 'Acetyl'
                else:
                    raise ValueError(f"未知修饰: {value}")
        return modification