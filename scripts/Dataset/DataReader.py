from math import ceil
import multiprocessing
import random
from tqdm import tqdm
import numpy as np
from typing import Callable, Dict, Iterable, List, Optional
from scipy.ndimage import gaussian_filter1d

# 配置日志
import logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def preprocess_fn(preprocessor):
    return preprocessor.preprocess()

def format_modified_sequence(sequence: str, modifications: dict[int, str]) -> str:
    """
    将序列和修饰信息格式化为带修饰标记的序列
    
    参数:
        sequence: 原始序列
        modifications: 修饰信息
        
    返回:
        带修饰标记的序列
    """
    result = ""
    for i, aa in enumerate(sequence):
        result += aa
        if i in modifications:
            result += f"({modifications[i]})"
            
    return result

class LightTokenizer:
    """
    A lightweight tokenizer similar to Transformer tokenizers.

    Features:
    1) Load vocab from a txt file (one token per line).
    2) Ensure [CLS] exists (and optionally other special tokens).
    3) Convert text/tokens <-> ids.
    """

    def __init__(
        self,
        vocab_path: Optional[str] = None,
        vocab_tokens: Optional[Iterable[str]] = None,
        cls_token: str = "[CLS]",
        unk_token: str = "[UNK]",
        pad_token: str = "[PAD]",
        sep_token: str = "[SEP]",
        mask_token: str = "[MASK]",
        tokenize_fn: Optional[Callable[[str], List[str]]] = None,
    ):
        self.cls_token = cls_token
        self.unk_token = unk_token
        self.pad_token = pad_token
        self.sep_token = sep_token
        self.mask_token = mask_token

        self._special_tokens = [unk_token, sep_token, pad_token, cls_token, mask_token]

        if vocab_tokens is not None:
            base_tokens = [str(t).strip() for t in vocab_tokens if str(t).strip()]
        elif vocab_path is not None:
            base_tokens = self.load_vocab(vocab_path)
        else:
            base_tokens = []

        # Keep order, remove duplicates.
        seen = set()
        ordered_tokens: List[str] = []
        for t in base_tokens:
            if t not in seen:
                ordered_tokens.append(t)
                seen.add(t)

        for st in self._special_tokens:
            if st not in seen:
                ordered_tokens.append(st)
                seen.add(st)

        self.id_to_token: List[str] = ordered_tokens
        self.token_to_id: Dict[str, int] = {t: i for i, t in enumerate(self.id_to_token)}

        # Default tokenize behavior: whitespace split; if no spaces then char split.
        if tokenize_fn is None:
            self.tokenize_fn = self._default_tokenize
        else:
            self.tokenize_fn = tokenize_fn

    @staticmethod
    def load_vocab(vocab_path: str) -> List[str]:
        with open(vocab_path, "r", encoding="utf-8") as f:
            tokens = [line.strip() for line in f if line.strip()]
        return tokens

    def _default_tokenize(self, text: str) -> List[str]:
        text = text.strip().lower()
        if not text:
            return []
        if ' ' in text:
            return text.split(' ')
        return [text]

    def add_token(self, token: str) -> int:
        token = token.strip()
        if not token:
            raise ValueError("token cannot be empty")
        if token in self.token_to_id:
            return self.token_to_id[token]
        new_id = len(self.id_to_token)
        self.id_to_token.append(token)
        self.token_to_id[token] = new_id
        return new_id

    def convert_tokens_to_ids(self, tokens: List[str]) -> List[int]:
        unk_id = self.token_to_id[self.unk_token]
        return [self.token_to_id.get(t, unk_id) for t in tokens]

    def encode(
        self,
        text: str,
        add_cls: bool = True,
        add_sep: bool = True
    ) -> List[int]:
        tokens = self.tokenize_fn(text)
        if add_cls:
            tokens = [self.cls_token] + tokens
        if add_sep:
            tokens = tokens + [self.sep_token]

        ids = self.convert_tokens_to_ids(tokens)

        return ids

class DataReader:
    def __init__(self, args, xic_datas: list):
        self.args = args
        self.split_ratio = [float(ratio) for ratio in args.get_config('Database', 'split_ratio', default='0.0, 0.0, 1.0').split(',')]
        self.shuffle = args.get_config('Database', 'shuffle', default=False)
        self.max_workers = args.get_config('General', 'threads', default=1)
        self.xic_datas = xic_datas

    def load_preprocess_data(self):
        logging.info(f"Loading {len(self.xic_datas)} XIC datas ...")

        logging.info(f"Preprocessing {len(self.xic_datas)} XIC datas")
        data = self._preprocess_data(self.xic_datas)

        if self.shuffle:
            logging.info(f"Shuffling data")
            random.shuffle(data)

        logging.info(f"Splitting data")
        train_data, val_data, test_data = self._split_data(data)
        return {
            'train_data': train_data,
            'val_data': val_data,
            'test_data': test_data
        }

    def _split_data(self, data):
        train_data = data[:int(len(data) * self.split_ratio[0])]
        val_data = data[int(len(data) * self.split_ratio[0]):int(len(data) * (self.split_ratio[0] + self.split_ratio[1]))]
        test_data = data[int(len(data) * (self.split_ratio[0] + self.split_ratio[1])):]
        return train_data, val_data, test_data

    def _preprocess_data(self, data: list):
        """使用多进程并行处理数据"""
        logging.info(f"Using multiprocessing to preprocess {len(data)} samples")
        chunk_size = ceil(len(data) / self.max_workers)
        data_chunks = [data[i:i+chunk_size] for i in range(0, len(data), chunk_size)]
        
        # 创建预处理器
        preprocessors = []
        for data_chunk in data_chunks:
            preprocessor = Preprocessor(self.args, data_chunk)
            preprocessors.append(preprocessor)
        
        processed_data = []
        with multiprocessing.Pool(processes=self.max_workers) as pool:
            results = list(tqdm(
                pool.imap(preprocess_fn, preprocessors), 
                total=len(preprocessors), 
                desc=f"processing data",
            ))
            
            # 收集处理结果
            for result in results:
                if result:
                    processed_data.extend(result)
        
        return processed_data

class Preprocessor:
    def __init__(self, args, data_chunk: list = None):
        self.RT_dim = args.get_config('Model', 'RT_dim', default=12)
        self.ion_num = args.get_config('Model', 'ion_num', default=10)
        self.peptide_tokenizer = LightTokenizer(vocab_path=args.get_config('Database', 'peptide_vocab_path', default='./Vocab/peptide_vocab.txt'))
        self.modification_tokenizer = LightTokenizer(vocab_path=args.get_config('Database', 'modification_vocab_path', default='./Vocab/modification_vocab.txt'))
        self.padding_token = args.get_config('Database', 'padding_token', default=31)
        self.preprocess_smooth = args.get_config('Database', 'preprocess_smooth', default=True)
        self.data_chunk = data_chunk

    def _interpolation(self, chrom: np.ndarray, ppm: np.ndarray, RT: np.ndarray, ref_RT_range: tuple[float, float]):
        if len(RT) == self.RT_dim:
            return chrom, ppm, RT

        RT_new = np.linspace(ref_RT_range[0], ref_RT_range[1], num = self.RT_dim)

        f = np.interp
        
        chrom_new = np.array([f(RT_new, RT, single_chrom) for single_chrom in chrom])
        ppm_new = np.array([f(RT_new, RT, single_ppm) for single_ppm in ppm])

        if self.preprocess_smooth:
            chrom_new = gaussian_filter1d(chrom_new, sigma=2, axis=1, mode='nearest')
            ppm_new = gaussian_filter1d(ppm_new, sigma=2, axis=1, mode='nearest')

        chrom_new[chrom_new < 0] = 0.0
        ppm_new[ppm_new < 0] = 0.0

        return chrom_new, ppm_new, RT_new

    def _preprocess_ion(self, chrom: np.ndarray, ppm: np.ndarray, mz: np.ndarray, ion_num: int):
        if len(chrom) == ion_num and len(mz) == ion_num:
            return chrom, ppm, mz

        if not len(chrom) == len(mz):
            raise ValueError(f'error: length of chrom is not same with mz')                                
        
        if len(chrom) > ion_num:
            return chrom[:ion_num, :], ppm[:ion_num, :], mz[:ion_num]

        if len(chrom) < ion_num:
            padding_count = ion_num - len(chrom)
            padding_chrom = np.zeros((padding_count, chrom.shape[1]))
            chrom = np.append(chrom, padding_chrom, axis=0)
            padding_ppm = np.zeros((padding_count, ppm.shape[1]))
            ppm = np.append(ppm, padding_ppm, axis=0)
            mz = np.append(mz, np.zeros(padding_count))
            return chrom, ppm, mz
    
    def _normalize(self, chrom: np.ndarray, channel_norm: bool = False):
        if channel_norm:
            max_values = np.max(chrom, axis=1, keepdims=True)
            normalized_chrom = chrom / (max_values + 1e-5)  # [peptide_num/ion_num, RT_dim]
        else:
            max_value = np.max(chrom)
            normalized_chrom = chrom / (max_value + 1e-5)  # [peptide_num/ion_num, RT_dim]
        
        return normalized_chrom

    def _preprocess_peptide(self, sequence: str, modification: dict):
        peptide = ' '.join(sequence)
        peptide_ids = self.peptide_tokenizer.encode(peptide, add_cls=True, add_sep=True)
        modification_str = [value for key, value in modification.items()]
        ids_tmp = self.modification_tokenizer.encode(' '.join(modification_str), add_cls=False, add_sep=False)
        modification_ids = [0] * len(peptide_ids)
        for index, (key, value) in enumerate(modification.items()):
            modification_ids[key + 1] = ids_tmp[index]
        return peptide_ids, modification_ids

    def preprocess(self):
        results = []
        for data in self.data_chunk:
            peptide = data['pre']['peptide']
            modification = data['pre']['modification']
            charge = data['pre']['charge']
            modified_peptide = format_modified_sequence(peptide, modification)
            peptide_ids, modification_ids = self._preprocess_peptide(peptide, modification)
            label = 1 if data['label'] == 1 else 0

            precursor_chrom = np.array(data['pre']['chrom'])
            fragment_chrom = np.array(data['frag']['chrom'])
            precursor_ppm = np.array(data['pre']['ppm'])
            fragment_ppm = np.array(data['frag']['ppm'])
            precursor_mz = np.array(data['pre']['mz'])
            fragment_mz = np.array(data['frag']['mz'])
            precursor_RT = np.array(data['pre']['RT'])
            fragment_RT = np.array(data['frag']['RT'])

            if len(precursor_RT) == 0 or len(fragment_RT) == 0:
                logging.warning(f"precursor_RT or fragment_RT is empty, skip this sample")
                continue
            
            global_RT_range = (min(min(precursor_RT), min(fragment_RT)), max(max(precursor_RT), max(fragment_RT)))
            precursor_chrom, precursor_ppm, precursor_RT = self._interpolation(chrom=precursor_chrom, ppm=precursor_ppm, RT=precursor_RT, ref_RT_range=global_RT_range)
            precursor_chrom = self._normalize(precursor_chrom)
            fragment_chrom, fragment_ppm, fragment_RT = self._interpolation(chrom=fragment_chrom, ppm=fragment_ppm, RT=fragment_RT, ref_RT_range=global_RT_range)
            fragment_chrom = self._normalize(fragment_chrom)

            precursor_chrom, precursor_ppm, precursor_mz = self._preprocess_ion(chrom=precursor_chrom, ppm=precursor_ppm, mz=precursor_mz, ion_num= 4)
            fragment_chrom, fragment_ppm, fragment_mz = self._preprocess_ion(chrom=fragment_chrom, ppm=fragment_ppm, mz=fragment_mz, ion_num=self.ion_num)

            results.append({
                'precursor_chrom': precursor_chrom,
                'precursor_ppm': precursor_ppm,
                'precursor_mz': precursor_mz,
                'precursor_RT': precursor_RT,
                'peptide_ids': peptide_ids,
                'modification_ids': modification_ids,
                'fragment_chrom': fragment_chrom,
                'charge': charge,
                'fragment_ppm': fragment_ppm,
                'fragment_mz': fragment_mz,
                'fragment_RT': fragment_RT,
                'label': label,
                'modified_peptide': modified_peptide,
            })
        return results