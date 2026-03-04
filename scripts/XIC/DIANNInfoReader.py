import pandas as pd
import re
from typing import List

class DIANNInfoReader:
    """DIA-NN info.tsv 结果读取和解析类"""
    def __init__(self):
        self.df = None
    
    def read(self, file_path: str):
        """读取DIA-NN结果文件并处理数据"""
        df = pd.read_csv(file_path, sep='\t')
        
        # 只保留需要的列
        needed_columns = ['decoy', 'pr_id', 'pr_mz', 'rt_start', 'rt', 'rt_stop', 'fg_mz']
        if all(col in df.columns for col in needed_columns):
            self.df = df[needed_columns].copy()
            
            # 解析肽段ID，提取序列和电荷
            peptide_info = [self.parse_peptide_id(pid) for pid in self.df['pr_id']]
            self.df['peptide'] = [info[0] for info in peptide_info]
            self.df['charge'] = [info[1] for info in peptide_info]
            
            # 将fragment_mz列转换为逗号分隔的字符串
            self.df['fragment_mz'] = self.df['fg_mz'].apply(lambda x: ','.join(x.split(';')))
            
            # 将 decoy 转化为 label
            self.df['label'] = self.df['decoy'].apply(lambda x: 1 if int(x) == 0 else 0)

            # 重命名列
            self.df = self.df.rename(columns={'pr_mz': 'precursor_mz'})
            
            # 选择最终需要的列
            final_columns = ['label', 'peptide', 'charge', 'precursor_mz', 'rt_start', 'rt', 'rt_stop', 'fragment_mz']
            self.df = self.df[final_columns]
            
            return self.df
        else:
            raise ValueError(f"输入文件缺少必要的列: {[col for col in needed_columns if col not in df.columns]}")
    
    def parse_peptide_id(self, peptide_id: str) -> tuple:
        """解析肽段ID，提取序列和电荷"""
        # 假设格式为 "序列+电荷"，如 "AAAAAAAAAAAAAAAGAGAGAK3"
        charge_str = ''
        for char in peptide_id[::-1]:
            if char.isdigit():
                charge_str += char
            else:
                break
        charge = int(charge_str[::-1])
        sequence = peptide_id[:-len(charge_str)]
        return sequence, charge
    
    def get_all_peptide_info(self) -> pd.DataFrame:
        """获取所有肽段信息"""
        if self.df is None:
            raise ValueError("请先调用read方法读取数据")
        return self.df
    

if __name__ == "__main__":
    reader = DIANNInfoReader()
    reader.read("diann_info.tsv")
    print(reader.get_all_peptide_info())

