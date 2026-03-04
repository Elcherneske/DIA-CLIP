import pandas as pd
import re
from typing import List

class DIANNOutReader:
    """DIA-NN out.tsv 结果读取和解析类"""
    def __init__(self):
        self.df = None
    
    def read(self, file_path: str):
        """读取DIA-NN结果文件并处理数据"""
        df = pd.read_csv(file_path, sep='\t')
        
        # 只保留需要的列
        needed_columns = ['Modified.Sequence', 'Precursor.Charge', 'Precursor.Mz', 'RT.Start', 'RT', 'RT.Stop', 'Fragment.Info']
        if all(col in df.columns for col in needed_columns):
            self.df = df[needed_columns].copy()
            
            # 重命名列
            self.df = self.df.rename(columns={
                'Modified.Sequence': 'peptide',
                'Precursor.Charge': 'charge',
                'Precursor.Mz': 'precursor_mz',
                'RT.Start': 'rt_start',
                'RT': 'rt',
                'RT.Stop': 'rt_stop',
                'Fragment.Info': 'fragment_info'
            })
            
            # 解析碎片离子信息
            self.df['fragment_mz'] = self.df['fragment_info'].apply(self._parse_fragment_info)
            
            # 添加标签列（这里假设所有肽段都是目标肽段）
            self.df['label'] = 1
            
            # 选择最终需要的列
            final_columns = ['label', 'peptide', 'charge', 'precursor_mz', 'rt_start', 'rt', 'rt_stop', 'fragment_mz']
            self.df = self.df[final_columns]
            
            return self.df
        else:
            raise ValueError(f"输入文件缺少必要的列: {[col for col in needed_columns if col not in df.columns]}")
    
    def _parse_fragment_info(self, fragment_info: str) -> str:
        """解析碎片离子信息，提取mz值"""
        if pd.isna(fragment_info):
            return ''
            
        fragments = fragment_info.split(';')
        mz_values = []
        for fragment in fragments:
            if not fragment.strip():
                continue
            mz = fragment.split('/')[1]
            mz_values.append(mz)
        return ','.join(mz_values)
    
    def get_all_peptide_info(self) -> pd.DataFrame:
        """获取所有肽段信息"""
        if self.df is None:
            raise ValueError("请先调用read方法读取数据")
        return self.df

if __name__ == "__main__":
    reader = DIANNOutReader()
    df = reader.read("out.tsv")
    print(reader.get_all_peptide_info())
