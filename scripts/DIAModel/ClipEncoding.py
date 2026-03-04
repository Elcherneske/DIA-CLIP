import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from .SpectrumEmbedding import SpectrumEmbeddingV2
from .PeptideEmbedding import PeptideEmbeddingV2
from .CommonModel import BasicModel

class ClipEncodingV3(BasicModel):
    def __init__(self, args):
        super().__init__(args)

        self.d_model = args.get_config('Model', 'd_model', default=128)
        self.n_head = args.get_config('Model', 'n_head', default=64)
        self.dim_feedforward = args.get_config('Model', 'hidden_layer', default=512)
        self.dropout = args.get_config('Model', 'dropout', default=0.15)
        self.peptide_layer_num = args.get_config('Model', 'peptide_encoding_layer_num', default=2)
        self.precursor_layer_num = args.get_config('Model', 'precursor_encoding_layer_num', default=2)
        self.fragment_layer_num = args.get_config('Model', 'fragment_encoding_layer_num', default=2)
        self.cross_layer_num = args.get_config('Model', 'cross_encoding_layer_num', default=2)
        self.ts_layer_num = args.get_config('Model', 'chrom_encoding_layer_num', default=2)

        self.peptide_embedding = PeptideEmbeddingV2(args=args, num_layer=self.peptide_layer_num)
        self.precursor_embedding = SpectrumEmbeddingV2(args=args, num_layer=self.precursor_layer_num)
        self.fragment_embedding = SpectrumEmbeddingV2(args=args, num_layer=self.fragment_layer_num)
        self.cross_encoding = SpectrumEmbeddingV2(args=args, num_layer=self.cross_layer_num)
        self.ts_encoding = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=self.d_model,
                nhead=self.n_head,
                batch_first=True,
                dropout=self.dropout,
                dim_feedforward=self.dim_feedforward
            ),
            num_layers=self.ts_layer_num
        )
        self.layer_norm = nn.LayerNorm(self.d_model)

    def forward(self, data):
        peptide_feature = self.peptide_forward(data=data)
        spec_feature = self.spec_forward(data=data)

        peptide_feature = peptide_feature.squeeze(1)
        spec_feature = spec_feature.squeeze(1)

        return {"peptide_feature": peptide_feature, "spec_feature": spec_feature}

    def peptide_forward(self, data):
        peptides = data["peptide_ids"]
        modifications = data["modification_ids"]
        charges = data["charge"]
        feature = self.peptide_embedding(peptides, modifications, charges) # after layer norm
        feature = F.normalize(feature, dim=-1, p=2)
        return feature

    def spec_forward(self, data):
        precursor_chrom = data["precursor_chrom"]
        precursor_mz = data["precursor_mz"]
        precursor_RT = data["precursor_RT"]
        precursor_ppm = data["precursor_ppm"]
        frag_chrom = data["fragment_chrom"]
        frag_mz = data["fragment_mz"]
        frag_RT = data["fragment_RT"]
        frag_ppm = data["fragment_ppm"]

        precursor_feature = self.precursor_embedding(spec=precursor_chrom, ppm=precursor_ppm, RT=precursor_RT, mz=precursor_mz)
        fragment_feature = self.fragment_embedding(spec=frag_chrom, ppm=frag_ppm, RT=frag_RT, mz=frag_mz)

        combine_spec = torch.cat([precursor_chrom, frag_chrom], dim=1)
        combine_ppm = torch.cat([precursor_ppm, frag_ppm], dim=1)
        combine_mz = torch.cat([precursor_mz, frag_mz], dim=1)
        combine_RT = torch.cat([precursor_RT, frag_RT], dim=1)
        cross_feature = self.cross_encoding(spec=combine_spec, ppm=combine_ppm, RT=combine_RT, mz=combine_mz)

        feature = torch.cat([precursor_feature, fragment_feature, cross_feature], dim=1)
        feature = self.layer_norm(torch.mean(feature, dim=1, keepdim=True) + self.ts_encoding(feature)[:, :1, :])
        feature = F.normalize(feature, dim=-1, p=2)
        return feature