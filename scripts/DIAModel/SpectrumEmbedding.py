import torch
import torch.nn as nn
from .CommonModel import MaskTransformer, GaussianKernel, BasicModel

class SpectrumEmbeddingV2(nn.Module):
    def __init__(
            self,
            args,
            num_layer=8,
    ):
        super().__init__()
        self.chrom_embedding = ChromatogramEmbeddingV2(
            args=args,
            num_layer=num_layer
        )

    def forward(self, spec, ppm, mz, RT):
        """
        :param spec: [batch, ion_num, RT_dim]
        :param ppm: [batch, ion_num, RT_dim]
        :param RT: [batch, RT_dim]
        :param mz: [batch, ion_num]
        :return: [batch, 1, d_model]
        """
        chrom_feature = self.chrom_embedding(chrom=spec, ppm=ppm, RT=RT, mz=mz)
        feature = chrom_feature[:, :1, :]
        return feature


class ChromatogramEmbeddingV2(BasicModel):
    def __init__(
            self,
            args,
            num_layer = 8,
    ):
        super().__init__(args)

        self.d_model = args.get_config('Model', 'd_model', default=128)
        self.RT_dim = args.get_config('Model', 'RT_dim', default=12) 
        self.n_head = args.get_config('Model', 'n_head', default=64)
        self.hidden_layer = args.get_config('Model', 'hidden_layer', default=512)
        self.dropout = args.get_config('Model', 'dropout', default=0.15)
        self.ts_encoding = nn.ModuleList([MaskTransformer(args=args, rotation=False) for _ in range(num_layer)])
        self.mz_embedding = MZBinEmbeddingV2(args=args, num_layer=args.get_config('Model', 'mz_bin_layer_num', default=2))
        self.chrom_proj = nn.Sequential(
            nn.Linear(self.RT_dim, self.hidden_layer),
            nn.GELU(),
            nn.LayerNorm(self.hidden_layer),
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_layer, self.d_model),
            nn.GELU(),
            nn.LayerNorm(self.d_model),
            nn.Dropout(self.dropout),
        )
        self.ppm_proj = nn.Sequential(
            nn.Linear(self.RT_dim, self.hidden_layer),
            nn.GELU(),
            nn.LayerNorm(self.hidden_layer),
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_layer, self.d_model),
            nn.GELU(),
            nn.LayerNorm(self.d_model),
            nn.Dropout(self.dropout),
        )
        self.layer_norm = nn.LayerNorm(self.d_model)

    def form_mask(self, mz):
        """
        :param mz: [Batch, ion_num]
        :return: [Batch, ion_num, ion_num]
        """
        B, ion_num = mz.shape
        device = mz.device
        mask = (mz == 0).to(device)  # [B, ion_num]
        mask = mask.unsqueeze(-1) | mask.unsqueeze(1)  # [B, ion_num, ion_num]
        mask[torch.eye(ion_num).unsqueeze(0).repeat(B, 1, 1) == 1] = False
        return mask

    def norm_chrom(self, chrom):
        """
        :param chrom: [batch, ion_num, RT_dim]
        :return: [batch, ion_num, RT_dim]
        """
        return chrom - chrom.mean(dim = -1, keepdim=True)

    def forward(self, chrom, ppm, mz, RT):
        """
        :param chrom: [batch, ion_num, RT_dim]
        :param ppm: [batch, ion_num, RT_dim]
        :param RT: [batch, RT_dim]
        :param mz: [batch, ion_num]
        :return: [batch, ion_num, d_model]
        """
        B, ion_num, RT_dim = chrom.shape
        device = chrom.device
        mask = self.form_mask(mz)
        pos_mask = (mz > 0).to(device) # [B, ion_num]
        chrom = self.norm_chrom(chrom)
        chrom_feature = self.chrom_proj(chrom) #[batch, ion_num, d_model]
        ppm_feature = self.ppm_proj(ppm) #[batch, ion_num, d_model]
        mz_emb = self.mz_embedding(spec=chrom, mz=mz, RT=RT) #[batch, ion_num, d_model]
        feature = chrom_feature + ppm_feature + mz_emb

        origin_feature = feature
        for layer in self.ts_encoding:
            feature, mask = layer(feature, mask)

        # Calculate average of features where pos_mask is not 0
        origin_feature_sum = torch.sum(origin_feature * pos_mask.unsqueeze(-1), dim=1, keepdim=False)  # [B, d_model]
        origin_feature_count = torch.sum(pos_mask, dim=1, keepdim=True).to(device).to(self.dtype)  # [B, 1]
        origin_feature_count = torch.clamp(origin_feature_count, min=1.0)
        avg_origin_feature = origin_feature_sum / origin_feature_count #[B, d_model]
        feature = self.layer_norm(feature + avg_origin_feature.unsqueeze(1))
        return feature


class MZBinEmbeddingV2(BasicModel):
    def __init__(
            self,
            args,
            num_layer = 8,
    ):
        super().__init__(args)
        self.d_model = args.get_config('Model', 'd_model', default=128)
        self.n_head = args.get_config('Model', 'n_head', default=64)
        self.hidden_layer = args.get_config('Model', 'hidden_layer', default=512)
        self.dropout = args.get_config('Model', 'dropout', default=0.15)
        self.bin_size = args.get_config('Model', 'bin_size', default=2500)    
        self.max_mz_range = args.get_config('Model', 'max_mz_range', default=250)
        self.ts_encoding = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=self.d_model, 
                nhead=self.n_head, 
                batch_first=True, 
                dropout=self.dropout, 
                dim_feedforward=self.hidden_layer
            ), 
            num_layers=num_layer
        )
        self.mz_bin_proj = nn.Sequential(
            nn.Linear(self.bin_size, self.hidden_layer),
            nn.GELU(),
            nn.LayerNorm(self.hidden_layer),
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_layer, self.d_model),
            nn.GELU(),
            nn.LayerNorm(self.d_model),
            nn.Dropout(self.dropout),
        )
        self.layer_norm = nn.LayerNorm(self.d_model)
        self.mz_gaussian_kernel = GaussianKernel(centers=torch.linspace(0.0, self.max_mz_range, self.bin_size), sigma=torch.tensor(2 * self.max_mz_range/self.bin_size, dtype=self.dtype))

    def forward(self, spec, mz, RT):
        """
        :param spec: [batch, ion_num, RT_dim]
        :param RT: [batch, RT_dim]
        :param mz: [batch, ion_num]
        :return: [batch, RT_dim, d_model]
        """
        B, ion_num, RT_dim = spec.shape
        mz = mz.unsqueeze(-1) # [batch, ion_num, 1]
        mz_feature = self.mz_gaussian_kernel(mz) #[batch, ion_num, bin_size]
        mz_feature = self.mz_bin_proj(mz_feature) #[batch, ion_num, d_model]
        feature = self.layer_norm(mz_feature + self.ts_encoding(mz_feature))
        return feature