import torch
import torch.nn as nn
from .CommonModel import BasicModel

class ClipDecodingV2(BasicModel):
    def __init__(self, args):
        super().__init__(args)

        self.ion_num = args.get_config('Model', 'ion_num', default=10)
        self.d_model = args.get_config('Model', 'd_model', default=128)
        self.n_head = args.get_config('Model', 'n_head', default=64)
        self.hidden_layer = args.get_config('Model', 'hidden_layer', default=512)
        self.dropout = args.get_config('Model', 'dropout', default=0.15)
        self.decoding_layer_num = args.get_config('Model', 'decoding_layer_num', default=2)
        self.feature_diff = args.get_config('Model', 'feature_diff', default=False)

        # Transformer decoder
        self.transformer_decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(
                d_model=self.d_model,
                nhead=self.n_head,
                dim_feedforward=self.hidden_layer,
                dropout=self.dropout,
                batch_first=True
            ),
            num_layers=self.decoding_layer_num
        )

        # Final projection layer to produce score
        self.project_layer = nn.Sequential(
            nn.Linear(2 * self.d_model, self.d_model // 2),
            nn.GELU(),
            nn.LayerNorm(self.d_model // 2),
            nn.Dropout(self.dropout),
            nn.Linear(self.d_model // 2, 1),
            nn.Sigmoid()
        )

        self.corr_mlp_layer = nn.Sequential(
            nn.Linear((4 + self.ion_num)**2, self.hidden_layer),
            nn.GELU(),
            nn.LayerNorm(self.hidden_layer),
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_layer, self.hidden_layer),
            nn.GELU(),
            nn.LayerNorm(self.hidden_layer),
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_layer, self.d_model),
            nn.GELU(),
            nn.LayerNorm(self.d_model),
            nn.Dropout(self.dropout),
        )
    
    def pearson_correlation(self, precursor_chrom, fragment_chrom):
        """
        :param precursor_chrom: [batch, ion_num, RT_dim]
        :param fragment_chrom: [batch, ion_num, RT_dim]
        :return: [batch, ion_num, ion_num]
        """
        def compute_corr_for_single_batch(x):
            return torch.corrcoef(x)
        B, _, _ = precursor_chrom.shape
        chrom = torch.cat([precursor_chrom, fragment_chrom], dim=1)
        all_correlation_matrices_vmap = torch.vmap(compute_corr_for_single_batch, in_dims=0, out_dims=0)(chrom)
        all_correlation_matrices_vmap = all_correlation_matrices_vmap.reshape(B, 1, -1)
        return all_correlation_matrices_vmap

    def forward(self, peptide_feature, spec_feature, data):
        """
        Args:
            peptide_feature: Tensor of shape [batch, 1, d_model] - query
            spec_feature: Tensor of shape [batch, 1, d_model] - memory
        Returns:
            score: Tensor of shape [batch] with values between 0 and 1
        """
        # Apply transformer decoder
        # peptide_feature is the query (tgt), spec_feature is the memory (src)
        peptide_feature = peptide_feature.unsqueeze(1)
        spec_feature = spec_feature.unsqueeze(1)
        if self.feature_diff:
            decoded_feature = self.transformer_decoder(
                tgt=peptide_feature - spec_feature,
                memory=peptide_feature - spec_feature
            )
        else:
            decoded_feature = self.transformer_decoder(
                tgt=peptide_feature,
                memory=spec_feature
            )

        precursor_chrom = data["precursor_chrom"]
        precursor_mz = data["precursor_mz"]
        frag_chrom = data["fragment_chrom"]
        frag_mz = data["fragment_mz"]
        corr_feature = self.pearson_correlation(precursor_chrom, frag_chrom) #[batch, 1, ion_num * ion_num]
        corr_feature = torch.nan_to_num(corr_feature, nan=0.0)
        # mz = torch.cat([precursor_mz, frag_mz], dim=-1)
        # mz = (mz != 0).to(mz.dtype)
        # no_empty_num = torch.sum(mz, dim=-1, keepdim=True).reshape(mz.shape[0], 1, 1) #[batch, 1, 1]
        corr_feature = self.corr_mlp_layer(corr_feature)

        feature = torch.cat([decoded_feature, corr_feature], dim=-1)

        # Project to score (0-1)
        score = self.project_layer(feature.squeeze(1)).squeeze(-1)

        return {"score": score}
