import torch
import torch.nn as nn
from .CommonModel import MaskTransformer

class PeptideEmbeddingV2(nn.Module):
    def __init__(
            self,
            args,
            num_layer=8,
    ):
        super().__init__()
        self.model = args.get_config('Model', 'model_peptide', default="trans")

        if self.model == "trans":
            self.peptide_embedding = PeptideTransformerEmbeddingV2(
                args=args,
                num_layer=num_layer
            )
        else:
            raise ValueError(f"Invalid model: {self.model}")

    def forward(self, peptides, modifications, charges):
        """
        :param peptides: [Batch, seq]
        :param modifications: [Batch, seq]
        :param charges: [Batch]
        :return: [batch, 1, d_model]
        """
        if self.model == 'trans':
            peptide_feature = self.peptide_embedding(peptides, modifications, charges)
        else:
            raise ValueError(f"Invalid model: {self.model}")
        peptide_feature = peptide_feature[:, :1, :]
        return peptide_feature


class PeptideTransformerEmbeddingV2(nn.Module):
    def __init__(
            self,
            args,
            num_layer=8,
    ):
        super().__init__()
        if args.get_config('General', 'dtype', default='float') == 'float':
            self.dtype = torch.float32
        else:
            self.dtype = torch.half

        self.d_model = args.get_config('Model', 'd_model', default=128)
        self.n_head = args.get_config('Model', 'n_head', default=64)
        self.dim_feedforward = args.get_config('Model', 'hidden_layer', default=512)
        self.dropout = args.get_config('Model', 'dropout', default=0.15)
        self.max_charge = args.get_config('Model', 'max_charge', default=10)
        self.max_modification = args.get_config('Model', 'max_modification', default=16)
        self.padding_token = args.get_config('Database', 'padding_token', default=31)

        self.word_emb = nn.Embedding(num_embeddings=32, embedding_dim=self.d_model)
        self.modification_emb = nn.Embedding(num_embeddings=self.max_modification, embedding_dim=self.d_model)
        self.charge_emb = nn.Embedding(num_embeddings=self.max_charge, embedding_dim=self.d_model)
        self.ts_encoding = nn.ModuleList([MaskTransformer(args=args, rotation=True) for _ in range(num_layer)])
        self.norm_layer = nn.LayerNorm(self.d_model)

    def modification_embedding(self, modifications):
        """
        :param modifications: [Batch, seq]
        :return: [batch, seq, d_model]
        """
        modification_feature = self.modification_emb(modifications)
        return modification_feature
    
    def charge_embedding(self, charges):
        """
        :param charges: [Batch]
        :return: [batch, 1, d_model]
        """
        assert charges.max() < self.max_charge, f"Input charges exceed embedding size: {charges.max()}"
        assert charges.min() >= 0, f"Input charges must be non-negative: {charges.min()}"

        charge_feature = self.charge_emb(charges)
        charge_feature = charge_feature.unsqueeze(1)
        return charge_feature

    def word_embedding(self, peptide_ids):
        """
        :param peptide_ids: [Batch, seq]
        :return: [batch, seq, d_model]
        """
        assert peptide_ids.max() < 32, f"Input indices exceed embedding size: {peptide_ids.max()}"
        assert peptide_ids.min() >= 0, f"Input indices must be non-negative: {peptide_ids.min()}"

        B, seq = peptide_ids.shape
        device = peptide_ids.device
        mask = (peptide_ids == self.padding_token).to(device)  # [B, seq]
        mask = mask.unsqueeze(-1) | mask.unsqueeze(1)  # [B, seq, seq]
        mask[torch.eye(seq).unsqueeze(0).repeat(B, 1, 1) == 1] = False
        word_feature = self.word_emb(peptide_ids)
        return word_feature, mask

    def forward(self, peptides, modifications, charges):
        """
        :param peptides: [Batch, seq]
        :param modifications: [Batch, seq]
        :param charges: [Batch]
        :return: [batch, seq, d_model]
        """

        word_emb, mask = self.word_embedding(peptides)
        modification_emb = self.modification_embedding(modifications)
        charge_emb = self.charge_embedding(charges)
        feature = word_emb + modification_emb + charge_emb
        origin_feature = feature
        for layer in self.ts_encoding:
            feature, mask = layer(feature, mask)
        feature = self.norm_layer(feature + origin_feature)
        return feature


if __name__ == "__main__":
    pass

