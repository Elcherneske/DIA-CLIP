import torch.nn as nn
import torch
from .ClipEncoding import ClipEncodingV3
from .ClipDecoding import ClipDecodingV2
from .CommonModel import BasicModel

class DIAClipV3(BasicModel):
    def __init__(self, args):
        super().__init__(args)
        self.encoding_model = ClipEncodingV3(args=args)
        self.decoding_model = ClipDecodingV2(args=args)

    def forward(self, data):
        encoding_result = self.encoding_model(data)
        peptide_feature = encoding_result["peptide_feature"]
        spec_feature = encoding_result["spec_feature"]
        decoding_result = self.decoding_model(peptide_feature, spec_feature, data)
        score = decoding_result["score"]
        return {"peptide_feature": peptide_feature, "spec_feature": spec_feature, "score": score}