import torch
import torch.nn as nn
import math

class BasicModel(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        if args.get_config('General', 'dtype') == 'float':
            self.dtype = torch.float32
        else:
            self.dtype = torch.half

    def load(self, file_path):
        """
        :param file_path: (str): 模型参数文件的路径
        """
        try:
            load_device = torch.device(self.args.get_config('General', 'device'))
            state_dict = torch.load(file_path, weights_only=True, map_location=load_device)
            self.load_state_dict(state_dict)
            self.to(self.dtype)
            print(f"load model from {file_path}, dtype: {self.dtype}")
        except Exception as e:
            print(f"error when loading model: {e}")

    def parameter_init(self):
        def custom_model_init(m):
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, mean=0, std=0.5)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)  # 将权重初始化为1
                nn.init.zeros_(m.bias)
        self.apply(custom_model_init)
    
    def freeze(self):
        for param in self.parameters():
            param.requires_grad = False

class MaskTransformer(nn.Module):
    def __init__(self, args, rotation=False):
        super().__init__()
        # self.device = args.get_config('General', 'device', default='cpu')
        if args.get_config('General', 'dtype', default='float') == 'float':
            self.dtype = torch.float32
        else:
            self.dtype = torch.half
        self.d_model = args.get_config('Model', 'd_model', default=128)
        self.n_head = args.get_config('Model', 'n_head', default=64)
        self.dim_feedforward = args.get_config('Model', 'hidden_layer', default=512)
        self.dropout = args.get_config('Model', 'dropout', default=0.20)
        self.rotation = rotation
        self.multiheadAttention = nn.MultiheadAttention(embed_dim=self.d_model, num_heads=self.n_head,
                                                        dropout=self.dropout, batch_first=True)
        self.feedforward_layer = nn.Sequential(
            nn.Linear(self.d_model, self.dim_feedforward),
            nn.ReLU(),
            nn.Linear(self.dim_feedforward, self.d_model)
        )
        self.norm_layer1 = nn.LayerNorm(self.d_model)
        self.norm_layer2 = nn.LayerNorm(self.d_model)
        self.q_proj = nn.Linear(self.d_model, self.d_model)
        self.k_proj = nn.Linear(self.d_model, self.d_model)
        self.v_proj = nn.Linear(self.d_model, self.d_model)

    def forward(self, x, attn_mask):
        """
        :param x: [batch, seq, d_model]
        :param attn_mask: [batch, seq, seq]
        :return: [batch, seq, d_model]
        """
        B, seq, d_model = x.shape
        query = self.q_proj(x)
        key = self.k_proj(x)
        value = self.v_proj(x)

        if self.rotation:
            query, key = self._apply_rotary_emb(query, key)

        # 重塑mask以适应多头注意力
        mask = attn_mask.unsqueeze(1).repeat(1, self.n_head, 1, 1).reshape(B * self.n_head, seq, seq)

        # 应用注意力机制
        attn_output, attn_weight = self.multiheadAttention(query, key, value, attn_mask=mask)

        # 残差连接和层归一化
        x = self.norm_layer1(x + attn_output)
        x = self.norm_layer2(x + self.feedforward_layer(x))
        return x, attn_mask

    def _apply_rotary_emb(self, xq: torch.Tensor, xk: torch.Tensor):
        """
        :param xq: [batch_size, seq_len, dim]
        :param xk: [batch_size, seq_len, dim]
        :return: [batch_size, seq_len, dim]
        """
        device = xq.device
        # xq_.shape = [batch_size, seq_len, dim // 2, 2]
        xq_ = xq.float().reshape(*xq.shape[:-1], -1, 2)
        xk_ = xk.float().reshape(*xk.shape[:-1], -1, 2)

        # 转为复数域
        xq_ = torch.view_as_complex(xq_)
        xk_ = torch.view_as_complex(xk_)

        # 计算旋转角度
        freqs_cis = self._precompute_freqs_cis(self.d_model, xq.shape[1], device=device)

        # 应用旋转操作，然后将结果转回实数域
        # xq_out.shape = [batch_size, seq_len, dim]
        xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(2)
        xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(2)
        return xq_out.to(self.dtype), xk_out.to(self.dtype)

    def _precompute_freqs_cis(self, dim: int, seq_len: int, device: torch.device, theta: float = 10000.0):
        # 计算词向量元素两两分组之后，每组元素对应的旋转角度
        freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim)).to(device)
        # 生成 token 序列索引 t = [0, 1,..., seq_len-1]
        t = torch.arange(seq_len, device=device, dtype=torch.float32)
        # freqs.shape = [seq_len, dim // 2]
        freqs = torch.outer(t, freqs).to(device)
        # 计算结果是个复数张量
        freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
        return freqs_cis


class GaussianKernel(nn.Module):
    def __init__(self, centers, sigma):
        """
        :param centers: Tensor[bin_size]
        :param sigma: Tensor(1)
        """
        super().__init__()
        self.register_buffer('centers', centers)
        self.register_buffer('sigma', sigma)

    def forward(self, x):
        """
        :param x: [batch, x, ion_num]
        :return feature: [batch, x, bin_size]
        """
        centers = self.centers.reshape(1, 1, 1, -1)  # [1, 1, 1, bin_size]
        x = x.unsqueeze(-1)  # [batch, x, ion_num, 1]
        dist = (x - centers) ** 2
        gaussian_features = torch.exp(-dist / (2 * self.sigma ** 2))
        gaussian_features = torch.sum(gaussian_features, dim=2, keepdim=False)
        gaussian_features = gaussian_features.to(x.dtype)
        return gaussian_features


if __name__ == "__main__":
    x = torch.randn(1, 1, 2)
    centers = torch.linspace(0, 2, 5)
    kernel = GaussianKernel(centers, 1.0)
    print(kernel(x))