# model.py

import torch
import torch.nn as nn
import math
from typing import Optional

class PositionalEncoding(nn.Module):
    """
    位置编码模块。由于Transformer没有循环结构，我们需要为输入序列注入位置信息。
    这里使用sin和cos函数的组合来为每个位置创建一个独特的、可学习相对位置的编码。
    """
    # 显式声明 pe 是一个 Tensor，以帮助类型检查器
    pe: torch.Tensor

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        # 创建一个足够长的位置编码矩阵
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)

        # 偶数维度使用sin，奇数维度使用cos
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        
        # 注册为buffer，它不是模型参数，但会随模型移动(如.to(device))
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, d_model]
        """
        # 将位置编码加到输入的词嵌入上
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)

class MultiHeadAttention(nn.Module):
    """
    多头注意力模块。它允许模型同时从不同位置、不同表示子空间关注信息。
    这比单一注意力机制更强大。
    """
    def __init__(self, d_model: int, nhead: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % nhead == 0, "d_model must be divisible by nhead"
        
        self.d_model = d_model
        self.nhead = nhead
        self.head_dim = d_model // nhead
        
        # 定义Q, K, V和输出的线性变换层
        self.fc_q = nn.Linear(d_model, d_model)
        self.fc_k = nn.Linear(d_model, d_model)
        self.fc_v = nn.Linear(d_model, d_model)
        self.fc_o = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(p=dropout)
        
    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # query, key, value shape: [seq_len, batch_size, d_model]
        batch_size = query.shape[1]
        
        # 1. 线性变换并分割成多头
        # [seq_len, batch_size, d_model] -> [batch_size, nhead, seq_len, head_dim]
        Q = self.fc_q(query).view(query.shape[0], batch_size, self.nhead, self.head_dim).permute(1, 2, 0, 3)
        K = self.fc_k(key).view(key.shape[0], batch_size, self.nhead, self.head_dim).permute(1, 2, 0, 3)
        V = self.fc_v(value).view(value.shape[0], batch_size, self.nhead, self.head_dim).permute(1, 2, 0, 3)
        
        # 2. 计算注意力分数
        # energy shape: [batch_size, nhead, query_len, key_len]
        energy = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)
        
        if mask is not None:
            # 这里的mask处理需要非常小心
            # causal_mask [query_len, key_len] 会被广播到 [1, 1, query_len, key_len]
            # padding_mask [batch_size, key_len] 会被广播到 [batch_size, 1, 1, key_len]
            # 当两者结合时，需要确保它们的形状兼容
            energy = energy.masked_fill(mask == 1, -1e10) # 注意：我们将使用1代表要屏蔽的位置
            
        attention = torch.softmax(energy, dim=-1)
        attention = self.dropout(attention)
        
        # 3. 用注意力权重加权V
        # x shape: [batch_size, nhead, query_len, head_dim]
        x = torch.matmul(attention, V)
        
        # 4. 拼接多头并进行最终线性变换
        # -> [batch_size, query_len, nhead, head_dim] -> [query_len, batch_size, d_model]
        x = x.permute(2, 0, 1, 3).contiguous()
        x = x.view(query.shape[0], batch_size, self.d_model)
        
        x = self.fc_o(x)
        return x

class PositionwiseFeedforward(nn.Module):
    """
    前馈神经网络。在每个注意力层之后，对每个位置的向量进行非线性变换。
    它由两个线性层和一个ReLU激活函数组成。
    """
    def __init__(self, d_model: int, dim_feedforward: int, dropout: float = 0.1):
        super().__init__()
        self.linear_1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(p=dropout)
        self.linear_2 = nn.Linear(dim_feedforward, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.linear_2(self.dropout(torch.relu(self.linear_1(x))))
        return x

class EncoderLayer(nn.Module):
    """
    单个编码器层。由一个多头自注意力模块和一个前馈网络组成。
    使用了残差连接和层归一化来稳定训练。
    """
    def __init__(self, d_model: int, nhead: int, dim_feedforward: int, dropout: float):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, nhead, dropout)
        self.feed_forward = PositionwiseFeedforward(d_model, dim_feedforward, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(p=dropout)
        
    def forward(self, src: torch.Tensor, src_key_padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # 自注意力部分
        # 我们的 MultiHeadAttention 只接受一个掩码，所以我们传入 padding_mask
        mask = None
        if src_key_padding_mask is not None:
             # 我们需要将 padding_mask [batch_size, src_len] 扩展为 [batch_size, 1, 1, src_len] 以便广播
            mask = src_key_padding_mask.unsqueeze(1).unsqueeze(2)

        attn_output = self.self_attn(src, src, src, mask=mask)
        # 残差连接与层归一化
        src = src + self.dropout(attn_output)
        src = self.norm1(src)
        
        # 前馈网络部分
        ff_output = self.feed_forward(src)
        # 残差连接与层归一化
        src = src + self.dropout(ff_output)
        src = self.norm2(src)
        return src

class DecoderLayer(nn.Module):
    """
    单个解码器层。比编码器层多了一个交叉注意力模块，用于关注编码器的输出。
    """
    def __init__(self, d_model: int, nhead: int, dim_feedforward: int, dropout: float):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, nhead, dropout)
        self.cross_attn = MultiHeadAttention(d_model, nhead, dropout)
        self.feed_forward = PositionwiseFeedforward(d_model, dim_feedforward, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(p=dropout)
        
    def forward(self, tgt: torch.Tensor, memory: torch.Tensor, tgt_mask: Optional[torch.Tensor], tgt_key_padding_mask: Optional[torch.Tensor], memory_key_padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # 1. 带掩码的自注意力
        # 合并掩码：causal mask 和 padding mask
        
        # 我们约定，模型外部传入的padding mask中，True代表需要mask的位置。
        # generate_square_subsequent_mask 返回的mask中，True代表需要屏蔽的位置。
        
        self_attn_mask = None
        if tgt_mask is not None:
            self_attn_mask = tgt_mask.unsqueeze(0) # [1, tgt_len, tgt_len] for broadcasting
        
        if tgt_key_padding_mask is not None:
            # [batch_size, 1, 1, tgt_len] for broadcasting
            padding_mask = tgt_key_padding_mask.unsqueeze(1).unsqueeze(2)
            if self_attn_mask is None:
                self_attn_mask = padding_mask
            else:
                self_attn_mask = self_attn_mask | padding_mask

        attn_output = self.self_attn(tgt, tgt, tgt, mask=self_attn_mask)
        tgt = tgt + self.dropout(attn_output)
        tgt = self.norm1(tgt)
        
        # 2. 交叉注意力
        cross_attn_mask = None
        if memory_key_padding_mask is not None:
            cross_attn_mask = memory_key_padding_mask.unsqueeze(1).unsqueeze(2)

        cross_attn_output = self.cross_attn(tgt, memory, memory, mask=cross_attn_mask)
        tgt = tgt + self.dropout(cross_attn_output)
        tgt = self.norm2(tgt)
        
        # 前馈网络
        ff_output = self.feed_forward(tgt)
        tgt = tgt + self.dropout(ff_output)
        tgt = self.norm3(tgt)
        return tgt

class Transformer(nn.Module):
    """
    完整的Transformer模型，将所有组件组装在一起。
    此版本使用torch.nn.TransformerEncoder和torch.nn.TransformerDecoder，
    这比手动实现更稳定、更推荐。
    """
    def __init__(self, src_vocab_size: int, tgt_vocab_size: int, d_model: int, nhead: int, num_encoder_layers: int, num_decoder_layers: int, dim_feedforward: int, dropout: float):
        super().__init__()
        
        # 词嵌入层
        self.src_embedding = nn.Embedding(src_vocab_size, d_model)
        self.tgt_embedding = nn.Embedding(tgt_vocab_size, d_model)
        
        # 位置编码
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        
        # 使用PyTorch内置的Encoder Layer
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, dropout=dropout, batch_first=False)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)
        
        # 使用PyTorch内置的Decoder Layer
        decoder_layer = nn.TransformerDecoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, dropout=dropout, batch_first=False)
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_decoder_layers)
        
        # 最终输出层
        self.fc_out = nn.Linear(d_model, tgt_vocab_size)
        
        self.d_model = d_model
        self.apply(self._init_weights)

    def _init_weights(self, module):
        # 初始化权重，这是一个很好的实践
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.Embedding):
            # 论文建议使用 Xavier 初始化
            nn.init.xavier_uniform_(module.weight)

    def forward(self, 
                src: torch.Tensor, 
                tgt: torch.Tensor, 
                tgt_mask: torch.Tensor, 
                src_key_padding_mask: torch.Tensor, 
                tgt_key_padding_mask: torch.Tensor,
                memory_key_padding_mask: torch.Tensor) -> torch.Tensor:
        # 1. 嵌入和位置编码
        # 论文中提到将嵌入权重乘以sqrt(d_model)
        src_emb = self.pos_encoder(self.src_embedding(src) * math.sqrt(self.d_model))
        tgt_emb = self.pos_encoder(self.tgt_embedding(tgt) * math.sqrt(self.d_model))
        
        # 2. 编码器处理
        # PyTorch的TransformerEncoder期望的输入形状是 [src_len, batch_size, d_model]
        memory = self.transformer_encoder(src_emb, src_key_padding_mask=src_key_padding_mask)
        
        # 3. 解码器处理
        # PyTorch的TransformerDecoder期望的输入形状是 [tgt_len, batch_size, d_model]
        output = self.transformer_decoder(tgt_emb, memory, 
                                          tgt_mask=tgt_mask, 
                                          tgt_key_padding_mask=tgt_key_padding_mask,
                                          memory_key_padding_mask=memory_key_padding_mask)
            
        # 4. 最终输出
        return self.fc_out(output)