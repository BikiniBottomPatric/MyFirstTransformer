#!/usr/bin/env python3
# model.py
# 项目宪法：Transformer模型 - 严格遵循"Attention is All You Need"论文
# 绝对忠于原文精神：100%手写核心Module + 论文标准架构

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
import logging

# 导入配置
import config

# 设置日志
logging.basicConfig(
    level=getattr(logging, config.LOG_LEVEL),
    format=config.LOG_FORMAT
)
logger = logging.getLogger(__name__)

class PositionalEncoding(nn.Module):
    """
    位置编码 - 严格遵循"Attention is All You Need"论文
    
    使用正弦和余弦函数生成位置编码，允许模型学习相对位置信息。
    PE(pos, 2i) = sin(pos / 10000^(2i/d_model))
    PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
    """
    
    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # 创建位置编码矩阵
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        
        # 计算除数项：10000^(2i/d_model)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           (-math.log(10000.0) / d_model))
        
        # 应用正弦和余弦函数
        pe[:, 0::2] = torch.sin(position * div_term)  # 偶数位置
        pe[:, 1::2] = torch.cos(position * div_term)  # 奇数位置
        
        # 添加批次维度并注册为buffer（不参与梯度更新）
        pe = pe.unsqueeze(0).transpose(0, 1)  # [max_len, 1, d_model]
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [seq_len, batch_size, d_model] 或 [batch_size, seq_len, d_model]
        Returns:
            添加位置编码后的张量
        """
        if x.dim() == 3 and x.size(0) != x.size(1):  # 判断是否为 [batch_size, seq_len, d_model]
            # 转换为 [seq_len, batch_size, d_model]
            x = x.transpose(0, 1)
            seq_len = x.size(0)
            # 确保位置编码长度足够
            if seq_len > self.pe.size(0):
                # 如果序列长度超过预设最大长度，只使用可用的位置编码
                x[:self.pe.size(0)] = x[:self.pe.size(0)] + self.pe
            else:
                x = x + self.pe[:seq_len, :]
            x = x.transpose(0, 1)  # 转换回 [batch_size, seq_len, d_model]
        else:
            # 假设输入为 [seq_len, batch_size, d_model]
            seq_len = x.size(0)
            if seq_len > self.pe.size(0):
                # 如果序列长度超过预设最大长度，只使用可用的位置编码
                x[:self.pe.size(0)] = x[:self.pe.size(0)] + self.pe
            else:
                x = x + self.pe[:seq_len, :]
        
        return self.dropout(x)

class MultiHeadAttention(nn.Module):
    """
    多头注意力机制 - 严格遵循"Attention is All You Need"论文
    
    Attention(Q,K,V) = softmax(QK^T / sqrt(d_k))V
    MultiHead(Q,K,V) = Concat(head_1, ..., head_h)W^O
    where head_i = Attention(QW_i^Q, KW_i^K, VW_i^V)
    """
    
    def __init__(self, d_model: int, nhead: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % nhead == 0, f"d_model ({d_model}) 必须能被 nhead ({nhead}) 整除"
        
        self.d_model = d_model
        self.nhead = nhead
        self.d_k = d_model // nhead  # 每个头的维度
        
        # 线性投影层
        self.w_q = nn.Linear(d_model, d_model, bias=False)
        self.w_k = nn.Linear(d_model, d_model, bias=False)
        self.w_v = nn.Linear(d_model, d_model, bias=False)
        self.w_o = nn.Linear(d_model, d_model, bias=False)
        
        self.dropout = nn.Dropout(dropout)
        
        # 缩放因子
        self.scale = math.sqrt(self.d_k)
    
    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            query: [batch_size, seq_len_q, d_model]
            key: [batch_size, seq_len_k, d_model]
            value: [batch_size, seq_len_v, d_model]
            mask: [batch_size, seq_len_q, seq_len_k] 或 [seq_len_q, seq_len_k] 或 [batch_size, seq_len]
        
        Returns:
            output: [batch_size, seq_len_q, d_model]
            attention_weights: [batch_size, nhead, seq_len_q, seq_len_k]
        """
        batch_size, seq_len_q, _ = query.size()
        seq_len_k = key.size(1)
        seq_len_v = value.size(1)
        
        # 线性投影并重塑为多头格式
        Q = self.w_q(query).view(batch_size, seq_len_q, self.nhead, self.d_k).transpose(1, 2)
        K = self.w_k(key).view(batch_size, seq_len_k, self.nhead, self.d_k).transpose(1, 2)
        V = self.w_v(value).view(batch_size, seq_len_v, self.nhead, self.d_k).transpose(1, 2)
        # 形状: [batch_size, nhead, seq_len, d_k]
        
        # 计算注意力分数
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        # 形状: [batch_size, nhead, seq_len_q, seq_len_k]
        
        # 应用掩码
        if mask is not None:
            # 处理不同维度的掩码
            if mask.dim() == 2:  # [seq_len_q, seq_len_k] 或 [batch_size, seq_len]
                if mask.size(0) == seq_len_q and mask.size(1) == seq_len_k:
                    # 因果掩码: [seq_len_q, seq_len_k] -> [1, 1, seq_len_q, seq_len_k]
                    mask = mask.unsqueeze(0).unsqueeze(0)
                elif mask.size(0) == batch_size:
                    # 填充掩码: [batch_size, seq_len] -> [batch_size, 1, 1, seq_len]
                    mask = mask.unsqueeze(1).unsqueeze(2)
                    # 扩展到所有查询位置: [batch_size, 1, seq_len_q, seq_len]
                    mask = mask.expand(batch_size, 1, seq_len_q, seq_len_k)
            elif mask.dim() == 3:  # [batch_size, seq_len_q, seq_len_k]
                mask = mask.unsqueeze(1)  # [batch_size, 1, seq_len_q, seq_len_k]
            
            # 将掩码位置设为负无穷
            scores = scores.masked_fill(mask == 0, float('-inf'))
        
        # 计算注意力权重
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        # 应用注意力权重到值
        context = torch.matmul(attention_weights, V)
        # 形状: [batch_size, nhead, seq_len_q, d_k]
        
        # 重塑并连接多头
        context = context.transpose(1, 2).contiguous().view(
            batch_size, seq_len_q, self.d_model
        )
        
        # 最终线性投影
        output = self.w_o(context)
        
        return output, attention_weights

class PositionwiseFeedforward(nn.Module):
    """
    位置前馈网络 - 严格遵循"Attention is All You Need"论文
    
    FFN(x) = max(0, xW_1 + b_1)W_2 + b_2
    
    两层线性变换，中间使用ReLU激活函数。
    内层维度通常是模型维度的4倍。
    """
    
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch_size, seq_len, d_model]
        Returns:
            [batch_size, seq_len, d_model]
        """
        return self.linear2(self.dropout(F.relu(self.linear1(x))))

class EncoderLayer(nn.Module):
    """
    Transformer编码器层 - 严格遵循"Attention is All You Need"论文
    
    每层包含：
    1. 多头自注意力机制
    2. 残差连接和层归一化
    3. 位置前馈网络
    4. 残差连接和层归一化
    """
    
    def __init__(self, d_model: int, nhead: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, nhead, dropout)
        self.feed_forward = PositionwiseFeedforward(d_model, d_ff, dropout)
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, src: torch.Tensor, src_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            src: [batch_size, seq_len, d_model]
            src_mask: [batch_size, seq_len, seq_len] 或 [seq_len, seq_len]
        Returns:
            [batch_size, seq_len, d_model]
        """
        # 多头自注意力 + 残差连接 + 层归一化
        attn_output, _ = self.self_attn(src, src, src, src_mask)
        src = self.norm1(src + self.dropout(attn_output))
        
        # 前馈网络 + 残差连接 + 层归一化
        ff_output = self.feed_forward(src)
        src = self.norm2(src + self.dropout(ff_output))
        
        return src

class DecoderLayer(nn.Module):
    """
    Transformer解码器层 - 严格遵循"Attention is All You Need"论文
    
    每层包含：
    1. 掩码多头自注意力机制
    2. 残差连接和层归一化
    3. 编码器-解码器多头注意力机制
    4. 残差连接和层归一化
    5. 位置前馈网络
    6. 残差连接和层归一化
    """
    
    def __init__(self, d_model: int, nhead: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, nhead, dropout)
        self.cross_attn = MultiHeadAttention(d_model, nhead, dropout)
        self.feed_forward = PositionwiseFeedforward(d_model, d_ff, dropout)
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, tgt: torch.Tensor, memory: torch.Tensor,
                tgt_mask: Optional[torch.Tensor] = None,
                memory_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            tgt: [batch_size, tgt_seq_len, d_model]
            memory: [batch_size, src_seq_len, d_model] (编码器输出)
            tgt_mask: [tgt_seq_len, tgt_seq_len] (因果掩码)
            memory_mask: [batch_size, tgt_seq_len, src_seq_len] (源序列掩码)
        Returns:
            [batch_size, tgt_seq_len, d_model]
        """
        # 掩码多头自注意力 + 残差连接 + 层归一化
        self_attn_output, _ = self.self_attn(tgt, tgt, tgt, tgt_mask)
        tgt = self.norm1(tgt + self.dropout(self_attn_output))
        
        # 编码器-解码器注意力 + 残差连接 + 层归一化
        cross_attn_output, _ = self.cross_attn(tgt, memory, memory, memory_mask)
        tgt = self.norm2(tgt + self.dropout(cross_attn_output))
        
        # 前馈网络 + 残差连接 + 层归一化
        ff_output = self.feed_forward(tgt)
        tgt = self.norm3(tgt + self.dropout(ff_output))
        
        return tgt

class TransformerEncoder(nn.Module):
    """
    Transformer编码器 - 多层编码器层的堆叠
    """
    
    def __init__(self, encoder_layer: EncoderLayer, num_layers: int):
        super().__init__()
        self.layers = nn.ModuleList([encoder_layer for _ in range(num_layers)])
        self.num_layers = num_layers
    
    def forward(self, src: torch.Tensor, src_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            src: [batch_size, seq_len, d_model]
            src_mask: [batch_size, seq_len, seq_len] 或 [seq_len, seq_len]
        Returns:
            [batch_size, seq_len, d_model]
        """
        output = src
        for layer in self.layers:
            output = layer(output, src_mask)
        return output

class TransformerDecoder(nn.Module):
    """
    Transformer解码器 - 多层解码器层的堆叠
    """
    
    def __init__(self, decoder_layer: DecoderLayer, num_layers: int):
        super().__init__()
        self.layers = nn.ModuleList([decoder_layer for _ in range(num_layers)])
        self.num_layers = num_layers
    
    def forward(self, tgt: torch.Tensor, memory: torch.Tensor,
                tgt_mask: Optional[torch.Tensor] = None,
                memory_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            tgt: [batch_size, tgt_seq_len, d_model]
            memory: [batch_size, src_seq_len, d_model]
            tgt_mask: [tgt_seq_len, tgt_seq_len]
            memory_mask: [batch_size, tgt_seq_len, src_seq_len]
        Returns:
            [batch_size, tgt_seq_len, d_model]
        """
        output = tgt
        for layer in self.layers:
            output = layer(output, memory, tgt_mask, memory_mask)
        return output

class Transformer(nn.Module):
    """
    完整的Transformer模型 - 严格遵循"Attention is All You Need"论文
    
    包含：
    1. 词嵌入层（源语言和目标语言共享或分离）
    2. 位置编码
    3. Transformer编码器
    4. Transformer解码器
    5. 输出投影层
    """
    
    def __init__(self, vocab_size: int, d_model: int = 512, nhead: int = 8,
                 num_encoder_layers: int = 6, num_decoder_layers: int = 6,
                 d_ff: int = 2048, max_seq_len: int = 5000, dropout: float = 0.1,
                 share_embeddings: bool = True):
        super().__init__()
        
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.share_embeddings = share_embeddings
        
        # 词嵌入层
        if share_embeddings:
            # 源语言和目标语言共享词嵌入（论文推荐）
            self.embedding = nn.Embedding(vocab_size, d_model)
            self.src_embedding = self.embedding
            self.tgt_embedding = self.embedding
        else:
            # 分离的词嵌入
            self.src_embedding = nn.Embedding(vocab_size, d_model)
            self.tgt_embedding = nn.Embedding(vocab_size, d_model)
        
        # 位置编码
        self.pos_encoding = PositionalEncoding(d_model, max_seq_len, dropout)
        
        # 编码器和解码器
        encoder_layer = EncoderLayer(d_model, nhead, d_ff, dropout)
        self.encoder = TransformerEncoder(encoder_layer, num_encoder_layers)
        
        decoder_layer = DecoderLayer(d_model, nhead, d_ff, dropout)
        self.decoder = TransformerDecoder(decoder_layer, num_decoder_layers)
        
        # 输出投影层
        self.output_projection = nn.Linear(d_model, vocab_size)
        
        # 先初始化权重
        self._init_weights()
        
        # 然后进行权重绑定（必须在初始化之后）
        if share_embeddings:
            self.output_projection.weight = self.embedding.weight
        
        logger.info(f"🚀 Transformer模型初始化完成")
        logger.info(f"📊 模型参数: vocab_size={vocab_size}, d_model={d_model}, nhead={nhead}")
        logger.info(f"📊 层数: encoder={num_encoder_layers}, decoder={num_decoder_layers}")
        logger.info(f"📊 参数总数: {self.count_parameters():,}")
    
    def _init_weights(self):
        """权重初始化 - 遵循论文建议"""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def count_parameters(self) -> int:
        """计算模型参数总数"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def create_padding_mask(self, seq: torch.Tensor, pad_idx: int) -> torch.Tensor:
        """创建填充掩码"""
        return (seq != pad_idx).unsqueeze(1).unsqueeze(2)  # [batch_size, 1, 1, seq_len]
    
    def create_causal_mask(self, size: int) -> torch.Tensor:
        """创建因果掩码（下三角矩阵）"""
        mask = torch.tril(torch.ones(size, size, dtype=torch.bool))
        return mask  # [size, size]
    
    def forward(self, src: torch.Tensor, tgt: torch.Tensor,
                src_mask: Optional[torch.Tensor] = None,
                tgt_mask: Optional[torch.Tensor] = None,
                memory_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        前向传播
        
        Args:
            src: [batch_size, src_seq_len] 源序列token IDs
            tgt: [batch_size, tgt_seq_len] 目标序列token IDs
            src_mask: [batch_size, src_seq_len] 或 [tgt_seq_len, tgt_seq_len] 源序列填充掩码或因果掩码
            tgt_mask: [tgt_seq_len, tgt_seq_len] 目标序列因果掩码
            memory_mask: [batch_size, tgt_seq_len, src_seq_len] 交叉注意力掩码
        
        Returns:
            [batch_size, tgt_seq_len, vocab_size] 输出logits
        """
        # 词嵌入 + 位置编码
        src_emb = self.src_embedding(src) * math.sqrt(self.d_model)
        tgt_emb = self.tgt_embedding(tgt) * math.sqrt(self.d_model)
        
        src_emb = self.pos_encoding(src_emb)
        tgt_emb = self.pos_encoding(tgt_emb)
        
        # 编码器
        memory = self.encoder(src_emb, src_mask)
        
        # 解码器
        output = self.decoder(tgt_emb, memory, tgt_mask, memory_mask)
        
        # 输出投影
        logits = self.output_projection(output)
        
        return logits
    
    def encode(self, src: torch.Tensor, src_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """仅编码器前向传播（用于推理）"""
        src_emb = self.src_embedding(src) * math.sqrt(self.d_model)
        src_emb = self.pos_encoding(src_emb)
        return self.encoder(src_emb, src_mask)
    
    def decode(self, tgt: torch.Tensor, memory: torch.Tensor,
               tgt_mask: Optional[torch.Tensor] = None,
               memory_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """仅解码器前向传播（用于推理）"""
        tgt_emb = self.tgt_embedding(tgt) * math.sqrt(self.d_model)
        tgt_emb = self.pos_encoding(tgt_emb)
        output = self.decoder(tgt_emb, memory, tgt_mask, memory_mask)
        return self.output_projection(output)

def create_transformer_model(vocab_size: int) -> Transformer:
    """
    创建Transformer模型 - 使用配置文件参数
    
    Args:
        vocab_size: 词汇表大小
    
    Returns:
        Transformer模型实例
    """
    # 确保位置编码长度足够支持beam search的最大解码长度
    max_pos_len = max(config.MAX_SEQ_LEN, config.MAX_DECODE_LENGTH + 50)  # 额外50个位置作为缓冲
    
    model = Transformer(
        vocab_size=vocab_size,
        d_model=config.D_MODEL,
        nhead=config.NHEAD,
        num_encoder_layers=config.NUM_ENCODER_LAYERS,
        num_decoder_layers=config.NUM_DECODER_LAYERS,
        d_ff=config.D_FF,
        max_seq_len=max_pos_len,
        dropout=config.DROPOUT,
        share_embeddings=config.SHARE_EMBEDDINGS
    )
    
    logger.info(f"✅ Transformer模型创建成功")
    logger.info(f"📊 模型配置: Base Model (d_model={config.D_MODEL}, nhead={config.NHEAD})")
    logger.info(f"📊 参数估计: {model.count_parameters() / 1e6:.1f}M")
    
    return model

def main():
    """主函数 - 用于测试模型"""
    print("🚀 Transformer模型测试 - 项目宪法实现")
    print("="*80)
    print("📋 核心原则:")
    print("  ✅ 绝对忠于原文精神: 100%手写核心Module")
    print("  ✅ 严格论文架构: MultiHeadAttention + EncoderLayer + DecoderLayer")
    print("  ✅ 权重初始化: Xavier Uniform")
    print("  ✅ 位置编码: 正弦余弦函数")
    print("="*80)
    
    try:
        # 创建测试模型
        vocab_size = 37000  # BPE词汇表大小
        model = create_transformer_model(vocab_size)
        
        # 测试前向传播
        batch_size = 2
        src_seq_len = 10
        tgt_seq_len = 8
        
        # 创建测试数据
        src = torch.randint(0, vocab_size, (batch_size, src_seq_len))
        tgt = torch.randint(0, vocab_size, (batch_size, tgt_seq_len))
        
        # 创建掩码
        src_mask = model.create_padding_mask(src, config.PAD_IDX)
        tgt_mask = model.create_causal_mask(tgt_seq_len)
        
        print(f"\n🧪 测试前向传播...")
        print(f"📊 输入形状:")
        print(f"  src: {src.shape}")
        print(f"  tgt: {tgt.shape}")
        print(f"  src_mask: {src_mask.shape}")
        print(f"  tgt_mask: {tgt_mask.shape}")
        
        # 前向传播
        with torch.no_grad():
            output = model(src, tgt, src_mask, tgt_mask)
        
        print(f"📊 输出形状: {output.shape}")
        print(f"📊 输出范围: [{output.min().item():.3f}, {output.max().item():.3f}]")
        
        # 测试编码器和解码器分离
        print(f"\n🧪 测试编码器/解码器分离...")
        with torch.no_grad():
            memory = model.encode(src, src_mask)
            decoder_output = model.decode(tgt, memory, tgt_mask)
        
        print(f"📊 编码器输出: {memory.shape}")
        print(f"📊 解码器输出: {decoder_output.shape}")
        
        # 验证输出一致性
        diff = torch.abs(output - decoder_output).max().item()
        print(f"📊 输出一致性检查: 最大差异 = {diff:.6f}")
        
        if diff < 1e-5:
            print("✅ 模型测试通过!")
        else:
            print("❌ 模型测试失败: 输出不一致")
        
        print(f"\n🎯 模型已准备就绪，可用于训练")
        
    except Exception as e:
        print(f"❌ 模型测试失败: {str(e)}")
        raise

if __name__ == "__main__":
    main()