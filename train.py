#!/usr/bin/env python3
"""
Transformer训练脚本 - BLEU 25目标优化版本
项目宪法：严格遵循"Attention is All You Need"论文 + 所有BLEU提升技术

核心优化技术：
1. 增强学习率调度（Warmup + 缩放）
2. 标签平滑（Label Smoothing）
3. Beam Search解码
4. 梯度累积
5. 早停策略
6. 长度惩罚
7. 正则化（Dropout）
8. 检查点管理
9. TensorBoard监控
10. BLEU评估
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F

import os
import time
import math
import itertools
import json
import argparse
from typing import Dict, List, Tuple, Optional, Any
from tqdm import tqdm
import logging

# 项目模块
import config
from model import create_transformer_model
from data_utils import create_data_loaders
from beam_search import create_beam_search_decoder

# 设置日志
logging.basicConfig(
    level=getattr(logging, config.LOG_LEVEL),
    format=config.LOG_FORMAT,
    handlers=[
        logging.FileHandler('logs/train.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class EnhancedTrainer:
    """
    增强版Transformer训练器 - 集成所有BLEU提升技术
    
    核心特性：
    - 增强学习率调度
    - 标签平滑损失
    - Beam Search解码
    - 梯度累积
    - 早停策略
    - 检查点管理
    - TensorBoard监控
    - BLEU评估
    """
    
    def __init__(self, resume_from: Optional[str] = None):
        """初始化训练器"""
        logger.info("🚀 初始化增强版Transformer训练器")
        
        # 创建必要目录
        os.makedirs(config.CHECKPOINTS_DIR, exist_ok=True)
        os.makedirs('logs', exist_ok=True)
        
        # 数据加载器
        logger.info("📚 加载数据...")
        self.train_loader, self.valid_loader, self.test_loader = create_data_loaders()
        
        # 获取词汇表信息
        from data_utils import get_vocab_info
        self.vocab_info = get_vocab_info()
        self.vocab_size = self.vocab_info['vocab_size']
        
        # 加载SentencePiece模型用于BLEU计算
        import sentencepiece as spm
        bpe_model_path = self.vocab_info['bpe_model_path']
        self.sp_model = spm.SentencePieceProcessor(model_file=bpe_model_path)
        
        logger.info(f"📊 词汇表大小: {self.vocab_size}")
        logger.info(f"🔤 SentencePiece模型已加载: {bpe_model_path}")
        
        # 模型
        logger.info("🏗️ 创建模型...")
        from model import create_transformer_model
        self.model = create_transformer_model(self.vocab_size).to(config.DEVICE)
        
        total_params = sum(p.numel() for p in self.model.parameters())
        logger.info(f"📈 模型参数: {total_params:,} ({total_params/1e6:.1f}M)")
        
        # 损失函数（带标签平滑）
        self.criterion = nn.CrossEntropyLoss(
            label_smoothing=config.LABEL_SMOOTHING_EPS,
            ignore_index=config.PAD_IDX
        )
        
        # 优化器
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=1.0,  # 会被学习率调度器覆盖
            betas=(0.9, 0.98),
            eps=1e-9
        )
        
        # Beam Search解码器
        self.beam_decoder = create_beam_search_decoder(self.model)
        
        # TensorBoard - 使用config.py中定义的日志目录
        self.writer = SummaryWriter(log_dir=config.TENSORBOARD_LOG_DIR, comment="_enhanced_transformer")
        
        # 训练状态
        self.global_step = 0
        self.best_bleu = 0.0
        self.no_improvement_steps = 0
        self.start_time = time.time()
        
        # ==== 添加恢复逻辑 ====
        if resume_from:
            if os.path.exists(resume_from):
                logger.info(f"🔄 从检查点恢复训练: {resume_from}")
                checkpoint = torch.load(resume_from, map_location=config.DEVICE)
                self.model.load_state_dict(checkpoint['model_state_dict'])
                self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                self.global_step = checkpoint['global_step']
                self.best_bleu = checkpoint.get('best_bleu', 0.0)  # 使用 .get 保证向后兼容
                logger.info(f"✅ 恢复成功! 从物理步 {self.global_step} 继续")
            else:
                logger.error(f"❌ 指定的恢复检查点不存在: {resume_from}")
                # 可以选择退出或从头开始
                raise FileNotFoundError(f"Checkpoint not found: {resume_from}")
        
        logger.info("✅ 训练器初始化完成")
    
    def enhanced_lr_schedule(self, step: int) -> float:
        """
        增强学习率调度 - 基于"Attention is All You Need"论文
        
        LR = d_model^(-0.5) * min(step^(-0.5), step * warmup_steps^(-1.5)) * scale
        
        Args:
            step: 当前训练步数
        
        Returns:
            学习率
        """
        d_model = config.D_MODEL
        warmup_steps = config.WARMUP_STEPS
        scale = config.LEARNING_RATE_SCALE
        
        step = max(1, step)  # 避免除零
        
        lr = (d_model ** -0.5) * min(
            step ** -0.5,
            step * (warmup_steps ** -1.5)
        ) * scale
        
        return lr
    
    def compute_loss(self, src: torch.Tensor, tgt: torch.Tensor) -> torch.Tensor:
        """
        计算损失
        
        Args:
            src: [seq_len, batch_size] 源序列
            tgt: [seq_len, batch_size] 目标序列
        
        Returns:
            损失值
        """
        # 准备输入和输出
        tgt_input = tgt[:-1, :]  # 移除最后一个token作为输入
        tgt_output = tgt[1:, :]  # 移除第一个token作为输出
        
        # 创建因果掩码
        tgt_mask = self.model.create_causal_mask(tgt_input.size(0)).to(config.DEVICE)
        
        # 前向传播
        logits = self.model(src=src, tgt=tgt_input, tgt_mask=tgt_mask)
        
        # 计算损失
        loss = self.criterion(
            logits.reshape(-1, logits.shape[-1]),
            tgt_output.reshape(-1)
        )
        
        return loss
    
    def train_step_amp(self, batch_data: Dict[str, torch.Tensor], scaler: torch.cuda.amp.GradScaler) -> float:
        """
        执行单个训练步骤 - AMP混合精度版本
        
        Args:
            batch_data: 包含src, tgt_input, tgt_output, src_mask, tgt_mask, causal_mask的字典
            scaler: AMP梯度缩放器
        
        Returns:
            损失值
        """
        # 从批次数据中提取张量
        src = batch_data['src'].to(config.DEVICE)
        tgt_input = batch_data['tgt_input'].to(config.DEVICE)
        tgt_output = batch_data['tgt_output'].to(config.DEVICE)
        src_mask = batch_data.get('src_mask', None)
        tgt_mask = batch_data.get('causal_mask', None)
        
        if src_mask is not None:
            src_mask = src_mask.to(config.DEVICE)
        if tgt_mask is not None:
            tgt_mask = tgt_mask.to(config.DEVICE)
        
        # 序列长度限制
        if src.size(1) > config.MAX_SEQ_LEN:
            src = src[:, :config.MAX_SEQ_LEN]
            if src_mask is not None:
                src_mask = src_mask[:, :config.MAX_SEQ_LEN]
        
        if tgt_input.size(1) > config.MAX_SEQ_LEN:
            tgt_input = tgt_input[:, :config.MAX_SEQ_LEN]
            tgt_output = tgt_output[:, :config.MAX_SEQ_LEN]
            if tgt_mask is not None:
                tgt_mask = tgt_mask[:config.MAX_SEQ_LEN, :config.MAX_SEQ_LEN]
        
        # 学习率调度
        lr = self.enhanced_lr_schedule(self.global_step)
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
        
        # 梯度累积开始
        if self.global_step % config.GRADIENT_ACCUMULATION_STEPS == 1:
            self.optimizer.zero_grad()
        
        # 2. 使用 autocast 上下文管理器 (使用新API)
        # 在这个代码块内，所有符合条件的CUDA操作都会自动使用float16
        with torch.amp.autocast('cuda'):
            logits = self.model(src, tgt_input, src_mask=src_mask, tgt_mask=tgt_mask)
            loss = self.criterion(
                logits.reshape(-1, logits.shape[-1]),  # [batch_size * seq_len, vocab_size]
                tgt_output.reshape(-1)  # [batch_size * seq_len]
            )
            scaled_loss = loss / config.GRADIENT_ACCUMULATION_STEPS
        
        # 使用 scaler.scale() 和 scaler.step()
        # scaler.scale() 将损失乘以缩放因子，然后进行反向传播
        scaler.scale(scaled_loss).backward()
        
        # 梯度更新
        if self.global_step % config.GRADIENT_ACCUMULATION_STEPS == 0:
            # 在更新前，反缩放梯度以进行梯度裁剪
            scaler.unscale_(self.optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            # scaler.step() 会检查梯度是否溢出
            # 如果没有溢出，它会调用 optimizer.step() 更新权重
            # 如果溢出了，它会跳过此次更新
            scaler.step(self.optimizer)
            
            # scaler.update() 更新缩放因子，为下一次迭代做准备
            scaler.update()
        
        # 返回损失和梯度范数（如果有梯度更新）
        if self.global_step % config.GRADIENT_ACCUMULATION_STEPS == 0:
            return loss.item(), grad_norm.item()
        else:
            return loss.item(), None
    
    def train_step(self, batch_data: Dict[str, torch.Tensor]) -> float:
        """
        执行单个训练步骤 - 保留原版本作为备用
        
        Args:
            batch_data: 包含src, tgt_input, tgt_output, src_mask, tgt_mask, causal_mask的字典
        
        Returns:
            损失值
        """
        # 从批次数据中提取张量
        src = batch_data['src'].to(config.DEVICE)
        tgt_input = batch_data['tgt_input'].to(config.DEVICE)
        tgt_output = batch_data['tgt_output'].to(config.DEVICE)
        src_mask = batch_data.get('src_mask', None)
        tgt_mask = batch_data.get('causal_mask', None)
        
        if src_mask is not None:
            src_mask = src_mask.to(config.DEVICE)
        if tgt_mask is not None:
            tgt_mask = tgt_mask.to(config.DEVICE)
        
        # 序列长度限制
        if src.size(1) > config.MAX_SEQ_LEN:
            src = src[:, :config.MAX_SEQ_LEN]
            if src_mask is not None:
                src_mask = src_mask[:, :config.MAX_SEQ_LEN]
        
        if tgt_input.size(1) > config.MAX_SEQ_LEN:
            tgt_input = tgt_input[:, :config.MAX_SEQ_LEN]
            tgt_output = tgt_output[:, :config.MAX_SEQ_LEN]
            if tgt_mask is not None:
                tgt_mask = tgt_mask[:config.MAX_SEQ_LEN, :config.MAX_SEQ_LEN]
        
        # 学习率调度
        lr = self.enhanced_lr_schedule(self.global_step)
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
        
        # 梯度累积开始
        if self.global_step % config.GRADIENT_ACCUMULATION_STEPS == 1:
            self.optimizer.zero_grad()
        
        # 前向传播
        logits = self.model(src, tgt_input, src_mask=src_mask, tgt_mask=tgt_mask)
        
        # 计算损失 - 正确reshape logits和target
        loss = self.criterion(
            logits.reshape(-1, logits.shape[-1]),  # [batch_size * seq_len, vocab_size]
            tgt_output.reshape(-1)  # [batch_size * seq_len]
        )
        
        # 梯度累积缩放
        scaled_loss = loss / config.GRADIENT_ACCUMULATION_STEPS
        scaled_loss.backward()
        
        # 梯度更新
        if self.global_step % config.GRADIENT_ACCUMULATION_STEPS == 0:
            grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
        
        # 返回损失和梯度范数（如果有梯度更新）
        if self.global_step % config.GRADIENT_ACCUMULATION_STEPS == 0:
            return loss.item(), grad_norm.item()
        else:
            return loss.item(), None
    
    def evaluate_bleu(self, dataloader: DataLoader, max_samples: int = 100) -> float:
        """
        BLEU评估 - 使用Beam Search（符合论文标准）
        
        Args:
            dataloader: 数据加载器
            max_samples: 最大评估样本数
        
        Returns:
            BLEU分数
        """
        try:
            from sacrebleu import corpus_bleu
            from beam_search import BeamSearchDecoder
        except ImportError:
            logger.warning("sacrebleu或beam_search未安装，使用简化BLEU评估")
            return self.simple_bleu_evaluation(dataloader, max_samples)
        
        self.model.eval()
        references = []
        hypotheses = []
        
        # 创建Beam Search解码器（论文标准：beam_size=4）
        beam_decoder = BeamSearchDecoder(
            model=self.model,
            beam_size=4,  # 论文标准
            max_length=config.MAX_DECODE_LENGTH,
            length_penalty=0.6,  # 论文标准
            early_stopping=True
        )
        
        count = 0
        sample_count = 0
        with torch.no_grad():
            # 遍历批次
            for batch_data in tqdm(dataloader, desc="📊 BLEU评估 (Beam Search)", leave=False):
                if count >= max_samples:
                    break
                
                # 从批次中提取张量
                src_batch = batch_data['src'].to(config.DEVICE)
                tgt_batch = batch_data['tgt_output'].to(config.DEVICE)
                src_mask_batch = batch_data.get('src_mask', None)
                if src_mask_batch is not None:
                    src_mask_batch = src_mask_batch.to(config.DEVICE)

                # 优化：只处理批次中的第一个样本，避免重复生成
                batch_size = min(src_batch.size(0), max_samples - count)
                for i in range(batch_size):
                    if count >= max_samples:
                        break

                    # 获取单个样本，并保持批次维度 [1, seq_len]
                    src_sentence = src_batch[i:i+1]
                    src_mask = src_mask_batch[i:i+1] if src_mask_batch is not None else None
                    
                    # 使用Beam Search解码（论文标准）
                    try:
                        beam_result = beam_decoder.search(src_sentence, src_mask, verbose=False)
                        pred_tokens = beam_result['sequences'][0] if beam_result['sequences'] else []
                    except Exception as e:
                        logger.warning(f"Beam Search解码失败: {e}，回退到贪心解码")
                        pred_tokens = self.greedy_decode(src_sentence, src_mask)
                    
                    # 转换为文本
                    src_tokens = src_batch[i].cpu().numpy()
                    tgt_tokens = tgt_batch[i].cpu().numpy()
                    
                    src_text = self._tokens_to_text(src_tokens, self.vocab_size)
                    ref_text = self._tokens_to_text(tgt_tokens, self.vocab_size)
                    hyp_text = self._tokens_to_text(pred_tokens, self.vocab_size)
                    
                    # 打印前两个样本用于调试
                    if sample_count < 2:
                        logger.info(f"\n📝 样本 {sample_count + 1} (Beam Search):")
                        logger.info(f"   源文本: {src_text}")
                        logger.info(f"   参考译文: {ref_text}")
                        logger.info(f"   模型译文: {hyp_text}")
                        sample_count += 1
                    
                    if ref_text.strip() and hyp_text.strip():  # 确保两者都不为空
                        references.append(ref_text)
                        hypotheses.append(hyp_text)
                    
                    count += 1
        
        if not references or not hypotheses or len(hypotheses) != len(references):
            logger.warning("❌ 没有有效的翻译对或数量不匹配，返回0 BLEU")
            return 0.0
        
        logger.info(f"📊 收集了 {len(references)} 个有效翻译对 (Beam Search)")
        
        # 计算BLEU
        try:
            # sacrebleu 期望的格式是：hypotheses是一个列表，references是一个列表的列表
            bleu = corpus_bleu(hypotheses, [references])
            logger.info(f"📊 BLEU详情: {bleu}")
            return bleu.score
        except Exception as e:
            logger.warning(f"❌ BLEU计算失败: {e}")
            return 0.0
    
    def simple_bleu_evaluation(self, dataloader: DataLoader, max_samples: int = 50) -> float:
        """
        简化BLEU评估（当sacrebleu不可用时）
        """
        self.model.eval()
        total_score = 0.0
        count = 0
        
        with torch.no_grad():
            for batch_data in dataloader:
                if count >= max_samples:
                    break
                
                src = batch_data['src'].to(config.DEVICE)
                tgt_output = batch_data['tgt_output'].to(config.DEVICE)
                
                # 编码
                src_mask = batch_data.get('src_mask', None)
                if src_mask is not None:
                    src_mask = src_mask.to(config.DEVICE)
                
                # 贪心解码
                pred_tokens = self.greedy_decode(src[:1], src_mask[:1] if src_mask is not None else None)
                tgt_tokens = tgt_output[0].cpu().numpy()  # 获取第一个样本
                
                # 简单的token匹配分数
                matches = sum(1 for p, t in zip(pred_tokens, tgt_tokens) if p == t)
                total_score += matches / max(len(pred_tokens), len(tgt_tokens), 1)
                count += 1
        
        return (total_score / max(count, 1)) * 100  # 转换为百分比
    
    def greedy_decode(self, src: torch.Tensor, src_mask: Optional[torch.Tensor] = None, max_length: int = 100) -> List[int]:
        """
        贪心解码（备用方案）
        
        Args:
            src: 源序列 [1, src_seq_len]
            src_mask: 源序列掩码
            max_length: 最大解码长度
        
        Returns:
            解码的token序列
        """
        self.model.eval()
        
        # 编码
        memory = self.model.encode(src, src_mask)
        
        # 解码
        ys = torch.ones(1, 1).fill_(config.BOS_IDX).type(torch.long).to(config.DEVICE)
        
        for _ in range(max_length):
            tgt_mask = self.model.create_causal_mask(ys.size(1)).to(config.DEVICE)
            out = self.model.decode(ys, memory, tgt_mask)
            prob = F.softmax(out[:, -1], dim=-1)
            next_word = torch.argmax(prob, dim=-1).item()
            
            if next_word == config.EOS_IDX:
                break
            
            ys = torch.cat([ys, torch.ones(1, 1).type(torch.long).fill_(next_word).to(config.DEVICE)], dim=1)
        
        return ys[0, 1:].cpu().tolist()  # 移除BOS
    
    def _tokens_to_text(self, tokens: List[int], vocab_size: int) -> str:
        """
        将token转换为文本用于BLEU计算
        
        Args:
            tokens: token序列
            vocab_size: 词汇表大小
            
        Returns:
            str: 解码后的文本
        """
        try:
            # 过滤特殊token
            filtered_tokens = []
            for token in tokens:
                if isinstance(token, torch.Tensor):
                    token = token.item()
                token_int = int(token)
                if token_int not in [config.PAD_IDX, config.UNK_IDX, config.BOS_IDX, config.EOS_IDX]:
                    filtered_tokens.append(token_int)
            
            if not filtered_tokens:
                return ""
            
            # 使用SentencePiece解码
            if hasattr(self, 'sp_model'):
                text = self.sp_model.decode_ids(filtered_tokens)
                # 替换▁为空格，这是SentencePiece的标准做法
                text = text.replace('▁', ' ').strip()
                return text
            else:
                logger.warning("SentencePiece模型未加载")
                return ""
            
        except Exception as e:
            logger.warning(f"文本解码失败: {str(e)}")
            return ""
    
    def validate(self, skip_bleu: bool = False) -> Tuple[float, float]:
        """
        验证模型
        
        Args:
            skip_bleu: 是否跳过BLEU评估
        
        Returns:
            (验证损失, BLEU分数)
        """
        logger.info(f"🔍 验证模型 (Step {self.global_step})")
        
        self.model.eval()
        total_loss = 0.0
        count = 0
        
        with torch.no_grad():
            for batch_data in tqdm(self.valid_loader, desc="🔍 验证中", leave=False):
                if count >= 100:  # 限制验证时间
                    break
                
                src = batch_data['src'].to(config.DEVICE)
                tgt_input = batch_data['tgt_input'].to(config.DEVICE)
                tgt_output = batch_data['tgt_output'].to(config.DEVICE)
                src_mask = batch_data.get('src_mask', None)
                tgt_mask = batch_data.get('causal_mask', None)
                
                if src_mask is not None:
                    src_mask = src_mask.to(config.DEVICE)
                if tgt_mask is not None:
                    tgt_mask = tgt_mask.to(config.DEVICE)
                
                # 前向传播
                logits = self.model(src, tgt_input, src_mask=src_mask, tgt_mask=tgt_mask)
                
                # 计算损失 - 正确reshape logits和target
                loss = self.criterion(
                    logits.reshape(-1, logits.shape[-1]),  # [batch_size * seq_len, vocab_size]
                    tgt_output.reshape(-1)  # [batch_size * seq_len]
                )
                total_loss += loss.item()
                count += 1
        
        avg_loss = total_loss / max(count, 1)
        
        # BLEU评估（可选）
        if skip_bleu:
            bleu_score = 0.0
            logger.info(f"📉 验证损失: {avg_loss:.4f}")
            logger.info(f"⏭️ 跳过BLEU评估")
        else:
            bleu_score = self.evaluate_bleu(self.valid_loader, max_samples=50)
            logger.info(f"📉 验证损失: {avg_loss:.4f}")
            logger.info(f"📊 BLEU分数: {bleu_score:.2f}")
        
        # TensorBoard记录（使用逻辑步）
        logical_step = self.global_step // config.GRADIENT_ACCUMULATION_STEPS
        self.writer.add_scalar('Validation/Loss', avg_loss, logical_step)
        if not skip_bleu:
            self.writer.add_scalar('Validation/BLEU', bleu_score, logical_step)
        
        return avg_loss, bleu_score
    
    def save_checkpoint(self, bleu_score: float, val_loss: float, is_best: bool = False):
        """
        保存检查点
        """
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'global_step': self.global_step,
            'best_bleu': self.best_bleu,
            'bleu_score': bleu_score,
            'val_loss': val_loss,
            'config': {
                'src_vocab_size': self.vocab_size,
                'tgt_vocab_size': self.vocab_size,
                'd_model': config.D_MODEL,
                'nhead': config.NHEAD,
                'num_encoder_layers': config.NUM_ENCODER_LAYERS,
                'num_decoder_layers': config.NUM_DECODER_LAYERS,
                'dim_feedforward': config.DIM_FEEDFORWARD,
                'dropout': config.DROPOUT
            }
        }
        
        # 保存最新检查点
        latest_path = os.path.join(config.CHECKPOINTS_DIR, 'latest_model.pt')
        torch.save(checkpoint, latest_path)
        
        # 保存最佳模型
        if is_best:
            best_path = os.path.join(config.CHECKPOINTS_DIR, f'best_model_bleu{bleu_score:.1f}.pt')
            torch.save(checkpoint, best_path)
            logger.info(f"💾 保存最佳模型: {best_path}")
        
        # 定期保存（基于逻辑步）
        logical_step = self.global_step // config.GRADIENT_ACCUMULATION_STEPS
        if logical_step % config.SAVE_EVERY_LOGICAL_STEPS == 0:
            step_path = os.path.join(config.CHECKPOINTS_DIR, f'model_logical_step_{logical_step}.pt')
            torch.save(checkpoint, step_path)
    
    def train(self):
        """
        主训练循环 - [已集成AMP混合精度训练]
        """
        logger.info("🚀 开始训练 (已启用混合精度AMP)")
        logger.info(f"🎯 目标: BLEU ≥ 25.0")
        logger.info(f"📊 训练步数: {config.TRAIN_STEPS:,}")
        logger.info(f"📊 验证频率: 每 {config.VALIDATE_EVERY_LOGICAL_STEPS} 逻辑步")
        logger.info(f"📊 早停耐心: {config.EARLY_STOPPING_PATIENCE} 逻辑步")
        
        # 1. 初始化 GradScaler (使用新API避免警告)
        # GradScaler 用于自动进行梯度缩放，防止float16梯度下溢
        scaler = torch.amp.GradScaler('cuda')
        
        self.model.train()
        train_iter = itertools.cycle(self.train_loader)
        logical_step = 0  # 逻辑更新步数
        
        # 使用tqdm作为主循环的进度条，监控物理步
        progress_bar = tqdm(range(1, config.TRAIN_STEPS + 1), desc="🚀 训练中", unit="step")
        
        for self.global_step in progress_bar:
            batch_data = next(train_iter)
            train_result = self.train_step_amp(batch_data, scaler)  # 使用AMP版本的训练步骤
            
            # 处理返回值
            if isinstance(train_result, tuple):
                loss, grad_norm = train_result
            else:
                loss = train_result
                grad_norm = None
            
            # ================= 核心修正：所有逻辑都基于参数更新点 =================
            if self.global_step % config.GRADIENT_ACCUMULATION_STEPS == 0:
                logical_step += 1
                
                # --- 更新TensorBoard ---
                self.writer.add_scalar('Training/Loss', loss, logical_step)
                lr = self.optimizer.param_groups[0]['lr']
                self.writer.add_scalar('Training/LearningRate', lr, logical_step)
                if grad_norm is not None:
                    self.writer.add_scalar('Training/GradNorm', grad_norm, logical_step)
                
                # --- 更新进度条描述 ---
                progress_bar.set_postfix({
                    'Logical Step': logical_step,
                    'Loss': f'{loss:.4f}',
                    'LR': f'{lr:.2e}',
                    'Best BLEU': f'{self.best_bleu:.3f}'
                })
                
                # --- 定期日志 (基于逻辑步) ---
                if logical_step > 0 and logical_step % config.LOG_EVERY_LOGICAL_STEPS == 0:
                    elapsed = time.time() - self.start_time
                    progress = self.global_step / config.TRAIN_STEPS * 100
                    steps_per_sec = self.global_step / elapsed if elapsed > 0 else 0
                    eta_seconds = (config.TRAIN_STEPS - self.global_step) / steps_per_sec if steps_per_sec > 0 else 0
                    eta_hours = eta_seconds / 3600
                    
                    logger.info(
                        f"📊 Step {self.global_step:6d}/{config.TRAIN_STEPS} ({progress:5.1f}%) | "
                        f"Logical: {logical_step} | "
                        f"Loss: {loss:.4f} | LR: {lr:.2e} | "
                        f"Speed: {steps_per_sec:.1f} steps/s | "
                        f"ETA: {eta_hours:.1f}h | "
                        f"Best BLEU: {self.best_bleu:.3f}"
                    )
                
                # --- 定期验证 (基于逻辑步) ---
                if logical_step > 0 and logical_step % config.VALIDATE_EVERY_LOGICAL_STEPS == 0:
                    logger.info(f"🔍 开始第 {logical_step} 逻辑步验证评估...")
                    
                    # 在训练初期跳过昂贵的BLEU评估
                    if self.global_step < config.SKIP_BLEU_BEFORE_STEPS:
                        logger.info(f"⏭️ 跳过BLEU评估 (步数 {self.global_step} < {config.SKIP_BLEU_BEFORE_STEPS})")
                        val_loss, _ = self.validate(skip_bleu=True)
                        bleu_score = 0.0
                    else:
                        val_loss, bleu_score = self.validate()
                    
                    # 检查改进（使用配置的最小提升阈值）
                    improvement = bleu_score - self.best_bleu
                    is_best = improvement > config.MIN_DELTA_BLEU
                    
                    logger.info(f"📋 验证结果: Loss={val_loss:.4f}, BLEU={bleu_score:.3f}")
                    
                    if is_best:
                        self.best_bleu = bleu_score
                        self.no_improvement_steps = 0
                        logger.info(f"🎉 新纪录! BLEU: {self.best_bleu:.3f} (+{improvement:.3f})")
                        
                        if self.best_bleu >= 25.0:
                            logger.info(f"🏆 达成目标! BLEU {self.best_bleu:.3f} ≥ 25.0")
                            self.save_checkpoint(bleu_score, val_loss, is_best=True)
                            break
                    else:
                        self.no_improvement_steps += config.VALIDATE_EVERY_LOGICAL_STEPS
                        if improvement > 0:
                            logger.info(f"📈 小幅提升: BLEU {bleu_score:.3f} (+{improvement:.3f}, 需>{config.MIN_DELTA_BLEU:.3f})")
                        else:
                            logger.info(f"📉 无提升: BLEU {bleu_score:.3f} ({improvement:+.3f})")
                    
                    # 保存检查点
                    self.save_checkpoint(bleu_score, val_loss, is_best=is_best)
                    
                    # 早停检查（使用配置的耐心参数，基于逻辑步）
                    patience_logical_steps = config.EARLY_STOPPING_PATIENCE * config.VALIDATE_EVERY_LOGICAL_STEPS
                    if self.no_improvement_steps >= patience_logical_steps:
                        logger.info(f"⏹️ 早停: 连续 {self.no_improvement_steps} 逻辑步无改善 (耐心: {patience_logical_steps} 逻辑步)")
                        break
                    
                    self.model.train()
                
                # --- 定期保存检查点 (基于逻辑步) ---
                if logical_step > 0 and logical_step % config.SAVE_EVERY_LOGICAL_STEPS == 0:
                    self.save_checkpoint(0.0, 0.0, is_best=False)  # 定期保存
            
            # 检查是否达到目标
            if self.best_bleu >= 25.0:
                break
            
            # GPU内存清理
            if self.global_step % 1000 == 0:
                torch.cuda.empty_cache()
        
        # 训练完成
        total_time = time.time() - self.start_time
        logger.info("🏁 训练完成!")
        logger.info(f"🎯 最佳BLEU: {self.best_bleu:.2f}")
        logger.info(f"⏱️ 总训练时间: {total_time/3600:.1f}小时")
        logger.info(f"📊 总训练步数: {self.global_step:,}")
        logger.info(f"📊 总逻辑步数: {logical_step:,}")
        
        if self.best_bleu >= 25.0:
            logger.info("🎉 SUCCESS! 达成BLEU ≥ 25.0 目标!")
        else:
            logger.info("⚠️ 未达目标，建议继续训练或使用模型集成")
        
        self.writer.close()
        return self.best_bleu
    
    def test(self) -> float:
        """
        测试最佳模型
        
        Returns:
            测试集BLEU分数
        """
        logger.info("🧪 测试最佳模型")
        
        # 加载最佳模型
        best_model_path = os.path.join(config.CHECKPOINTS_DIR, 'latest_model.pt')
        if os.path.exists(best_model_path):
            checkpoint = torch.load(best_model_path, map_location=config.DEVICE)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            logger.info(f"📂 加载模型: {best_model_path}")
        else:
            logger.warning("未找到保存的模型，使用当前模型")
        
        # 测试评估
        test_bleu = self.evaluate_bleu(self.test_loader, max_samples=200)
        logger.info(f"📊 测试集BLEU: {test_bleu:.2f}")
        
        return test_bleu

def main():
    """
    主函数 - [已集成命令行参数]
    """
    parser = argparse.ArgumentParser(description='Transformer训练与测试脚本')
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'test', 'resume'],
                       help='运行模式: "train" (训练), "test" (测试), 或 "resume" (恢复训练)')
    parser.add_argument('--checkpoint', type=str, default=None,
                       help='测试或恢复时要加载的检查点路径')
    
    args = parser.parse_args()
    
    print("🚀 Transformer训练 - BLEU 25目标优化版本")
    print("="*80)
    print("📋 项目宪法原则:")
    print("  ✅ 绝对忠于原文精神: 严格遵循'Attention is All You Need'论文")
    print("  ✅ 工程实践至上: 集成所有BLEU提升技术")
    print("  ✅ 性能导向: 目标BLEU ≥ 25.0")
    print("  ✅ 信息透明: 详细日志 + TensorBoard监控")
    print("  ✅ 稳定可靠: 检查点保存 + 早停策略")
    print("="*80)
    print(f"🎯 核心优化技术:")
    print(f"  📈 增强学习率调度: Warmup={config.WARMUP_STEPS}, Scale={config.LEARNING_RATE_SCALE}")
    print(f"  🎭 标签平滑: ε={config.LABEL_SMOOTHING_EPS}")
    print(f"  🔍 Beam Search: size={config.BEAM_SIZE}, α={config.LENGTH_PENALTY}")
    print(f"  📊 梯度累积: {config.GRADIENT_ACCUMULATION_STEPS} 步")
    print(f"  ⏹️ 早停策略: {config.EARLY_STOPPING_PATIENCE} 逻辑步耐心")
    print(f"  💾 检查点: 每 {config.SAVE_EVERY_LOGICAL_STEPS} 逻辑步保存")
    print(f"  🚀 混合精度训练: AMP (RTX 4060优化)")
    print(f"  📊 验证频率: 每 {config.VALIDATE_EVERY_LOGICAL_STEPS} 逻辑步")
    print(f"  📊 日志频率: 每 {config.LOG_EVERY_LOGICAL_STEPS} 逻辑步")
    print("="*80)
    
    try:
        # 将args传给Trainer
        trainer = EnhancedTrainer(resume_from=args.checkpoint if args.mode == 'resume' else None)
        
        if args.mode == 'train' or args.mode == 'resume':
            # 开始训练
            best_bleu = trainer.train()
            # 训练结束后自动测试最佳模型
            logger.info("\n🏁 训练完成，开始使用最佳模型进行最终测试...")
            test_bleu = trainer.test()
            
            print(f"\n🏆 最终结果:")
            print(f"  🎯 最佳验证BLEU: {best_bleu:.2f}")
            print(f"  🧪 测试集BLEU: {test_bleu:.2f}")
            
            if best_bleu >= 25.0:
                print(f"🎉 SUCCESS! 达成BLEU ≥ 25.0 目标!")
            else:
                print(f"⚠️ 未达目标，建议继续训练或使用模型集成")
        
        elif args.mode == 'test':
            if not args.checkpoint:
                logger.error("❌ 测试模式需要指定检查点路径，请使用 --checkpoint /path/to/model.pt")
                return
            
            logger.info(f"🧪 开始测试模式，加载检查点: {args.checkpoint}")
            checkpoint = torch.load(args.checkpoint, map_location=config.DEVICE)
            # 兼容 average_checkpoints.py 的输出格式
            state_dict = checkpoint['model_state_dict'] if 'model_state_dict' in checkpoint else checkpoint
            trainer.model.load_state_dict(state_dict)
            logger.info("✅ 模型加载成功")
            
            # 在测试集上评估
            test_bleu = trainer.evaluate_bleu(trainer.test_loader, max_samples=config.MAX_EVAL_SAMPLES)
            logger.info(f"🏆 测试集最终BLEU分数: {test_bleu:.3f}")
            
            print(f"\n🏆 测试结果:")
            print(f"  🧪 测试集BLEU: {test_bleu:.3f}")
        
    except KeyboardInterrupt:
        logger.info("⚠️ 训练被用户中断")
    except Exception as e:
        logger.error(f"❌ 训练失败: {str(e)}")
        raise

if __name__ == "__main__":
    main()