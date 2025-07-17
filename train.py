# train.py

import torch
import torch.nn as nn
import torch.optim as optim
import time
import math
import itertools
from torch.utils.tensorboard.writer import SummaryWriter
import torch.nn.functional as F

# 从我们自己的文件中导入
import config
from model import Transformer # 注意：这里导入我们自己写的Transformer
from data_utils import build_vocab, get_dataloaders
from torchtext.data.metrics import bleu_score

# --- 我们将不再使用这个自定义的、有问题的损失函数 ---
# class LabelSmoothingLoss(nn.Module):
#     """
#     实现了标签平滑的交叉熵损失。
#     """
#     def __init__(self, size: int, padding_idx: int, smoothing: float = 0.0):
#         super(LabelSmoothingLoss, self).__init__()
#         # 使用 KL 散度损失作为基础
#         self.criterion = nn.KLDivLoss(reduction='sum')
#         self.padding_idx = padding_idx
#         self.confidence = 1.0 - smoothing
#         self.smoothing = smoothing
#         self.size = size
#         self.true_dist = None
    
#     def forward(self, x: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
#         """
#         x: 模型的对数概率输出 (batch_size * seq_len, vocab_size)
#         target: 真实的标签 (batch_size * seq_len)
#         """
#         assert x.size(1) == self.size
#         # 创建平滑后的目标分布
#         true_dist = x.data.clone()
#         true_dist.fill_(self.smoothing / (self.size - 2))
#         true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
#         # 将 padding 位置的分布设为0
#         true_dist[:, self.padding_idx] = 0
#         mask = torch.nonzero(target.data == self.padding_idx)
#         if mask.dim() > 0:
#             true_dist.index_fill_(0, mask.squeeze(), 0.0)
#         self.true_dist = true_dist
#         # 计算损失
#         log_probs = F.log_softmax(x, dim=-1)
#         return self.criterion(log_probs, true_dist.clone().detach())


def generate_square_subsequent_mask(sz: int) -> torch.Tensor:
    """为解码器的自注意力生成一个上三角掩码"""
    # 返回一个布尔类型的掩码，True代表需要屏蔽的位置
    mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    return mask

def create_mask(src: torch.Tensor, tgt: torch.Tensor):
    """为源序列和目标序列创建所有需要的掩码"""
    src_seq_len = src.shape[0]
    tgt_seq_len = tgt.shape[0]

    # 解码器自注意力掩码，防止看到未来的词
    tgt_mask = generate_square_subsequent_mask(tgt_seq_len).to(config.DEVICE)
    # 编码器自注意力掩码，全False，不屏蔽任何东西
    # PyTorch的TransformerEncoder不需要这个src_mask
    # src_mask = torch.zeros((src_seq_len, src_seq_len), device=config.DEVICE).type(torch.bool)

    # Encoder和Decoder的padding mask, True代表需要屏蔽的位置
    src_padding_mask = (src == config.PAD_IDX).transpose(0, 1)
    tgt_padding_mask = (tgt == config.PAD_IDX).transpose(0, 1)
    return tgt_mask, src_padding_mask, tgt_padding_mask


def train_epoch(model, optimizer, criterion, dataloader, current_step):
    model.train()
    total_loss = 0
    
    for i, (src, tgt) in enumerate(dataloader):
        current_step += 1
        src, tgt = src.to(config.DEVICE), tgt.to(config.DEVICE)
        
        tgt_input = tgt[:-1, :]
        
        tgt_mask, src_padding_mask, tgt_padding_mask = create_mask(src, tgt_input)
        
        # --- 根据论文公式更新学习率 ---
        lr = (config.D_MODEL ** -0.5) * min(current_step ** -0.5, current_step * (config.WARMUP_STEPS ** -1.5))
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        
        optimizer.zero_grad()
        
        logits = model(src=src, 
                       tgt=tgt_input, 
                       tgt_mask=tgt_mask, 
                       src_key_padding_mask=src_padding_mask, 
                       tgt_key_padding_mask=tgt_padding_mask, # 传递目标序列的padding mask
                       memory_key_padding_mask=src_padding_mask) # memory mask与src mask相同
        
        tgt_out = tgt[1:, :] # 真实目标：去掉<bos>
        loss = criterion(logits.reshape(-1, logits.shape[-1]), tgt_out.reshape(-1))
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()

    return total_loss / len(list(iter(dataloader))), current_step

def evaluate(model, criterion, dataloader, vocab_transform):
    model.eval()
    total_loss = 0
    generated_translations = []
    ground_truth_translations = []

    with torch.no_grad():
        for src, tgt in dataloader:
            src, tgt = src.to(config.DEVICE), tgt.to(config.DEVICE)
            
            tgt_input = tgt[:-1, :]
            
            tgt_mask, src_padding_mask, tgt_padding_mask = create_mask(src, tgt_input)
            
            logits = model(src=src, 
                           tgt=tgt_input, 
                           tgt_mask=tgt_mask, 
                           src_key_padding_mask=src_padding_mask, 
                           tgt_key_padding_mask=tgt_padding_mask, # 传递目标序列的padding mask
                           memory_key_padding_mask=src_padding_mask)
            
            tgt_out = tgt[1:, :]
            loss = criterion(logits.reshape(-1, logits.shape[-1]), tgt_out.reshape(-1))
            total_loss += loss.item()

            # --- BLEU Score Calculation ---
            # Get model predictions by taking the argmax
            pred_tokens = logits.argmax(dim=-1).transpose(0, 1) # -> [batch_size, seq_len]
            
            # Convert token indices back to text
            # 直接使用传入的vocab_transform
            tgt_itos = vocab_transform[config.TGT_LANGUAGE].get_itos()

            for i in range(pred_tokens.shape[0]):
                pred_seq = [tgt_itos[token] for token in pred_tokens[i] if token not in [config.PAD_IDX, config.BOS_IDX, config.EOS_IDX]]
                # Clean up sentence until <eos> is found
                try:
                    eos_index = pred_seq.index('<eos>')
                    pred_seq = pred_seq[:eos_index]
                except ValueError:
                    pass # No <eos> found
                generated_translations.append(pred_seq)
            
            for i in range(tgt_out.shape[1]):
                true_seq = [tgt_itos[token] for token in tgt_out[:, i] if token not in [config.PAD_IDX, config.BOS_IDX, config.EOS_IDX]]
                ground_truth_translations.append([true_seq]) # bleu_score expects a list of references

    # Calculate BLEU score for the entire validation set
    if generated_translations and ground_truth_translations:
        total_bleu_score = bleu_score(generated_translations, ground_truth_translations)
            
    return total_loss / len(list(iter(dataloader))), total_bleu_score

def run():
    # 1. 数据准备
    train_loader, valid_loader, test_loader = get_dataloaders()
    vocab_transform = build_vocab()
    src_vocab_size = len(vocab_transform[config.SRC_LANGUAGE])
    tgt_vocab_size = len(vocab_transform[config.TGT_LANGUAGE])
    
    model = Transformer(
        src_vocab_size,
        tgt_vocab_size,
        config.D_MODEL,
        config.NHEAD,
        config.NUM_ENCODER_LAYERS,
        config.NUM_DECODER_LAYERS,
        config.DIM_FEEDFORWARD,
        config.DROPOUT
    ).to(config.DEVICE)
    
    # 3. 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss(label_smoothing=config.LABEL_SMOOTHING_EPS, 
                                    ignore_index=config.PAD_IDX)
    optimizer = optim.Adam(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9)
    
    writer = SummaryWriter(comment="_wmt14")
    print("TensorBoard logs are being written to './runs'. Use `tensorboard --logdir ./runs` to visualize them.")

    # 4. 训练循环 (基于步数)
    print("Starting training by steps...")
    current_step = 0
    train_iter = itertools.cycle(train_loader) # 创建无限迭代器

    while current_step < config.WMT_TRAIN_STEPS:
        
        src, tgt = next(train_iter)
        src, tgt = src.to(config.DEVICE), tgt.to(config.DEVICE)
        
        tgt_input = tgt[:-1, :]
        
        tgt_mask, src_padding_mask, tgt_padding_mask = create_mask(src, tgt_input)
        
        # --- 根据论文公式更新学习率 ---
        current_step += 1 # 步数在这里增加
        lr = (config.D_MODEL ** -0.5) * min(current_step ** -0.5, current_step * (config.WARMUP_STEPS ** -1.5))
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        
        optimizer.zero_grad()
        
        logits = model(src=src, 
                       tgt=tgt_input, 
                       tgt_mask=tgt_mask, 
                       src_key_padding_mask=src_padding_mask, 
                       tgt_key_padding_mask=tgt_padding_mask,
                       memory_key_padding_mask=src_padding_mask)
        
        tgt_out = tgt[1:, :]
        loss = criterion(logits.reshape(-1, logits.shape[-1]), tgt_out.reshape(-1))
        
        loss.backward()
        optimizer.step()
        
        # --- 定期验证和记录日志 ---
        if current_step % config.WMT_VALIDATE_EVERY_N_STEPS == 0:
            start_time = time.time()
            
            # 使用 evaluate 函数计算验证集损失和BLEU
            val_loss, val_bleu = evaluate(model, criterion, valid_loader, vocab_transform)
            
            end_time = time.time()
            val_time = end_time - start_time

            # 由于训练损失是每个step都计算，而验证是N个step才一次
            # 为了对齐，我们只记录验证时刻的单步训练损失
            train_loss = loss.item()
            
            print(f'Step: {current_step}/{config.WMT_TRAIN_STEPS} | Val Time: {val_time:.2f}s')
            print(f'\tTrain Loss: {train_loss:.3f} | Train PPL: {math.exp(train_loss):7.3f}')
            print(f'\t Val. Loss: {val_loss:.3f} |  Val. PPL: {math.exp(val_loss):7.3f} | Val. BLEU: {val_bleu*100:.2f}')
            
            # 将指标写入 TensorBoard
            writer.add_scalar('Loss/train', train_loss, current_step)
            writer.add_scalar('Loss/val', val_loss, current_step)
            writer.add_scalar('Perplexity/train', math.exp(train_loss), current_step)
            writer.add_scalar('Perplexity/val', math.exp(val_loss), current_step)
            writer.add_scalar('BLEU/val', val_bleu, current_step)
            writer.add_scalar('learning_rate', lr, current_step)

    # 训练结束后，在测试集上进行最终评估
    test_loss, test_bleu = evaluate(model, criterion, test_loader, vocab_transform)
    print("\n--- 训练完成 ---")
    print(f'Test Loss: {test_loss:.3f} | Test PPL: {math.exp(test_loss):7.3f} | Test BLEU: {test_bleu*100:.2f}')

    writer.close()

if __name__ == '__main__':
    print(f"Running on device: {config.DEVICE}")
    run()