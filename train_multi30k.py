# train_multi30k.py

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
from model import Transformer 
# --- 注意：导入为Multi30k准备的新工具 ---
from data_utils_multi30k import build_vocab_multi30k, get_dataloaders_multi30k
from torchtext.data.metrics import bleu_score

# --- 复用train.py中的辅助模块 ---

def generate_square_subsequent_mask(sz: int) -> torch.Tensor:
    """为解码器的自注意力生成一个上三角掩码"""
    mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    return mask

def create_mask(src: torch.Tensor, tgt: torch.Tensor):
    """为源序列和目标序列创建所有需要的掩码"""
    src_seq_len = src.shape[0]
    tgt_seq_len = tgt.shape[0]

    # 解码器自注意力掩码，防止看到未来的词
    tgt_mask = generate_square_subsequent_mask(tgt_seq_len).to(config.DEVICE)

    # Encoder和Decoder的padding mask, True代表需要屏蔽的位置
    src_padding_mask = (src == config.PAD_IDX).transpose(0, 1)
    tgt_padding_mask = (tgt == config.PAD_IDX).transpose(0, 1)
    return tgt_mask, src_padding_mask, tgt_padding_mask


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
                           tgt_key_padding_mask=tgt_padding_mask,
                           memory_key_padding_mask=src_padding_mask)
            
            tgt_out = tgt[1:, :]
            loss = criterion(logits.reshape(-1, logits.shape[-1]), tgt_out.reshape(-1))
            total_loss += loss.item()

            pred_tokens = logits.argmax(dim=-1).transpose(0, 1)
            tgt_itos = vocab_transform[config.TGT_LANGUAGE].get_itos()

            for i in range(pred_tokens.shape[0]):
                pred_seq = [tgt_itos[token] for token in pred_tokens[i] if token not in [config.PAD_IDX, config.BOS_IDX, config.EOS_IDX]]
                try:
                    eos_index = pred_seq.index('<eos>')
                    pred_seq = pred_seq[:eos_index]
                except ValueError:
                    pass
                generated_translations.append(pred_seq)
            
            for i in range(tgt_out.shape[1]):
                true_seq = [tgt_itos[token] for token in tgt_out[:, i] if token not in [config.PAD_IDX, config.BOS_IDX, config.EOS_IDX]]
                ground_truth_translations.append([true_seq])

    total_bleu_score = 0
    if generated_translations and ground_truth_translations:
        total_bleu_score = bleu_score(generated_translations, ground_truth_translations)
            
    return total_loss / len(list(iter(dataloader))), total_bleu_score

def run():
    print("--- Starting Multi30k Experiment ---")
    
    # 1. 数据准备
    train_loader, valid_loader = get_dataloaders_multi30k()
    vocab_transform = build_vocab_multi30k()
    src_vocab_size = len(vocab_transform[config.SRC_LANGUAGE])
    tgt_vocab_size = len(vocab_transform[config.TGT_LANGUAGE])
    
    # 2. 模型初始化 (与主训练脚本完全相同)
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
    
    # 3. 损失函数和优化器
    # 使用PyTorch内置的、支持标签平滑的CrossEntropyLoss，这是一个更稳定和推荐的做法
    criterion = nn.CrossEntropyLoss(label_smoothing=config.LABEL_SMOOTHING_EPS, 
                                    ignore_index=config.PAD_IDX)

    optimizer = optim.Adam(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9)
    
    # 初始化 TensorBoard 记录器，并为本次运行添加一个独特的后缀
    writer = SummaryWriter(comment="_multi30k")
    print("TensorBoard logs for this run are in './runs/*_multi30k'.")

    # 4. 训练循环 (基于步数)
    print("Starting training by steps...")
    model.train()
    
    # itertools.cycle 会无限循环我们的dataloader
    train_iterator = iter(itertools.cycle(train_loader))
    
    total_loss = 0
    start_time = time.time()
    
    for step in range(1, config.TRAIN_STEPS + 1):
        # 从无限迭代器中获取下一个批次
        src, tgt = next(train_iterator)
        src, tgt = src.to(config.DEVICE), tgt.to(config.DEVICE)
        
        tgt_input = tgt[:-1, :]
        
        tgt_mask, src_padding_mask, tgt_padding_mask = create_mask(src, tgt_input)
        
        # --- 根据论文公式在每一步更新学习率 ---
        lr = (config.D_MODEL ** -0.5) * min(step ** -0.5, step * (config.WARMUP_STEPS ** -1.5))
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
        
        total_loss += loss.item()
        
        # --- 定期评估和打印日志 ---
        if step % config.VALIDATE_EVERY_N_STEPS == 0:
            end_time = time.time()
            time_per_step_interval = (end_time - start_time) / config.VALIDATE_EVERY_N_STEPS
            
            avg_train_loss = total_loss / config.VALIDATE_EVERY_N_STEPS
            train_ppl = math.exp(avg_train_loss)
            
            # 在验证集上评估
            model.eval() # 切换到评估模式
            val_loss, val_bleu = evaluate(model, criterion, valid_loader, vocab_transform)
            val_ppl = math.exp(val_loss)
            model.train() # 切换回训练模式
            
            print(f'Step: {step:7d}/{config.TRAIN_STEPS} | Time/Step: {time_per_step_interval:.2f}s')
            print(f'\tTrain Loss: {avg_train_loss:.3f} | Train PPL: {train_ppl:7.3f}')
            print(f'\t Val. Loss: {val_loss:.3f} |  Val. PPL: {val_ppl:7.3f} | Val. BLEU: {val_bleu*100:.2f}')
            
            # 将指标写入 TensorBoard
            writer.add_scalar('Loss/train', avg_train_loss, step)
            writer.add_scalar('Loss/val', val_loss, step)
            writer.add_scalar('Perplexity/train', train_ppl, step)
            writer.add_scalar('Perplexity/val', val_ppl, step)
            writer.add_scalar('BLEU/val', val_bleu, step)
            writer.add_scalar('learning_rate', lr, step)

            # 重置计数器
            total_loss = 0
            start_time = time.time()

    print("\n--- Multi30k Experiment Finished ---")
    
    # 最终评估
    final_val_loss, final_val_bleu = evaluate(model, criterion, valid_loader, vocab_transform)
    print(f'Final Validation Loss: {final_val_loss:.3f} | Final Validation PPL: {math.exp(final_val_loss):7.3f} | Final Validation BLEU: {final_val_bleu*100:.2f}')
    
    writer.close()

if __name__ == '__main__':
    print(f"Running on device: {config.DEVICE}")
    run() 