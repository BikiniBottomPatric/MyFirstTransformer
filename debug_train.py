# debug_train.py
# 这个文件是 train.py 的一个副本，专门用于快速调试和实验。
# 它只加载一小部分数据，让你可以在几分钟内跑完一个epoch，
# 从而快速验证参数修改的效果，而不用等待数小时。

import torch
import torch.nn as nn
import torch.optim as optim
import time
import math
from torch.utils.tensorboard.writer import SummaryWriter
import torch.utils.data as data

# 从我们自己的文件中导入
import config
from model import Transformer
from data_utils import build_vocab, get_dataloaders
from torchtext.data.metrics import bleu_score

# ----------------- 快速实验的核心改动 -----------------
# 定义一个我们想要使用的数据比例
DATA_SUBSET_FRACTION = 0.01 # 只使用1%的数据进行快速实验
# ----------------------------------------------------

class LabelSmoothingLoss(nn.Module):
    def __init__(self, size: int, padding_idx: int, smoothing: float = 0.0):
        super(LabelSmoothingLoss, self).__init__()
        self.criterion = nn.KLDivLoss(reduction='sum')
        self.padding_idx = padding_idx
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.size = size
        self.true_dist = None
    
    def forward(self, x: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        assert x.size(1) == self.size
        true_dist = x.data.clone()
        true_dist.fill_(self.smoothing / (self.size - 2))
        true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        true_dist[:, self.padding_idx] = 0
        mask = torch.nonzero(target.data == self.padding_idx)
        if mask.dim() > 0:
            true_dist.index_fill_(0, mask.squeeze(), 0.0)
        self.true_dist = true_dist
        return self.criterion(x, true_dist.clone().detach())

def generate_square_subsequent_mask(sz: int) -> torch.Tensor:
    mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
    return mask

def create_mask(src: torch.Tensor, tgt: torch.Tensor):
    src_seq_len = src.shape[0]
    tgt_seq_len = tgt.shape[0]
    tgt_mask = generate_square_subsequent_mask(tgt_seq_len).to(config.DEVICE)
    src_mask = torch.zeros((src_seq_len, src_seq_len), device=config.DEVICE).type(torch.bool)
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
        
        lr = (config.D_MODEL ** -0.5) * min(current_step ** -0.5, current_step * (config.WARMUP_STEPS ** -1.5))
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        
        optimizer.zero_grad()
        logits = model(src=src, tgt=tgt_input, tgt_mask=tgt_mask, 
                       src_key_padding_mask=src_padding_mask, 
                       tgt_key_padding_mask=tgt_padding_mask,
                       memory_key_padding_mask=src_padding_mask)
        
        tgt_out = tgt[1:, :]
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
            
            logits = model(src=src, tgt=tgt_input, tgt_mask=tgt_mask, 
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
                except ValueError: pass
                generated_translations.append(pred_seq)
            
            for i in range(tgt_out.shape[1]):
                true_seq = [tgt_itos[token] for token in tgt_out[:, i] if token not in [config.PAD_IDX, config.BOS_IDX, config.EOS_IDX]]
                ground_truth_translations.append([true_seq])

    if generated_translations and ground_truth_translations:
        total_bleu_score = bleu_score(generated_translations, ground_truth_translations)
    else:
        total_bleu_score = 0.0
            
    return total_loss / len(list(iter(dataloader))), total_bleu_score

def run():
    print(f"--- 快速调试模式 ---")
    print(f"--- 将只使用 {DATA_SUBSET_FRACTION*100:.1f}% 的数据 ---")

    # 1. 正常加载全部数据
    full_train_loader, full_valid_loader, full_test_loader = get_dataloaders()

    # 2. 创建原始数据集的子集
    train_subset_size = int(len(full_train_loader.dataset) * DATA_SUBSET_FRACTION)
    valid_subset_size = int(len(full_valid_loader.dataset) * DATA_SUBSET_FRACTION)
    test_subset_size = int(len(full_test_loader.dataset) * DATA_SUBSET_FRACTION)
    
    # 保证子集至少有一个样本
    train_subset_size = max(1, train_subset_size)
    valid_subset_size = max(1, valid_subset_size)
    test_subset_size = max(1, test_subset_size)

    train_subset, _ = data.random_split(full_train_loader.dataset, [train_subset_size, len(full_train_loader.dataset) - train_subset_size])
    valid_subset, _ = data.random_split(full_valid_loader.dataset, [valid_subset_size, len(full_valid_loader.dataset) - valid_subset_size])
    test_subset, _ = data.random_split(full_test_loader.dataset, [test_subset_size, len(full_test_loader.dataset) - test_subset_size])
    
    # 3. 用子集创建新的DataLoader
    train_loader = data.DataLoader(train_subset, batch_size=config.BATCH_SIZE, collate_fn=full_train_loader.collate_fn, shuffle=True)
    valid_loader = data.DataLoader(valid_subset, batch_size=config.BATCH_SIZE, collate_fn=full_valid_loader.collate_fn)
    test_loader = data.DataLoader(test_subset, batch_size=config.BATCH_SIZE, collate_fn=full_test_loader.collate_fn)
    
    print(f"训练集样本数: {train_subset_size}, 验证集样本数: {valid_subset_size}")


    vocab_transform = build_vocab()
    src_vocab_size = len(vocab_transform[config.SRC_LANGUAGE])
    tgt_vocab_size = len(vocab_transform[config.TGT_LANGUAGE])
    
    model = Transformer(src_vocab_size, tgt_vocab_size, config.D_MODEL, config.NHEAD, 
                        config.NUM_ENCODER_LAYERS, config.NUM_DECODER_LAYERS, 
                        config.DIM_FEEDFORWARD, config.DROPOUT).to(config.DEVICE)
    
    criterion = LabelSmoothingLoss(size=tgt_vocab_size, padding_idx=config.PAD_IDX, smoothing=config.LABEL_SMOOTHING_EPS)
    optimizer = optim.Adam(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9)
    
    writer = SummaryWriter(comment="_debug_run")
    print("TensorBoard logs are being written to './runs'.")
    
    print("Starting debug training...")
    step_num = 0
    for epoch in range(1, config.NUM_EPOCHS + 1):
        start_time = time.time()
        train_loss, step_num = train_epoch(model, optimizer, criterion, train_loader, step_num)
        val_loss, val_bleu = evaluate(model, criterion, valid_loader, vocab_transform)
        end_time = time.time()
        
        epoch_mins, epoch_secs = divmod(end_time - start_time, 60)
        
        print(f'Epoch: {epoch:02} | Time: {int(epoch_mins)}m {int(epoch_secs)}s')
        print(f'\tTrain Loss: {train_loss:.3f} | Train PPL: {math.exp(train_loss):7.3f}')
        print(f'\t Val. Loss: {val_loss:.3f} |  Val. PPL: {math.exp(val_loss):7.3f} | Val. BLEU: {val_bleu*100:.2f}')
        
        writer.add_scalar('Loss/train_debug', train_loss, epoch)
        writer.add_scalar('Loss/val_debug', val_loss, epoch)
        writer.add_scalar('Perplexity/train_debug', math.exp(train_loss), epoch)
        writer.add_scalar('Perplexity/val_debug', math.exp(val_loss), epoch)
        writer.add_scalar('BLEU/val_debug', val_bleu, epoch)

    test_loss, test_bleu = evaluate(model, criterion, test_loader, vocab_transform)
    print("\n--- 调试训练完成 ---")
    print(f'Test Loss: {test_loss:.3f} | Test PPL: {math.exp(test_loss):7.3f} | Test BLEU: {test_bleu*100:.2f}')

    writer.close()

if __name__ == '__main__':
    print(f"Running on device: {config.DEVICE}")
    run() 