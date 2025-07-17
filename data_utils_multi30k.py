# data_utils_multi30k.py
import torch
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence
import config
import os

# --- 全局变量用于缓存加载的词汇表 ---
_vocab_transform = {}
DATA_DIR = 'data_multi30k'

def build_vocab_multi30k():
    """从磁盘加载为Multi30k预先构建好的词汇表对象。"""
    global _vocab_transform
    if _vocab_transform:
        return _vocab_transform
    
    vocab_src_path = os.path.join(DATA_DIR, 'vocab_src.pt')
    vocab_tgt_path = os.path.join(DATA_DIR, 'vocab_tgt.pt')
    
    if not os.path.exists(vocab_src_path) or not os.path.exists(vocab_tgt_path):
        print(f"错误：找不到词汇表文件 '{vocab_src_path}' 或 '{vocab_tgt_path}'。")
        print("请先运行 'python preprocess_multi30k.py'。")
        exit()

    vocab_src = torch.load(vocab_src_path)
    vocab_tgt = torch.load(vocab_tgt_path)
        
    _vocab_transform[config.SRC_LANGUAGE] = vocab_src
    _vocab_transform[config.TGT_LANGUAGE] = vocab_tgt
    print("从磁盘加载 Multi30k 词汇表成功。")
    return _vocab_transform

class Multi30kDataset(Dataset):
    """一个简单的Dataset类，用于加载内存中的Multi30k数据。"""
    def __init__(self, split):
        data_path = os.path.join(DATA_DIR, f'{split}.pt')
        if not os.path.exists(data_path):
            print(f"错误：找不到数据文件 '{data_path}'。")
            print("请先运行 'python preprocess_multi30k.py'。")
            exit()
        self.data = torch.load(data_path)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

def collate_fn(batch):
    """
    为DataLoader整理数据。
    batch是字典的列表, e.g., [{'src': tensor, 'tgt': tensor}, ...]
    """
    src_batch, tgt_batch = [], []
    for item in batch:
        src_batch.append(item['src'])
        tgt_batch.append(item['tgt'])
    
    src_batch = pad_sequence(src_batch, padding_value=config.PAD_IDX, batch_first=False)
    tgt_batch = pad_sequence(tgt_batch, padding_value=config.PAD_IDX, batch_first=False)
    return src_batch, tgt_batch

def get_dataloaders_multi30k():
    """
    为Multi30k的训练和验证集创建并返回DataLoaders。
    """
    # 确保词汇表已经加载
    build_vocab_multi30k() 
    
    dataloaders = {}
    # For Multi30k, we use 'train' and 'validation' splits.
    for split in ['train', 'validation']:
        dataset = Multi30kDataset(split)
        
        shuffle = True if split == 'train' else False
        
        dataloaders[split] = DataLoader(
            dataset, 
            batch_size=config.BATCH_SIZE, 
            collate_fn=collate_fn, 
            shuffle=shuffle, 
            num_workers=config.NUM_WORKERS
        )
    
    print("从预处理文件创建 Multi30k DataLoaders 成功。")
    # Return train and validation loaders. We'll use validation for testing in this quick experiment.
    return dataloaders['train'], dataloaders['validation'] 