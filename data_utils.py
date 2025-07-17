# data_utils.py
import torch
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence
import config
import os
import glob
import json

# --- 全局变量用于缓存加载的词汇表 ---
_vocab_transform = {}

class PreprocessedDataset(Dataset):
    """
    一个更智能的Dataset类，用于处理分块存储的预处理数据。
    它会按需加载数据块，以节省内存。
    """
    def __init__(self, split):
        self.split = split
        # 查找所有属于该split的块文件
        self.chunk_files = sorted(glob.glob(f'data/{self.split}_chunk_*.pt'))
        
        # 从元数据文件加载长度信息，而不是手动计算
        metadata_path = 'data/metadata.json'
        if not os.path.exists(metadata_path):
            raise FileNotFoundError(f"未找到元数据文件 '{metadata_path}'。请先运行 'preprocess.py'。")
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        if self.split not in metadata:
            raise ValueError(f"在元数据中未找到 '{self.split}' 的信息。请重新运行 'preprocess.py'。")
            
        self.chunk_lengths = metadata[self.split]['chunk_lengths']
        self.total_length = metadata[self.split]['total_length']
        
        if len(self.chunk_files) != len(self.chunk_lengths):
            raise RuntimeError("数据块文件数量与元数据记录不匹配。请删除 'data' 目录并重新运行 'preprocess.py'。")

        self.cumulative_lengths = [sum(self.chunk_lengths[:i+1]) for i in range(len(self.chunk_lengths))]
        
        # 用于缓存最近加载的块，避免频繁IO
        self._cache = {}

    def __len__(self):
        return self.total_length

    def __getitem__(self, idx):
        # 确定索引idx属于哪个块
        chunk_index = 0
        while idx >= self.cumulative_lengths[chunk_index]:
            chunk_index += 1
        
        # 计算在块内的局部索引
        if chunk_index == 0:
            local_idx = idx
        else:
            local_idx = idx - self.cumulative_lengths[chunk_index - 1]
            
        # 检查缓存，如果块不在缓存中，则加载它
        if chunk_index not in self._cache:
            # 为了节省内存，只缓存最新的块
            self._cache.clear()
            self._cache[chunk_index] = torch.load(self.chunk_files[chunk_index])
            
        return self._cache[chunk_index][local_idx]

def build_vocab():
    """从磁盘加载预先构建好的词汇表对象。"""
    global _vocab_transform
    if _vocab_transform:
        return _vocab_transform
    
    try:
        vocab_src = torch.load('data/vocab_src.pt')
        vocab_tgt = torch.load('data/vocab_tgt.pt')
    except FileNotFoundError:
        print("错误：找不到词汇表文件 'data/vocab_src.pt' 或 'data/vocab_tgt.pt'。")
        print("请先运行 'python preprocess.py' 来生成预处理文件。")
        exit()
        
    _vocab_transform[config.SRC_LANGUAGE] = vocab_src
    _vocab_transform[config.TGT_LANGUAGE] = vocab_tgt
    print("从磁盘加载词汇表成功。")
    return _vocab_transform

def collate_fn(batch):
    """
    为DataLoader整理数据。
    batch是字典的列表, e.g., [{'src': tensor, 'tgt': tensor}, ...]
    """
    src_batch, tgt_batch = [], []
    for item in batch:
        src_batch.append(item['src'])
        tgt_batch.append(item['tgt'])
    
    # 将tensor列表填充为统一长度的batch
    # batch_first=False 使输出形状为 [seq_len, batch_size]
    src_batch = pad_sequence(src_batch, padding_value=config.PAD_IDX, batch_first=False)
    tgt_batch = pad_sequence(tgt_batch, padding_value=config.PAD_IDX, batch_first=False)
    return src_batch, tgt_batch

def get_dataloaders():
    """
    为训练、验证和测试集创建并返回最终的DataLoaders。
    此函数现在只从磁盘加载预处理好的数据。
    """
    # 确保词汇表已经加载，虽然在这里不直接使用，但在训练脚本中需要
    build_vocab() 
    
    dataloaders = {}
    for split in ['train', 'validation', 'test']:
        # PreprocessedDataset会自己检查文件是否存在，我们在这里不需要重复检查
        try:
            dataset = PreprocessedDataset(split)
        except FileNotFoundError as e:
            print(e)
            exit()
        
        # 为训练集启用shuffle
        shuffle = True if split == 'train' else False
        
        # 创建DataLoader
        dataloaders[split] = DataLoader(
            dataset, 
            batch_size=config.BATCH_SIZE, 
            collate_fn=collate_fn, 
            shuffle=shuffle, 
            num_workers=config.NUM_WORKERS
        )
    
    print("从预处理文件创建DataLoaders成功。")
    # 我们需要返回3个dataloader，而不是一个字典
    return dataloaders['train'], dataloaders['validation'], dataloaders['test']