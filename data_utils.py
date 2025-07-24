#!/usr/bin/env python3
# data_utils.py (最终优化版)
# 项目宪法：WMT14数据加载器 - 严格遵循"Attention is All You Need"论文
# 绝对适配现实硬件：分块加载 + 内存高效 + 批处理优化

import os
import json
import torch
import random
import math
import numpy as np
import sentencepiece as spm
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from torch.utils.data import Dataset, DataLoader, Sampler
from torch.nn.utils.rnn import pad_sequence
import logging
from tqdm import tqdm

# 导入配置
import config

# 设置日志
logging.basicConfig(
    level=getattr(logging, config.LOG_LEVEL),
    format=config.LOG_FORMAT
)
logger = logging.getLogger(__name__)

class WMT14ChunkedDataset(Dataset):
    """
    WMT14分块数据集 - 项目宪法实现
    
    核心原则：
    1. 绝对忠于原文精神：BOS/EOS处理 + 正确的序列格式
    2. 绝对适配现实硬件：分块加载，避免内存爆炸
    3. 绝对信息透明：加载进度 + 清晰错误信息
    4. 绝对工程专业：缓存机制 + 错误处理
    """
    
    def __init__(self, split_name: str, shuffle_chunks: bool = True):
        """
        初始化分块数据集
        
        Args:
            split_name: 数据分割名称 ('train', 'validation', 'test')
            shuffle_chunks: 是否打乱分块顺序
        """
        self.split_name = split_name
        self.shuffle_chunks = shuffle_chunks
        self.prepared_data_dir = Path(config.PREPARED_DATA_DIR)
        
        # 加载元数据
        self._load_metadata()
        
        # 加载BPE模型
        self._load_bpe_model()
        
        # 发现分块文件
        self._discover_chunks()
        
        # 当前加载的分块
        self.current_chunk_idx = -1
        self.current_chunk_data = []
        self.current_chunk_size = 0
        
        # 全局样本索引映射
        self._build_sample_index()
        
        logger.info(f"🚀 初始化{split_name}数据集")
        logger.info(f"📦 分块数量: {len(self.chunk_files)}")
        logger.info(f"📊 总样本数: {self.total_samples:,}")
        logger.info(f"🔤 词汇表大小: {self.vocab_size:,}")
    
    def _load_metadata(self):
        """加载预处理元数据"""
        metadata_file = self.prepared_data_dir / "metadata.json"
        
        if not metadata_file.exists():
            raise FileNotFoundError(
                f"❌ 元数据文件不存在: {metadata_file}\n"
                f"💡 请先运行 'python preprocess.py' 进行数据预处理"
            )
        
        try:
            with open(metadata_file, 'r', encoding='utf-8') as f:
                self.metadata = json.load(f)
            
            self.vocab_size = self.metadata['vocab_size']
            self.bpe_model_path = self.metadata['bpe_model_path']
            self.special_tokens = self.metadata['special_tokens']
            self.max_seq_len = self.metadata['max_seq_len']
            
            logger.info(f"✅ 元数据加载成功: {metadata_file}")
            
        except Exception as e:
            raise RuntimeError(f"❌ 元数据加载失败: {str(e)}")
    
    def _load_bpe_model(self):
        """加载BPE模型"""
        if not os.path.exists(self.bpe_model_path):
            raise FileNotFoundError(
                f"❌ BPE模型文件不存在: {self.bpe_model_path}\n"
                f"💡 请先运行 'python preprocess.py' 进行数据预处理"
            )
        
        try:
            # 检查是否为Hugging Face tokenizer格式
            if self.bpe_model_path.endswith('.json'):
                from transformers import AutoTokenizer
                tokenizer_dir = os.path.dirname(self.bpe_model_path)
                self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir)
                
                # 验证特殊token（处理token名称映射）
                vocab = self.tokenizer.get_vocab()
                token_mapping = {
                    '<s>': '<bos>',  # 映射<s>到<bos>
                    '</s>': '<eos>'  # 映射</s>到<eos>
                }
                
                for token, expected_id in self.special_tokens.items():
                    # 尝试原始token名称
                    actual_id = vocab.get(token, -1)
                    # 如果找不到，尝试映射的token名称
                    if actual_id == -1 and token in token_mapping:
                        actual_id = vocab.get(token_mapping[token], -1)
                    
                    if actual_id != expected_id:
                        logger.warning(f"⚠️ 特殊token不匹配: {token} 期望={expected_id}, 实际={actual_id}")
                    else:
                        logger.info(f"✅ 特殊token映射正确: {token} -> {actual_id}")
                
                logger.info(f"✅ Hugging Face tokenizer加载成功: {self.bpe_model_path}")
            else:
                # SentencePiece格式
                import sentencepiece as spm
                self.sp = spm.SentencePieceProcessor(model_file=self.bpe_model_path)
                
                # 验证特殊token
                for token, expected_id in self.special_tokens.items():
                    actual_id = self.sp.piece_to_id(token)
                    if actual_id != expected_id:
                        logger.warning(f"⚠️ 特殊token不匹配: {token} 期望={expected_id}, 实际={actual_id}")
                
                logger.info(f"✅ SentencePiece模型加载成功: {self.bpe_model_path}")
            
        except Exception as e:
            raise RuntimeError(f"❌ BPE模型加载失败: {str(e)}")
    
    def _discover_chunks(self):
        """发现分块文件"""
        chunk_dir = self.prepared_data_dir / f"{self.split_name}_chunks"
        
        if not chunk_dir.exists():
            raise FileNotFoundError(
                f"❌ 分块目录不存在: {chunk_dir}\n"
                f"💡 请先运行 'python preprocess.py' 进行数据预处理"
            )
        
        # 发现所有分块文件
        self.chunk_files = sorted(list(chunk_dir.glob("chunk_*.pt")))
        
        if not self.chunk_files:
            raise FileNotFoundError(
                f"❌ 未找到分块文件: {chunk_dir}\n"
                f"💡 请先运行 'python preprocess.py' 进行数据预处理"
            )
        
        # 打乱分块顺序（仅训练时）
        if self.shuffle_chunks and self.split_name == 'train':
            random.shuffle(self.chunk_files)
        
        logger.info(f"📦 发现{len(self.chunk_files)}个分块文件")
    
    def _build_sample_index(self):
        """构建样本索引映射 - 优化版本"""
        logger.info("🔍 构建样本索引...")
        
        # 优化：使用元数据中的信息，避免加载所有分块文件
        if self.split_name in self.metadata.get('splits', {}):
            split_info = self.metadata['splits'][self.split_name]
            self.total_samples = split_info['total_samples']
            chunks_created = split_info['chunks_created']
            
            # 估算每个分块的样本数量（除了最后一个可能不满）
            avg_chunk_size = self.total_samples // chunks_created
            remainder = self.total_samples % chunks_created
            
            self.chunk_sample_counts = [avg_chunk_size] * chunks_created
            if remainder > 0:
                self.chunk_sample_counts[-1] += remainder
            
            logger.info(f"✅ 快速索引构建完成: {self.total_samples:,} 样本 (基于元数据)")
        else:
            # 回退到原始方法（仅在元数据不可用时）
            logger.warning("⚠️ 元数据不完整，使用慢速索引构建...")
            self.chunk_sample_counts = []
            self.total_samples = 0
            
            # 快速扫描每个分块的样本数量
            for chunk_file in tqdm(self.chunk_files, desc="📊 扫描分块"):
                try:
                    chunk_data = torch.load(chunk_file, map_location='cpu')
                    sample_count = len(chunk_data['src_data'])
                    self.chunk_sample_counts.append(sample_count)
                    self.total_samples += sample_count
                    
                except Exception as e:
                    logger.error(f"❌ 分块文件损坏: {chunk_file} - {str(e)}")
                    self.chunk_sample_counts.append(0)
            
            logger.info(f"✅ 索引构建完成: {self.total_samples:,} 样本")
        
        # 构建累积索引
        self.cumulative_counts = [0]
        for count in self.chunk_sample_counts:
            self.cumulative_counts.append(self.cumulative_counts[-1] + count)
    
    def _load_chunk(self, chunk_idx: int):
        """加载指定分块到内存"""
        if chunk_idx == self.current_chunk_idx:
            return  # 已经加载
        
        try:
            chunk_file = self.chunk_files[chunk_idx]
            chunk_data = torch.load(chunk_file, map_location='cpu')
            
            self.current_chunk_data = list(zip(
                chunk_data['src_data'], 
                chunk_data['tgt_data']
            ))
            self.current_chunk_size = len(self.current_chunk_data)
            self.current_chunk_idx = chunk_idx
            
            # 打乱当前分块内的样本（仅训练时）
            if self.shuffle_chunks and self.split_name == 'train':
                random.shuffle(self.current_chunk_data)
            
        except Exception as e:
            logger.error(f"❌ 分块加载失败: {chunk_file} - {str(e)}")
            raise RuntimeError(f"分块加载失败: {str(e)}")
    
    def _find_chunk_and_local_idx(self, global_idx: int) -> Tuple[int, int]:
        """根据全局索引找到对应的分块和局部索引"""
        for chunk_idx in range(len(self.cumulative_counts) - 1):
            if self.cumulative_counts[chunk_idx] <= global_idx < self.cumulative_counts[chunk_idx + 1]:
                local_idx = global_idx - self.cumulative_counts[chunk_idx]
                return chunk_idx, local_idx
        
        raise IndexError(f"全局索引超出范围: {global_idx} >= {self.total_samples}")
    
    def __len__(self) -> int:
        """返回数据集大小"""
        return self.total_samples
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """获取单个样本"""
        if idx >= self.total_samples:
            raise IndexError(f"索引超出范围: {idx} >= {self.total_samples}")
        
        # 找到对应的分块和局部索引
        chunk_idx, local_idx = self._find_chunk_and_local_idx(idx)
        
        # 加载分块（如果需要）
        self._load_chunk(chunk_idx)
        
        # 获取样本
        src_tokens, tgt_tokens = self.current_chunk_data[local_idx]
        
        # 添加BOS/EOS token（遵循论文标准）
        # 源序列：不添加BOS/EOS（在Encoder中处理）
        # 目标序列：添加BOS作为输入，EOS作为标签
        tgt_input = [config.BOS_IDX] + tgt_tokens  # Decoder输入
        tgt_output = tgt_tokens + [config.EOS_IDX]  # Decoder标签
        
        return {
            'src': torch.tensor(src_tokens, dtype=torch.long),
            'tgt_input': torch.tensor(tgt_input, dtype=torch.long),
            'tgt_output': torch.tensor(tgt_output, dtype=torch.long)
        }

class ChunkedDynamicBatchSampler(Sampler):
    """
    分块动态批处理采样器 - 终极性能优化版
    """
    def __init__(self, dataset: WMT14ChunkedDataset, max_tokens: int, shuffle: bool = True):
        self.dataset = dataset
        self.max_tokens = max_tokens
        self.shuffle = shuffle
        logger.info(f"🚀 [ChunkedDynamicBatchSampler] 初始化完成")
        logger.info(f"📊 目标token数/批: {max_tokens:,}")

    def __iter__(self):
        chunk_indices = list(range(len(self.dataset.chunk_files)))
        if self.shuffle:
            random.shuffle(chunk_indices)
        
        for chunk_idx in chunk_indices:
            self.dataset._load_chunk(chunk_idx)
            
            chunk_size = self.dataset.current_chunk_size
            indices_in_chunk = list(range(chunk_size))
            
            lengths_in_chunk = [max(len(src), len(tgt) + 1) for src, tgt in self.dataset.current_chunk_data]
            
            sorted_local_indices = sorted(indices_in_chunk, key=lambda i: lengths_in_chunk[i])
            
            batch_of_indices = []
            current_max_len = 0
            for local_idx in sorted_local_indices:
                global_idx = self.dataset.cumulative_counts[chunk_idx] + local_idx
                seq_len = lengths_in_chunk[local_idx]
                
                # 更新当前批次的最大长度
                if not batch_of_indices:
                    current_max_len = seq_len
                else:
                    current_max_len = max(current_max_len, seq_len)

                if (len(batch_of_indices) + 1) * current_max_len > self.max_tokens:
                    if batch_of_indices:
                        yield batch_of_indices
                    batch_of_indices = [global_idx]
                    current_max_len = seq_len
                else:
                    batch_of_indices.append(global_idx)
            
            if batch_of_indices:
                yield batch_of_indices
    
    def __len__(self):
        # 这是一个合理的估计值，用于tqdm等工具
        # 这个数字不会影响实际的迭代次数
        if not hasattr(self, '_estimated_len'):
            try:
                avg_len = self.dataset.metadata['splits'][self.dataset.split_name]['avg_src_len']
                num_samples = self.dataset.total_samples
                sentences_per_batch = self.max_tokens / avg_len
                self._estimated_len = int(math.ceil(num_samples / sentences_per_batch))
            except:
                self._estimated_len = 65000  # 如果元数据有问题，回退到一个默认值
        return self._estimated_len

def collate_fn(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    """
    批处理整理函数 - 项目宪法实现
    
    核心原则：
    1. 绝对忠于原文精神：正确的填充和掩码
    2. 绝对适配现实硬件：高效的张量操作
    3. 绝对信息透明：清晰的张量维度
    4. 绝对工程专业：错误处理
    
    Args:
        batch: 批次样本列表
        
    Returns:
        Dict: 批处理后的张量字典
    """
    try:
        # 提取各个序列
        src_seqs = [item['src'] for item in batch]
        tgt_input_seqs = [item['tgt_input'] for item in batch]
        tgt_output_seqs = [item['tgt_output'] for item in batch]
        
        # 填充序列（使用PAD_IDX）
        src_padded = pad_sequence(src_seqs, batch_first=True, padding_value=config.PAD_IDX)
        tgt_input_padded = pad_sequence(tgt_input_seqs, batch_first=True, padding_value=config.PAD_IDX)
        tgt_output_padded = pad_sequence(tgt_output_seqs, batch_first=True, padding_value=config.PAD_IDX)
        
        # 创建注意力掩码
        # 源序列掩码：1表示有效token，0表示PAD
        src_mask = (src_padded != config.PAD_IDX)
        
        # 目标序列掩码：1表示有效token，0表示PAD
        tgt_mask = (tgt_input_padded != config.PAD_IDX)
        
        # 创建因果掩码（下三角矩阵）
        tgt_seq_len = tgt_input_padded.size(1)
        causal_mask = torch.tril(torch.ones(tgt_seq_len, tgt_seq_len, dtype=torch.bool))
        
        return {
            'src': src_padded,                    # [batch_size, src_seq_len]
            'tgt_input': tgt_input_padded,        # [batch_size, tgt_seq_len]
            'tgt_output': tgt_output_padded,      # [batch_size, tgt_seq_len]
            'src_mask': src_mask,                 # [batch_size, src_seq_len]
            'tgt_mask': tgt_mask,                 # [batch_size, tgt_seq_len]
            'causal_mask': causal_mask,           # [tgt_seq_len, tgt_seq_len]
            'batch_size': len(batch)
        }
        
    except Exception as e:
        logger.error(f"❌ 批处理整理失败: {str(e)}")
        raise RuntimeError(f"批处理整理失败: {str(e)}")

def create_data_loaders() -> Tuple[DataLoader, DataLoader, DataLoader]:
    logger.info("🚀 创建数据加载器 (已启用分块动态批处理)...")
    try:
        train_dataset = WMT14ChunkedDataset('train', shuffle_chunks=True)
        valid_dataset = WMT14ChunkedDataset('validation', shuffle_chunks=False)
        test_dataset = WMT14ChunkedDataset('test', shuffle_chunks=False)
        
        # 使用标准 DataLoader 配合自定义 batch_sampler
        train_sampler = ChunkedDynamicBatchSampler(train_dataset, config.BATCH_SIZE_TOKENS, shuffle=True)
        
        train_loader = DataLoader(
            train_dataset,
            batch_sampler=train_sampler,
            collate_fn=collate_fn,
            num_workers=config.NUM_WORKERS,
            pin_memory=config.PIN_MEMORY
        )
        
        # 验证和测试集很小，用固定批次大小
        valid_loader = DataLoader(
            valid_dataset,
            batch_size=config.MAX_BATCH_SIZE,
            shuffle=False,
            collate_fn=collate_fn,
            num_workers=config.NUM_WORKERS,
            pin_memory=config.PIN_MEMORY
        )
        
        test_loader = DataLoader(
            test_dataset,
            batch_size=config.MAX_BATCH_SIZE,
            shuffle=False,
            collate_fn=collate_fn,
            num_workers=config.NUM_WORKERS,
            pin_memory=config.PIN_MEMORY
        )
        
        logger.info(f"✅ 数据加载器创建成功")
        logger.info(f"📊 数据集统计:")
        logger.info(f"  训练集: {len(train_dataset):,} 样本")
        logger.info(f"  验证集: {len(valid_dataset):,} 样本")
        logger.info(f"  测试集: {len(test_dataset):,} 样本")
        logger.info(f"📦 分块动态批处理配置:")
        logger.info(f"  目标token数/批: {config.BATCH_SIZE_TOKENS:,}")
        logger.info(f"  最大序列长度: {config.MAX_SEQ_LEN}")
        
        return train_loader, valid_loader, test_loader
    except Exception as e:
        logger.error(f"❌ 数据加载器创建失败: {str(e)}")
        logger.error("💡 建议检查:")
        logger.error("  1. 是否已运行 'python preprocess.py'")
        logger.error("  2. 预处理数据是否完整")
        logger.error("  3. 配置参数是否正确")
        raise RuntimeError(f"数据加载器创建失败: {str(e)}")

def get_vocab_info() -> Dict[str, any]:
    """
    获取词汇表信息
    
    Returns:
        Dict: 词汇表信息字典
    """
    try:
        metadata_file = Path(config.PREPARED_DATA_DIR) / "metadata.json"
        
        if not metadata_file.exists():
            raise FileNotFoundError(
                f"❌ 元数据文件不存在: {metadata_file}\n"
                f"💡 请先运行 'python preprocess.py' 进行数据预处理"
            )
        
        with open(metadata_file, 'r', encoding='utf-8') as f:
            metadata = json.load(f)
        
        vocab_info = {
            'vocab_size': metadata['vocab_size'],
            'special_tokens': metadata['special_tokens'],
            'bpe_model_path': metadata['bpe_model_path'],
            'max_seq_len': metadata['max_seq_len']
        }
        
        logger.info(f"✅ 词汇表信息获取成功: {vocab_info['vocab_size']:,} tokens")
        return vocab_info
        
    except Exception as e:
        logger.error(f"❌ 词汇表信息获取失败: {str(e)}")
        raise RuntimeError(f"词汇表信息获取失败: {str(e)}")

def verify_data_integrity() -> bool:
    """
    验证数据完整性
    
    Returns:
        bool: 数据是否完整
    """
    logger.info("🔍 验证数据完整性...")
    
    try:
        # 检查必要文件
        prepared_data_dir = Path(config.PREPARED_DATA_DIR)
        
        required_files = [
            prepared_data_dir / "metadata.json",
            prepared_data_dir / f"{config.BPE_MODEL_PREFIX}.model"
        ]
        
        required_dirs = [
            prepared_data_dir / "train_chunks",
            prepared_data_dir / "validation_chunks",
            prepared_data_dir / "test_chunks"
        ]
        
        # 检查文件
        for file_path in required_files:
            if not file_path.exists():
                logger.error(f"❌ 缺少必要文件: {file_path}")
                return False
        
        # 检查目录
        for dir_path in required_dirs:
            if not dir_path.exists():
                logger.error(f"❌ 缺少必要目录: {dir_path}")
                return False
            
            # 检查分块文件
            chunk_files = list(dir_path.glob("chunk_*.pt"))
            if not chunk_files:
                logger.error(f"❌ 目录为空: {dir_path}")
                return False
        
        # 尝试创建数据加载器
        try:
            train_loader, valid_loader, test_loader = create_data_loaders()
            
            # 测试加载一个批次
            test_batch = next(iter(train_loader))
            
            logger.info(f"✅ 数据完整性验证通过")
            logger.info(f"📊 测试批次形状:")
            logger.info(f"  src: {test_batch['src'].shape}")
            logger.info(f"  tgt_input: {test_batch['tgt_input'].shape}")
            logger.info(f"  tgt_output: {test_batch['tgt_output'].shape}")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ 数据加载测试失败: {str(e)}")
            return False
        
    except Exception as e:
        logger.error(f"❌ 数据完整性验证失败: {str(e)}")
        return False

def main():
    """主函数 - 用于测试数据加载器"""
    print("🚀 WMT14数据加载器测试 - 项目宪法实现")
    print("="*80)
    
    try:
        # 验证数据完整性
        if not verify_data_integrity():
            print("❌ 数据完整性验证失败")
            print("💡 请先运行 'python preprocess.py' 进行数据预处理")
            return
        
        # 获取词汇表信息
        vocab_info = get_vocab_info()
        print(f"📊 词汇表大小: {vocab_info['vocab_size']:,}")
        
        # 创建数据加载器
        train_loader, valid_loader, test_loader = create_data_loaders()
        
        # 测试数据加载
        print("\n🧪 测试数据加载...")
        for i, batch in enumerate(train_loader):
            if i >= 3:  # 只测试前3个批次
                break
            
            print(f"批次 {i+1}:")
            print(f"  src: {batch['src'].shape} | 非零元素: {(batch['src'] != config.PAD_IDX).sum().item()}")
            print(f"  tgt_input: {batch['tgt_input'].shape} | 非零元素: {(batch['tgt_input'] != config.PAD_IDX).sum().item()}")
            print(f"  tgt_output: {batch['tgt_output'].shape} | 非零元素: {(batch['tgt_output'] != config.PAD_IDX).sum().item()}")
        
        print("\n✅ 数据加载器测试完成!")
        print("🎯 现在可以运行 'python train.py' 开始训练")
        
    except Exception as e:
        print(f"❌ 测试失败: {str(e)}")
        raise

if __name__ == "__main__":
    main()