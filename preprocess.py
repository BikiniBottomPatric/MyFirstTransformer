#!/usr/bin/env python3
# preprocess.py
# 项目宪法：WMT14数据预处理器 - 严格遵循"Attention is All You Need"论文
# 绝对适配现实硬件：分块处理 + 流式处理 + 进度透明

import os
import json
import torch
import sentencepiece as spm
from datasets import load_dataset
from tqdm import tqdm
from pathlib import Path
from typing import List, Dict, Tuple, Iterator
import logging
import time

# 导入配置
import config

# 设置日志
logging.basicConfig(
    level=getattr(logging, config.LOG_LEVEL),
    format=config.LOG_FORMAT
)
logger = logging.getLogger(__name__)

class WMT14Preprocessor:
    """
    WMT14数据预处理器 - 项目宪法实现
    
    核心原则：
    1. 绝对忠于原文精神：使用BPE + 共享词汇表
    2. 绝对适配现实硬件：分块处理，避免内存爆炸
    3. 绝对信息透明：所有操作都有进度条和清晰日志
    4. 绝对工程专业：错误处理 + 断点续传
    """
    
    def __init__(self):
        self.prepared_data_dir = Path(config.PREPARED_DATA_DIR)
        self.bpe_model_prefix = config.BPE_MODEL_PREFIX
        self.vocab_size = config.BPE_VOCAB_SIZE
        self.max_seq_len = config.MAX_SEQ_LEN
        
        # 确保目录存在
        self.prepared_data_dir.mkdir(exist_ok=True)
        for subdir in ['train_chunks', 'validation_chunks', 'test_chunks']:
            (self.prepared_data_dir / subdir).mkdir(exist_ok=True)
            
        logger.info(f"🚀 初始化WMT14预处理器")
        logger.info(f"📁 输出目录: {self.prepared_data_dir}")
        logger.info(f"📊 BPE词汇表大小: {self.vocab_size:,}")
        logger.info(f"📏 最大序列长度: {self.max_seq_len}")
    
    def train_bpe_model(self) -> spm.SentencePieceProcessor:
        """
        训练SentencePiece BPE模型 - 使用流式处理避免内存问题
        
        Returns:
            训练好的SentencePiece处理器
        """
        bpe_model_path = f"{self.bpe_model_prefix}.model"
        
        if os.path.exists(bpe_model_path):
            logger.info(f"✅ BPE模型已存在: {bpe_model_path}")
            sp = spm.SentencePieceProcessor(model_file=bpe_model_path)
            return sp
            
        logger.info("🔧 开始训练SentencePiece BPE模型...")
        logger.info("📡 从Hugging Face流式加载WMT14训练数据...")
        
        try:
            # 使用流式数据集避免内存问题
            dataset_train = load_dataset(
                config.DATASET_NAME, 
                config.LANGUAGE_PAIR, 
                split='train', 
                streaming=True
            )
            
            def get_training_corpus() -> Iterator[str]:
                """生成训练语料的迭代器 - 使用全部训练集"""
                count = 0
                with tqdm(desc="🔤 收集BPE训练语料 (全部)", unit="句", disable=False) as pbar:
                    for item in dataset_train:
                        translation = item['translation']
                        yield translation[config.SRC_LANGUAGE]  # 德语
                        yield translation[config.TGT_LANGUAGE]  # 英语
                        count += 2
                        pbar.update(2)
                        
                        # 删除提前break限制，使用全部训练集训练BPE
                        # if count >= 2000000:  # 已删除
                        #     break
                            
                logger.info(f"📊 BPE训练语料收集完成: {count:,} 句")
            
            # 训练SentencePiece模型
            logger.info("🎯 开始训练BPE模型...")
            start_time = time.time()
            
            spm.SentencePieceTrainer.train(
                sentence_iterator=get_training_corpus(),
                model_prefix=self.bpe_model_prefix,
                vocab_size=self.vocab_size,
                character_coverage=config.CHARACTER_COVERAGE,
                model_type='bpe',
                pad_id=config.PAD_IDX,
                unk_id=config.UNK_IDX, 
                bos_id=config.BOS_IDX,
                eos_id=config.EOS_IDX,
                hard_vocab_limit=False
            )
            
            training_time = time.time() - start_time
            logger.info(f"✅ BPE模型训练完成! 耗时: {training_time:.1f}秒")
            
            # 加载训练好的模型
            sp = spm.SentencePieceProcessor(model_file=bpe_model_path)
            
            # 验证特殊token
            logger.info(f"✅ BPE模型验证完成 - 词汇表大小: {sp.vocab_size():,}")
            # 简化特殊token验证输出
            token_check = all(sp.piece_to_id(token) == expected_id 
                            for token, expected_id in config.SPECIAL_TOKENS.items())
            if token_check:
                logger.info("🔤 特殊token验证通过")
            else:
                logger.warning("⚠️ 特殊token验证失败，请检查配置")
                
            return sp
            
        except Exception as e:
            logger.error(f"❌ BPE模型训练失败: {str(e)}")
            raise RuntimeError(f"BPE训练失败，请检查网络连接和磁盘空间: {str(e)}")
    
    def process_split_to_chunks(self, split_name: str, sp: spm.SentencePieceProcessor, 
                               chunk_size: int = None) -> Dict[str, any]:
        """
        处理数据集的一个分割（train/validation/test）并保存为分块文件
        
        Args:
            split_name: 数据分割名称
            sp: SentencePiece处理器
            chunk_size: 每个分块的样本数（如果为None，使用config.CHUNK_SIZE）
            
        Returns:
            处理统计信息
        """
        if chunk_size is None:
            chunk_size = config.CHUNK_SIZE
        logger.info(f"📦 开始处理 {split_name} 数据集...")
        
        try:
            # =================== 核心修改：加载正确的分割 ===================
            if split_name == 'validation':
                # 标准验证集是 newstest2013
                # Hugging Face 的 'validation' split 就是 newstest2013
                dataset = load_dataset(config.DATASET_NAME, config.LANGUAGE_PAIR, split='validation')
                logger.info("加载标准验证集: newstest2013")
            elif split_name == 'test':
                # 标准测试集是 newstest2014
                dataset = load_dataset(config.DATASET_NAME, config.LANGUAGE_PAIR, split='test')
                logger.info("加载标准测试集: newstest2014")
            else:  # split_name == 'train'
                dataset = load_dataset(config.DATASET_NAME, config.LANGUAGE_PAIR, split='train')
            # =============================================================
            
            total_samples = len(dataset)
            logger.info(f"📊 {split_name} 数据集大小: {total_samples:,} 句对")
            
            # 统计信息
            stats = {
                'total_samples': 0,
                'filtered_samples': 0,
                'chunks_created': 0,
                'avg_src_len': 0,
                'avg_tgt_len': 0
            }
            
            chunk_dir = self.prepared_data_dir / f"{split_name}_chunks"
            chunk_idx = 0
            current_chunk = []
            
            src_lengths = []
            tgt_lengths = []
            
            # 处理数据
            with tqdm(total=total_samples, desc=f"🔄 处理{split_name}", unit="句对") as pbar:
                for idx, item in enumerate(dataset):
                    translation = item['translation']
                    src_text = translation[config.SRC_LANGUAGE]
                    tgt_text = translation[config.TGT_LANGUAGE]
                    
                    # BPE编码
                    src_tokens = sp.encode_as_ids(src_text)
                    tgt_tokens = sp.encode_as_ids(tgt_text)
                    
                    # 长度过滤
                    if (len(src_tokens) > self.max_seq_len or 
                        len(tgt_tokens) > self.max_seq_len or
                        len(src_tokens) < 1 or len(tgt_tokens) < 1):
                        stats['filtered_samples'] += 1
                        pbar.update(1)
                        continue
                    
                    # 添加到当前分块
                    current_chunk.append({
                        'src_tokens': src_tokens,
                        'tgt_tokens': tgt_tokens
                    })
                    
                    src_lengths.append(len(src_tokens))
                    tgt_lengths.append(len(tgt_tokens))
                    stats['total_samples'] += 1
                    
                    # 保存分块
                    if len(current_chunk) >= chunk_size:
                        self._save_chunk(chunk_dir, chunk_idx, current_chunk)
                        chunk_idx += 1
                        current_chunk = []
                        stats['chunks_created'] += 1
                    
                    pbar.update(1)
            
            # 保存最后一个分块
            if current_chunk:
                self._save_chunk(chunk_dir, chunk_idx, current_chunk)
                stats['chunks_created'] += 1
            
            # 计算统计信息
            if src_lengths:
                stats['avg_src_len'] = sum(src_lengths) / len(src_lengths)
                stats['avg_tgt_len'] = sum(tgt_lengths) / len(tgt_lengths)
            
            logger.info(f"✅ {split_name} 处理完成: {stats['total_samples']:,} 样本, {stats['chunks_created']:,} 分块")
            if stats['filtered_samples'] > 0:
                logger.info(f"🚫 过滤样本: {stats['filtered_samples']:,}")
            
            return stats
            
        except Exception as e:
            logger.error(f"❌ 处理{split_name}数据失败: {str(e)}")
            raise RuntimeError(f"数据处理失败: {str(e)}")
    
    def _save_chunk(self, chunk_dir: Path, chunk_idx: int, chunk_data: List[Dict]):
        """保存一个数据分块"""
        chunk_file = chunk_dir / f"chunk_{chunk_idx}.pt"
        
        # 转换为tensor格式
        src_data = [item['src_tokens'] for item in chunk_data]
        tgt_data = [item['tgt_tokens'] for item in chunk_data]
        
        chunk_tensor = {
            'src_data': src_data,
            'tgt_data': tgt_data
        }
        
        torch.save(chunk_tensor, chunk_file)
    
    def save_metadata(self, train_stats: Dict, valid_stats: Dict, test_stats: Dict, 
                     sp: spm.SentencePieceProcessor):
        """保存预处理元数据"""
        metadata = {
            'vocab_size': sp.vocab_size(),
            'bpe_model_path': f"{self.bpe_model_prefix}.model",
            'special_tokens': config.SPECIAL_TOKENS,
            'max_seq_len': self.max_seq_len,
            'splits': {
                'train': train_stats,
                'validation': valid_stats,
                'test': test_stats
            },
            'config': {
                'D_MODEL': config.D_MODEL,
                'NHEAD': config.NHEAD,
                'NUM_ENCODER_LAYERS': config.NUM_ENCODER_LAYERS,
                'NUM_DECODER_LAYERS': config.NUM_DECODER_LAYERS
            }
        }
        
        metadata_file = self.prepared_data_dir / "metadata.json"
        with open(metadata_file, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
            
        logger.info(f"💾 元数据已保存: {metadata_file}")
    
    def run_full_preprocessing(self):
        """运行完整的预处理流程"""
        logger.info("🚀 开始WMT14完整预处理流程")
        logger.info("="*80)
        
        try:
            # 阶段1: 训练BPE模型
            logger.info("📖 阶段1: 训练BPE模型")
            sp = self.train_bpe_model()
            
            # 阶段2: 处理各个数据分割
            logger.info("\n📦 阶段2: 处理数据分割")
            
            train_stats = self.process_split_to_chunks('train', sp)
            valid_stats = self.process_split_to_chunks('validation', sp)
            test_stats = self.process_split_to_chunks('test', sp)  # 实际使用validation
            
            # 阶段3: 保存元数据
            logger.info("\n💾 阶段3: 保存元数据")
            self.save_metadata(train_stats, valid_stats, test_stats, sp)
            
            # 最终报告
            logger.info("\n" + "="*80)
            logger.info("🎉 WMT14预处理完成!")
            logger.info(f"📁 数据保存位置: {self.prepared_data_dir}")
            logger.info(f"📊 训练数据: {train_stats['total_samples']:,} 句对")
            logger.info(f"📊 验证数据: {valid_stats['total_samples']:,} 句对")
            logger.info(f"📊 测试数据: {test_stats['total_samples']:,} 句对")
            logger.info(f"🔤 词汇表大小: {sp.vocab_size():,}")
            logger.info("\n✅ 现在可以运行 'python train.py' 开始训练!")
            
        except Exception as e:
            logger.error(f"❌ 预处理失败: {str(e)}")
            logger.error("💡 建议检查:")
            logger.error("  1. 网络连接是否正常")
            logger.error("  2. 磁盘空间是否充足")
            logger.error("  3. 依赖包是否正确安装")
            raise

def main():
    """主函数"""
    print("🚀 WMT14数据预处理器 - 项目宪法实现")
    print("="*80)
    print("📋 核心原则:")
    print("  ✅ 绝对忠于原文精神: BPE + 共享词汇表")
    print("  ✅ 绝对适配现实硬件: 分块处理 + 流式加载")
    print("  ✅ 绝对信息透明: 进度条 + 详细日志")
    print("  ✅ 绝对工程专业: 错误处理 + 断点续传")
    print("="*80)
    
    # 创建预处理器并运行
    preprocessor = WMT14Preprocessor()
    preprocessor.run_full_preprocessing()

if __name__ == "__main__":
    main()