#!/usr/bin/env python3
# preprocess.py
# 项目宪法：WMT14数据预处理器 - [真正最终的、经过验证的正确版本]

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
import config

logging.basicConfig(level=getattr(logging, config.LOG_LEVEL), format=config.LOG_FORMAT)
logger = logging.getLogger(__name__)

class WMT14Preprocessor:
    def __init__(self):
        self.prepared_data_dir = Path(config.PREPARED_DATA_DIR)
        self.bpe_model_prefix = config.BPE_MODEL_PREFIX
        self.vocab_size = config.BPE_VOCAB_SIZE
        self.max_seq_len = config.MAX_SEQ_LEN
        
        self.prepared_data_dir.mkdir(exist_ok=True)
        for subdir in ['train_chunks', 'validation_chunks', 'test_chunks']:
            (self.prepared_data_dir / subdir).mkdir(exist_ok=True)
            
        logger.info(f"🚀 初始化WMT14预处理器")
        logger.info(f"📁 输出目录: {self.prepared_data_dir}")
        logger.info(f"📊 BPE词汇表大小: {self.vocab_size:,}")
        logger.info(f"📏 最大序列长度: {self.max_seq_len}")
    
    def train_bpe_model(self) -> spm.SentencePieceProcessor:
        bpe_model_path = f"{self.bpe_model_prefix}.model"
        if os.path.exists(bpe_model_path):
            logger.info(f"✅ BPE模型已存在: {bpe_model_path}")
            return spm.SentencePieceProcessor(model_file=bpe_model_path)
            
        logger.info("🔧 开始训练SentencePiece BPE模型...")
        try:
            dataset_train = load_dataset(config.DATASET_NAME, config.LANGUAGE_PAIR, split='train', streaming=True)
            
            def get_training_corpus() -> Iterator[str]:
                with tqdm(desc="🔤 收集BPE训练语料 (全部)", unit="句") as pbar:
                    for item in dataset_train:
                        translation = item['translation']
                        yield translation[config.SRC_LANGUAGE]
                        yield translation[config.TGT_LANGUAGE]
                        pbar.update(2)

            logger.info("🎯 开始训练BPE模型...")
            start_time = time.time()
            spm.SentencePieceTrainer.train(
                sentence_iterator=get_training_corpus(),
                model_prefix=self.bpe_model_prefix,
                vocab_size=self.vocab_size,
                character_coverage=config.CHARACTER_COVERAGE,
                model_type='bpe',
                pad_id=config.PAD_IDX, unk_id=config.UNK_IDX, 
                bos_id=config.BOS_IDX, eos_id=config.EOS_IDX,
                hard_vocab_limit=False
            )
            logger.info(f"✅ BPE模型训练完成! 耗时: {time.time() - start_time:.1f}秒")
            
            sp = spm.SentencePieceProcessor(model_file=bpe_model_path)
            logger.info(f"✅ BPE模型验证完成 - 词汇表大小: {sp.vocab_size():,}")
            return sp
        except Exception as e:
            logger.error(f"❌ BPE模型训练失败: {str(e)}")
            raise

    def process_split_to_chunks(self, split_name: str, sp: spm.SentencePieceProcessor, chunk_size: int = None) -> Dict[str, any]:
        if chunk_size is None:
            chunk_size = config.CHUNK_SIZE
        logger.info(f"📦 开始处理 {split_name} 数据集...")
        
        try:
            # =================== 关键修正 ===================
            # 这个 if/elif/else 结构确保了每个split_name都加载正确的数据源
            if split_name == 'validation':
                dataset = load_dataset(config.DATASET_NAME, config.LANGUAGE_PAIR, split='validation')
                logger.info("加载标准验证集: newstest2013")
            elif split_name == 'test':
                dataset = load_dataset(config.DATASET_NAME, config.LANGUAGE_PAIR, split='test')
                logger.info("加载标准测试集: newstest2014")
            else: # 'train'
                dataset = load_dataset(config.DATASET_NAME, config.LANGUAGE_PAIR, split='train')
            # ===============================================
            
            total_samples = len(dataset)
            stats = {'total_samples': 0, 'filtered_samples': 0, 'chunks_created': 0, 'avg_src_len': 0, 'avg_tgt_len': 0}
            chunk_dir = self.prepared_data_dir / f"{split_name}_chunks"
            chunk_idx, current_chunk, src_lengths, tgt_lengths = 0, [], [], []
            
            with tqdm(total=total_samples, desc=f"🔄 处理{split_name}", unit="句对") as pbar:
                for item in dataset:
                    pbar.update(1)
                    translation = item['translation']
                    src_tokens = sp.encode_as_ids(translation[config.SRC_LANGUAGE])
                    tgt_tokens = sp.encode_as_ids(translation[config.TGT_LANGUAGE])
                    
                    if not (1 <= len(src_tokens) <= self.max_seq_len and 1 <= len(tgt_tokens) <= self.max_seq_len):
                        stats['filtered_samples'] += 1
                        continue
                    
                    current_chunk.append({'src_tokens': src_tokens, 'tgt_tokens': tgt_tokens})
                    src_lengths.append(len(src_tokens))
                    tgt_lengths.append(len(tgt_tokens))
                    stats['total_samples'] += 1
                    
                    if len(current_chunk) >= chunk_size:
                        self._save_chunk(chunk_dir, chunk_idx, current_chunk)
                        chunk_idx += 1
                        current_chunk = []
                        stats['chunks_created'] += 1
            
            if current_chunk:
                self._save_chunk(chunk_dir, chunk_idx, current_chunk)
                stats['chunks_created'] += 1
            
            if src_lengths:
                stats['avg_src_len'] = sum(src_lengths) / len(src_lengths)
                stats['avg_tgt_len'] = sum(tgt_lengths) / len(tgt_lengths)
            
            logger.info(f"✅ {split_name} 处理完成: {stats['total_samples']:,} 样本, {stats['chunks_created']:,} 分块")
            if stats['filtered_samples'] > 0:
                logger.info(f"🚫 过滤样本: {stats['filtered_samples']:,}")
            return stats
        except Exception as e:
            logger.error(f"❌ 处理{split_name}数据失败: {str(e)}")
            raise

    def _save_chunk(self, chunk_dir: Path, chunk_idx: int, chunk_data: List[Dict]):
        chunk_file = chunk_dir / f"chunk_{chunk_idx}.pt"
        torch.save({
            'src_data': [item['src_tokens'] for item in chunk_data],
            'tgt_data': [item['tgt_tokens'] for item in chunk_data]
        }, chunk_file)
    
    def save_metadata(self, train_stats: Dict, valid_stats: Dict, test_stats: Dict, sp: spm.SentencePieceProcessor):
        metadata = {
            'vocab_size': sp.vocab_size(),
            'bpe_model_path': f"{self.bpe_model_prefix}.model",
            'special_tokens': config.SPECIAL_TOKENS,
            'max_seq_len': self.max_seq_len,
            'splits': {'train': train_stats, 'validation': valid_stats, 'test': test_stats},
            'config': {'D_MODEL': config.D_MODEL, 'NHEAD': config.NHEAD,
                       'NUM_ENCODER_LAYERS': config.NUM_ENCODER_LAYERS,
                       'NUM_DECODER_LAYERS': config.NUM_DECODER_LAYERS}
        }
        with open(self.prepared_data_dir / "metadata.json", 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
        logger.info(f"💾 元数据已保存: {self.prepared_data_dir / 'metadata.json'}")
    
    def run_full_preprocessing(self):
        logger.info("🚀 开始WMT14完整预处理流程")
        logger.info("="*80)
        try:
            sp = self.train_bpe_model()
            
            train_stats = self.process_split_to_chunks('train', sp)
            valid_stats = self.process_split_to_chunks('validation', sp)
            test_stats = self.process_split_to_chunks('test', sp)
            
            self.save_metadata(train_stats, valid_stats, test_stats, sp)
            
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
            raise

def main():
    print("🚀 WMT14数据预处理器 - 项目宪法实现")
    print("="*80)
    preprocessor = WMT14Preprocessor()
    preprocessor.run_full_preprocessing()

if __name__ == "__main__":
    main()