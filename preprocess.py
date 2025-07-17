# preprocess.py
import torch
import spacy
from datasets import load_dataset
from torchtext.vocab import build_vocab_from_iterator
import config
import os
from tqdm import tqdm
import json
import glob

def main():
    """
    最终优化版：
    该脚本执行所有耗时的数据预处理任务，并在过程中记录元数据，以避免内存峰值。
    """
    print("开始最终版预处理脚本...")

    if not os.path.exists('data'):
        os.makedirs('data')
        print("创建 'data' 目录")

    print("加载Spacy分词器...")
    try:
        spacy_de = spacy.load(config.SPACY_DE)
        spacy_en = spacy.load(config.SPACY_EN)
    except IOError:
        print(f"请先下载spacy模型: python -m spacy download {config.SPACY_DE} && python -m spacy download {config.SPACY_EN}")
        return
    print("Spacy分词器加载完毕。")

    if os.path.exists('data/vocab_src.pt') and os.path.exists('data/vocab_tgt.pt'):
        print("发现已存在的词汇表文件，直接加载。")
        vocab_src = torch.load('data/vocab_src.pt')
        vocab_tgt = torch.load('data/vocab_tgt.pt')
    else:
        print("加载WMT14训练集用于构建词汇表...")
        train_dataset = load_dataset('wmt14', f'{config.SRC_LANGUAGE}-{config.TGT_LANGUAGE}', split='train')
        print("构建源语言 (de) 词汇表...")
        vocab_src = build_vocab_from_iterator(
            ( (tok.text for tok in spacy_de.tokenizer(item['translation'][config.SRC_LANGUAGE])) for item in tqdm(train_dataset, desc="源语言分词") ),
            min_freq=2, specials=config.special_symbols, special_first=True
        )
        vocab_src.set_default_index(config.UNK_IDX)
        torch.save(vocab_src, 'data/vocab_src.pt')
        print("源语言词汇表构建完毕并保存。")

        print("构建目标语言 (en) 词汇表...")
        vocab_tgt = build_vocab_from_iterator(
            ( (tok.text for tok in spacy_en.tokenizer(item['translation'][config.TGT_LANGUAGE])) for item in tqdm(train_dataset, desc="目标语言分词") ),
            min_freq=2, specials=config.special_symbols, special_first=True
        )
        vocab_tgt.set_default_index(config.UNK_IDX)
        torch.save(vocab_tgt, 'data/vocab_tgt.pt')
        print("目标语言词汇表构建完毕并保存。")

    print("开始处理并分块保存完整数据集...")
    chunk_size = 100_000
    all_metadata = {}

    for split in ['train', 'validation', 'test']:
        print(f"处理 {split} 集...")
        dataset_split = load_dataset('wmt14', f'{config.SRC_LANGUAGE}-{config.TGT_LANGUAGE}', split=split)
        
        processed_data = []
        chunk_index = 0
        chunk_lengths = []
        
        for item in tqdm(dataset_split, desc=f"处理 {split} 集"):
            src_tokens = [tok.text for tok in spacy_de.tokenizer(item['translation'][config.SRC_LANGUAGE])]
            tgt_tokens = [tok.text for tok in spacy_en.tokenizer(item['translation'][config.TGT_LANGUAGE])]
            src_ids = [config.BOS_IDX] + [vocab_src[token] for token in src_tokens] + [config.EOS_IDX]
            tgt_ids = [config.BOS_IDX] + [vocab_tgt[token] for token in tgt_tokens] + [config.EOS_IDX]

            processed_data.append({
                'src': torch.tensor(src_ids, dtype=torch.long),
                'tgt': torch.tensor(tgt_ids, dtype=torch.long)
            })

            if len(processed_data) >= chunk_size:
                chunk_path = f'data/{split}_chunk_{chunk_index}.pt'
                torch.save(processed_data, chunk_path)
                chunk_lengths.append(len(processed_data)) # 在保存后立刻记录长度
                processed_data = [] # 释放内存
                chunk_index += 1
        
        if processed_data:
            chunk_path = f'data/{split}_chunk_{chunk_index}.pt'
            torch.save(processed_data, chunk_path)
            chunk_lengths.append(len(processed_data)) # 记录最后一块的长度
        
        all_metadata[split] = {'chunk_lengths': chunk_lengths, 'total_length': sum(chunk_lengths)}
        print(f"'{split}' 集处理完毕。")

    with open('data/metadata.json', 'w') as f:
        json.dump(all_metadata, f)
    print("\n元数据文件 'data/metadata.json' 已成功创建。")

    print("\n所有预处理已成功完成！你可以开始训练了。")

if __name__ == '__main__':
    main() 