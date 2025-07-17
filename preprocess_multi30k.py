# preprocess_multi30k.py
import torch
import os
import gzip # 导入gzip库来处理.gz文件
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
import config # Reuse config for special symbols

# --- 配置 ---
SRC_LANGUAGE = config.SRC_LANGUAGE
TGT_LANGUAGE = config.TGT_LANGUAGE
# 数据现在直接从克隆下来的 'dataset' 文件夹中读取，注意文件名是 .gz
DATA_PATH_TEMPLATE = 'dataset/data/task1/raw/{split}.{lang}.gz'

def read_lines(filepath):
    """从 .gz 压缩文件中读取所有行。"""
    if not os.path.exists(filepath):
        print(f"错误：找不到数据文件 '{filepath}'。")
        print("请确认你已经成功运行 'git clone https://github.com/multi30k/dataset.git'")
        exit()
    # 使用gzip.open来读取压缩文件
    with gzip.open(filepath, 'rt', encoding='utf-8') as f:
        return [line.strip() for line in f]

def build_vocabulary(tokenizer, data_iter):
    """从数据迭代器构建词汇表"""
    return build_vocab_from_iterator(
        (tokenizer(item) for item in data_iter),
        specials=config.special_symbols,
        special_first=True,
    )

def process_data(src_lines, tgt_lines, vocab_src, vocab_tgt, tokenizer_src, tokenizer_tgt):
    """将原始文本数据转换为Tensor"""
    processed_data = []
    for src_text, tgt_text in zip(src_lines, tgt_lines):
        src_tensor = torch.tensor(
            [config.BOS_IDX] + vocab_src(tokenizer_src(src_text)) + [config.EOS_IDX],
            dtype=torch.long
        )
        tgt_tensor = torch.tensor(
            [config.BOS_IDX] + vocab_tgt(tokenizer_tgt(tgt_text)) + [config.EOS_IDX],
            dtype=torch.long
        )
        processed_data.append({'src': src_tensor, 'tgt': tgt_tensor})
    return processed_data

def main():
    print("--- Starting Multi30k Preprocessing (from local git repository) ---")

    # 1. 检查数据目录是否存在 (作为git clone成功的标志)
    if not os.path.exists('dataset'):
         print("错误：找不到 'dataset' 文件夹。")
         print("请先在项目主目录下运行 'git clone https://github.com/multi30k/dataset.git'")
         return

    # 2. 加载分词器
    try:
        tokenizer_src = get_tokenizer('spacy', language=config.SPACY_DE)
        tokenizer_tgt = get_tokenizer('spacy', language=config.SPACY_EN)
        print("Spacy tokenizers loaded successfully.")
    except IOError:
        print(f"Spacy models '{config.SPACY_DE}' or '{config.SPACY_EN}' not found.")
        print("Please run: python -m spacy download de_core_news_sm")
        print("And: python -m spacy download en_core_web_sm")
        return

    # 3. 从本地文件读取数据
    train_src_lines = read_lines(DATA_PATH_TEMPLATE.format(split='train', lang=SRC_LANGUAGE))
    train_tgt_lines = read_lines(DATA_PATH_TEMPLATE.format(split='train', lang=TGT_LANGUAGE))
    val_src_lines = read_lines(DATA_PATH_TEMPLATE.format(split='val', lang=SRC_LANGUAGE))
    val_tgt_lines = read_lines(DATA_PATH_TEMPLATE.format(split='val', lang=TGT_LANGUAGE))
    print(f"Multi30k data loaded from local files. Train size: {len(train_src_lines)}, Valid size: {len(val_src_lines)}")

    # 4. 构建词汇表 (只使用训练集)
    print("Building vocabularies from training data...")
    vocab_src = build_vocabulary(tokenizer_src, train_src_lines)
    vocab_src.set_default_index(config.UNK_IDX)

    vocab_tgt = build_vocabulary(tokenizer_tgt, train_tgt_lines)
    vocab_tgt.set_default_index(config.UNK_IDX)
    print(f"Source vocab size: {len(vocab_src)}")
    print(f"Target vocab size: {len(vocab_tgt)}")

    # 5. 处理数据 (文本 -> Tensor) 并保存到我们自己的 data_multi30k 目录
    # 这样可以保持和其他数据处理流程的一致性
    output_dir = 'data_multi30k'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    print(f"Processing and saving data as Tensors to '{output_dir}'...")
    train_processed = process_data(train_src_lines, train_tgt_lines, vocab_src, vocab_tgt, tokenizer_src, tokenizer_tgt)
    torch.save(train_processed, os.path.join(output_dir, 'train.pt'))

    val_processed = process_data(val_src_lines, val_tgt_lines, vocab_src, vocab_tgt, tokenizer_src, tokenizer_tgt)
    torch.save(val_processed, os.path.join(output_dir, 'validation.pt'))

    print("Saving vocabularies...")
    torch.save(vocab_src, os.path.join(output_dir, 'vocab_src.pt'))
    torch.save(vocab_tgt, os.path.join(output_dir, 'vocab_tgt.pt'))

    print("\n--- Multi30k Preprocessing Finished Successfully! ---")
    print(f"All files are saved in the '{output_dir}' directory.")
    print("You can now run 'python train_multi30k.py'")

if __name__ == '__main__':
    main() 