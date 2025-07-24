# config.py
# 项目宪法：单一事实来源 (Single Source of Truth)
# 所有超参数和路径的统一配置文件

import torch
import os

# =============================================================================
# 第一章：核心原则配置
# =============================================================================

# --- 1. 硬件配置 (RTX 4060 8GB 适配) ---
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
MAX_GPU_MEMORY_GB = 8  # RTX 4060显存限制

# --- 2. WMT14数据集配置 (绝对忠于原文精神) ---
# 使用Hugging Face datasets库自动下载WMT14
DATASET_NAME = 'wmt14'
LANGUAGE_PAIR = 'de-en'
SRC_LANGUAGE = 'de'  # 德语
TGT_LANGUAGE = 'en'  # 英语

# 数据路径配置 - 使用Hugging Face缓存的WMT14数据
USE_HUGGINGFACE_DATASETS = True  # 使用HF datasets而非本地文件
RAW_DATA_DIR = "/home/patric/.cache/huggingface/datasets/wmt14/de-en/0.0.0/b199e406369ec1b7634206d3ded5ba45de2fe696"  # 使用Hugging Face缓存的WMT14数据
HUGGINGFACE_CACHE_DIR = "/home/patric/.cache/huggingface/datasets"  # Hugging Face缓存目录
PREPARED_DATA_DIR = "data_bpe_original"  # BPE处理后的数据
CHECKPOINTS_DIR = "checkpoints"
LOGS_DIR = "logs"

# 确保目录存在
for dir_path in [PREPARED_DATA_DIR, CHECKPOINTS_DIR, LOGS_DIR]:
    os.makedirs(dir_path, exist_ok=True)

# --- 3. BPE配置 (严格遵循论文) ---
BPE_MODEL_PREFIX = f"{PREPARED_DATA_DIR}/bpe_model"
BPE_VOCAB_SIZE = 37000  # 论文标准：共享词汇表，约37K
CHARACTER_COVERAGE = 0.9995  # SentencePiece推荐值
CHUNK_SIZE = 10000  # 数据分块大小（每个分块的样本数）

# 特殊Token配置 (与tokenizer.json一致)
UNK_IDX, PAD_IDX, BOS_IDX, EOS_IDX = 1, 4, 2, 3
SPECIAL_TOKENS = {
    '<unk>': UNK_IDX,
    '<pad>': PAD_IDX, 
    '<s>': BOS_IDX,     # 对应tokenizer.json中的<bos>
    '</s>': EOS_IDX     # 对应tokenizer.json中的<eos>
}

# 为了兼容性，保留原始名称的映射
BOS_TOKEN = '<s>'
EOS_TOKEN = '</s>'
UNK_TOKEN = '<unk>'
PAD_TOKEN = '<pad>'

# =============================================================================
# 第二章：模型架构配置 ("Base Model" from Paper)
# =============================================================================

# --- Transformer Base Model 配置 (严格遵循论文65M参数) ---
D_MODEL = 512              # 模型维度 (论文标准)
NUM_ENCODER_LAYERS = 6     # 编码器层数 (论文标准)
NUM_DECODER_LAYERS = 6     # 解码器层数 (论文标准)
NHEAD = 8                  # 多头注意力头数 (论文标准)
DIM_FEEDFORWARD = 2048     # 前馈网络维度 (论文标准)
D_FF = 2048                # 前馈网络维度 (别名，与DIM_FEEDFORWARD相同)
DROPOUT = 0.1              # Dropout率 (论文标准)
MAX_SEQ_LEN = 256         # 最大序列长度 (增加以匹配论文，适当平衡显存)

# --- 权重初始化配置 ---
XAVIER_UNIFORM_GAIN = 1.0
EMBEDDING_INIT_STD = D_MODEL ** -0.5  # 论文标准
SHARE_EMBEDDINGS = False  # 不共享嵌入权重以达到65M参数（原文标准）

# =============================================================================
# 第三章：训练配置 (8GB GPU适配 + 论文忠实)
# =============================================================================

# --- 批次配置 (动态批处理 + 梯度累积) ---
BATCH_SIZE_TOKENS = 4096   # 动态批处理：每批次目标token数 (适配8GB显存)
MAX_BATCH_SIZE = 32        # 固定批次大小的上限 (主要用于测试集)
GRADIENT_ACCUMULATION_STEPS = 6   # 梯度累积步数 (动态批处理下可以减少)
EFFECTIVE_BATCH_SIZE = BATCH_SIZE_TOKENS * GRADIENT_ACCUMULATION_STEPS  # 24576 tokens (论文推荐)

# 动态批处理配置
USE_DYNAMIC_BATCHING = True  # 启用动态批处理
DYNAMIC_BATCH_MAX_TOKENS = BATCH_SIZE_TOKENS  # 动态批处理的最大token数

# --- 学习率调度 (论文标准配置) ---
LEARNING_RATE_BASE = 1.0   # 基础学习率 (将被调度器控制)
LEARNING_RATE_SCALE = 0.7  # 学习率缩放因子
WARMUP_STEPS = 8000        # 增加预热步数以更好收敛
MAX_LEARNING_RATE = 1e-4   # 降低学习率以稳定训练
MIN_LEARNING_RATE = 1e-6   # 最小学习率
BETA1 = 0.9                # Adam beta1
BETA2 = 0.98               # Adam beta2 (论文值)
EPS = 1e-9                 # Adam epsilon

# --- 正则化配置 ---
LABEL_SMOOTHING_EPS = 0.1   # 论文标准标签平滑系数
GRADIENT_CLIP_NORM = 1.0    # 梯度裁剪阈值

# --- 训练步数配置 ---
TRAIN_STEPS = 1800000      # 180万物理步（15万次逻辑更新）
VALIDATE_EVERY_STEPS = 500
VALIDATE_EVERY_LOGICAL_STEPS = 1000  # 每1000次逻辑更新验证一次
LOG_EVERY_LOGICAL_STEPS = 500        # 每500次逻辑更新输出loss
BLEU_EVAL_EVERY_LOGICAL_STEPS = 1000 # 每1000次逻辑更新输出BLEU
SAVE_EVERY_LOGICAL_STEPS = 5000      # 检查点保存频率（逻辑步）

# --- 早停配置 ---
EARLY_STOPPING_PATIENCE = 15  # 连续15次验证无提升则停止（更宽松）
MIN_DELTA_BLEU = 0.05      # BLEU提升的最小阈值（更宽松）
SKIP_BLEU_BEFORE_STEPS = 16000  # 前2*WARMUP_STEPS步跳过BLEU评估

# =============================================================================
# 第四章：推理评估配置 (Beam Search + BLEU)
# =============================================================================

# --- Beam Search配置 ---
BEAM_SIZE = 5              # 增加束搜索大小以提高BLEU
LENGTH_PENALTY = 0.6       # 长度惩罚
MAX_DECODE_LENGTH = 256    # 最大解码长度
EARLY_STOPPING = True      # 早停策略
REPETITION_PENALTY = 1.0   # 重复惩罚
NO_REPEAT_NGRAM_SIZE = 0   # 禁止重复的n-gram大小

# --- BLEU评估配置 ---
BLEU_TOKENIZE = '13a'      # sacrebleu标准分词
BLEU_LOWERCASE = True      # 小写化
MAX_EVAL_SAMPLES = 3000    # 验证时最大样本数 (速度优化)

# =============================================================================
# 第五章：工程配置 (透明度 + 专业性)
# =============================================================================

# --- 进度条配置 ---
USE_TQDM = True            # 启用进度条
TQDM_DISABLE = False       # 不禁用tqdm
TQDM_NCOLS = 100          # 进度条宽度

# --- 日志配置 ---
LOG_LEVEL = 'INFO'         # 日志级别
LOG_FORMAT = '%(asctime)s - %(levelname)s - %(message)s'
TENSORBOARD_LOG_DIR = f"{LOGS_DIR}/tensorboard"

# --- 检查点配置 ---
CHECKPOINT_PREFIX = "transformer_wmt14"
BEST_MODEL_NAME = "best_model.pt"
LATEST_MODEL_NAME = "latest_model.pt"
CHECKPOINT_KEEP_LAST_N = 3  # 保留最近N个检查点

# --- 可复现性配置 ---
RANDOM_SEED = 42
CUDNN_DETERMINISTIC = True
CUDNN_BENCHMARK = False

# --- 内存优化配置 ---
EMPTY_CACHE_EVERY_N_STEPS = 1000  # 每N步清空CUDA缓存
PIN_MEMORY = True          # 数据加载器pin_memory
NUM_WORKERS = 0            # WSL2优化：使用0个worker

# =============================================================================
# 第六章：路径配置 (文件系统组织)
# =============================================================================

# --- 模型文件路径 ---
BEST_MODEL_PATH = f"{CHECKPOINTS_DIR}/{BEST_MODEL_NAME}"
LATEST_MODEL_PATH = f"{CHECKPOINTS_DIR}/{LATEST_MODEL_NAME}"
BPE_MODEL_PATH = f"{PREPARED_DATA_DIR}/tokenizer.json"  # 使用Hugging Face格式
BPE_VOCAB_PATH = f"{PREPARED_DATA_DIR}/vocab.json"

# SentencePiece模型路径 (用于BLEU计算)
SRC_BPE_MODEL_PATH = f"{PREPARED_DATA_DIR}/bpe_model.model"  # SentencePiece模型
TGT_BPE_MODEL_PATH = f"{PREPARED_DATA_DIR}/bpe_model.model"  # 共享词汇表，使用同一个模型

# --- 数据文件路径 ---
TRAIN_DATA_PATH = f"{PREPARED_DATA_DIR}/train_chunks"
VALID_DATA_PATH = f"{PREPARED_DATA_DIR}/validation_chunks"
TEST_DATA_PATH = f"{PREPARED_DATA_DIR}/test_chunks"
VOCAB_SRC_PATH = f"{PREPARED_DATA_DIR}/vocab_src.pt"
VOCAB_TGT_PATH = f"{PREPARED_DATA_DIR}/vocab_tgt.pt"
METADATA_PATH = f"{PREPARED_DATA_DIR}/metadata.json"

# =============================================================================
# 配置验证函数
# =============================================================================

def validate_config():
    """验证配置的合理性"""
    assert D_MODEL % NHEAD == 0, f"D_MODEL ({D_MODEL}) 必须能被 NHEAD ({NHEAD}) 整除"
    assert BPE_VOCAB_SIZE > 1000, f"BPE词汇表大小 ({BPE_VOCAB_SIZE}) 过小"
    assert WARMUP_STEPS > 0, f"预热步数 ({WARMUP_STEPS}) 必须大于0"
    assert EFFECTIVE_BATCH_SIZE >= 8192, f"有效批次大小 ({EFFECTIVE_BATCH_SIZE}) 建议不小于8192"
    
    print("✅ 配置验证通过")
    print(f"📊 模型参数估计: {(D_MODEL * D_MODEL * 6 * 4 + BPE_VOCAB_SIZE * D_MODEL * 2) / 1e6:.1f}M")
    print(f"💾 有效批次大小: {EFFECTIVE_BATCH_SIZE:,} tokens")
    print(f"🎯 目标设备: {DEVICE}")

if __name__ == "__main__":
    validate_config()