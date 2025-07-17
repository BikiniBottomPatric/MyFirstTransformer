import torch

# --- 数据集与分词器配置 ---
SRC_LANGUAGE = 'de'
TGT_LANGUAGE = 'en'

# Spacy模型名称
SPACY_DE = 'de_core_news_sm'
SPACY_EN = 'en_core_web_sm'

# --- 模型架构超参数 (严格参考论文的Base Model) ---
D_MODEL = 512           # 嵌入维度和模型内部维度
NUM_ENCODER_LAYERS = 6  # 编码器层数
NUM_DECODER_LAYERS = 6  # 解码器层数
NHEAD = 8               # 多头注意力的头数
DIM_FEEDFORWARD = 2048  # 前馈神经网络的隐藏层维度
DROPOUT = 0.1           # Dropout比例

# --- 训练超参数 ---
BATCH_SIZE = 128        # 批处理大小
NUM_EPOCHS = 10         # 训练轮数
LEARNING_RATE = 0.0001  # 固定的学习率 (后续可改为动态)
WARMUP_STEPS = 4000     # 学习率预热步数
LABEL_SMOOTHING_EPS = 0.1 # 标签平滑的 epsilon

# DataLoader 相关
NUM_WORKERS = 4  # 根据 CPU 核心数量自行调整 (0 表示主进程加载)

# --- 词汇表与特殊符号 ---
UNK_IDX, PAD_IDX, BOS_IDX, EOS_IDX = 0, 1, 2, 3
special_symbols = ['<unk>', '<pad>', '<bos>', '<eos>']

# --- 设备配置 ---
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# --- WMT14 完整训练配置 ---
WMT_TRAIN_STEPS = 100000              # 论文中建议的总训练步数
WMT_VALIDATE_EVERY_N_STEPS = 5000   # 每隔多少步进行一次验证

# --- Multi30k 快速实验配置 ---
TRAIN_STEPS = 10000              # 为快速实验设置的总训练步数
VALIDATE_EVERY_N_STEPS = 500   # 每隔多少步进行一次验证