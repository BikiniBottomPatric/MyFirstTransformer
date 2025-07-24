#!/bin/bash
# AutoDL服务器一键部署脚本
# 使用方法：bash autodl_deploy.sh

set -e  # 遇到错误立即退出

echo "🚀 开始AutoDL服务器部署..."

# 1. 检查是否在AutoDL环境
if [ ! -d "/root/autodl-tmp" ]; then
    echo "❌ 错误：当前不在AutoDL环境中"
    exit 1
fi

echo "✅ 检测到AutoDL环境"

# 2. 进入数据盘目录
cd /root/autodl-tmp

# 3. 克隆项目（如果不存在）
if [ ! -d "MyFirstTransformer" ]; then
    echo "📥 克隆项目代码..."
    git clone https://github.com/BikiniBottomPatric/MyFirstTransformer.git
else
    echo "📂 项目已存在，更新代码..."
    cd MyFirstTransformer
    git pull origin main
    cd ..
fi

cd MyFirstTransformer

# 4. 创建conda环境
echo "🐍 创建Python环境..."
if conda env list | grep -q "transformer"; then
    echo "环境已存在，激活环境..."
    source activate transformer
else
    echo "创建新环境..."
    conda create -n transformer python=3.9 -y
    source activate transformer
fi

# 5. 安装依赖
echo "📦 安装Python依赖..."
pip install -r requirements.txt

# 6. 创建必要目录
echo "📁 创建项目目录..."
mkdir -p /root/autodl-tmp/MyFirstTransformer/checkpoints
mkdir -p /root/autodl-tmp/MyFirstTransformer/logs
mkdir -p /root/autodl-tmp/MyFirstTransformer/data_bpe_original
mkdir -p /root/autodl-tmp/.cache/huggingface

# 7. 设置环境变量
echo "🔧 设置环境变量..."
export HF_HOME=/root/autodl-tmp/.cache/huggingface
export TRANSFORMERS_CACHE=/root/autodl-tmp/.cache/huggingface
export HF_DATASETS_CACHE=/root/autodl-tmp/.cache/huggingface

# 8. 验证配置
echo "🔍 验证配置..."
python -c "import config; config.validate_config()"

# 9. 清理系统盘空间
echo "🧹 清理系统盘空间..."
rm -rf /root/miniconda3/pkgs/* 2>/dev/null || true
rm -rf /root/.local/share/Trash/* 2>/dev/null || true

echo "✅ 部署完成！"
echo ""
echo "🎯 下一步操作："
echo "1. 启动训练："
echo "   screen -S training"
echo "   conda activate transformer"
echo "   python train.py > /root/autodl-tmp/MyFirstTransformer/logs/train.log 2>&1"
echo ""
echo "2. 监控训练："
echo "   tail -f /root/autodl-tmp/MyFirstTransformer/logs/train.log"
echo ""
echo "3. 恢复训练会话："
echo "   screen -r training"
echo ""
echo "📊 系统信息："
echo "- 项目路径: /root/autodl-tmp/MyFirstTransformer"
echo "- 日志路径: /root/autodl-tmp/MyFirstTransformer/logs"
echo "- 检查点: /root/autodl-tmp/MyFirstTransformer/checkpoints"
echo "- HF缓存: /root/autodl-tmp/.cache/huggingface"