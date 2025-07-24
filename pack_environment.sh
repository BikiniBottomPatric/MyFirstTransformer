#!/bin/bash
# 本地环境打包脚本
# 将本地开发环境打包，便于在AutoDL服务器上快速恢复

set -e

echo "📦 开始打包本地环境..."

# 1. 创建打包目录
PACK_DIR="transformer_package_$(date +%Y%m%d_%H%M%S)"
mkdir -p $PACK_DIR

echo "📁 创建打包目录: $PACK_DIR"

# 2. 导出conda环境
echo "🐍 导出conda环境配置..."
if conda env list | grep -q "transformer"; then
    conda activate transformer
    conda env export > $PACK_DIR/environment.yml
    pip freeze > $PACK_DIR/requirements_frozen.txt
    echo "✅ 环境配置已导出"
else
    echo "⚠️  未找到transformer环境，使用当前环境"
    pip freeze > $PACK_DIR/requirements_current.txt
fi

# 3. 复制项目文件（排除大文件和缓存）
echo "📂 复制项目文件..."
cp -r . $PACK_DIR/project/

# 4. 清理不必要的文件
echo "🧹 清理临时文件..."
cd $PACK_DIR/project

# 删除缓存和临时文件
rm -rf __pycache__/
rm -rf .git/
rm -rf checkpoints/
rm -rf logs/
rm -rf data_bpe_original/
rm -rf dataset/
rm -rf *.pyc
rm -rf .pytest_cache/
rm -rf .vscode/
rm -rf .idea/

cd ../..

# 5. 创建部署说明
cat > $PACK_DIR/DEPLOY_README.md << 'EOF'
# AutoDL部署指南

## 📦 包内容
- `project/`: 项目源代码
- `environment.yml`: conda环境配置
- `requirements_frozen.txt`: pip依赖列表
- `autodl_deploy.sh`: 一键部署脚本

## 🚀 AutoDL服务器部署步骤

### 方法一：使用Git（推荐）
```bash
# 1. 连接到AutoDL服务器
ssh root@connect.bjb1.seetacloud.com -p [你的端口]

# 2. 进入数据盘
cd /root/autodl-tmp

# 3. 克隆项目
git clone https://github.com/BikiniBottomPatric/MyFirstTransformer.git
cd MyFirstTransformer

# 4. 运行部署脚本
bash autodl_deploy.sh
```

### 方法二：上传打包文件
```bash
# 1. 上传打包文件到AutoDL
scp transformer_package_*.tar.gz root@connect.bjb1.seetacloud.com:/root/autodl-tmp/

# 2. 在AutoDL服务器上解压
cd /root/autodl-tmp
tar -xzf transformer_package_*.tar.gz
cd transformer_package_*/project

# 3. 创建环境
conda env create -f ../environment.yml
conda activate transformer

# 4. 运行训练
screen -S training
python train.py > logs/train.log 2>&1
```

## 🔧 环境变量设置
```bash
export HF_HOME=/root/autodl-tmp/.cache/huggingface
export TRANSFORMERS_CACHE=/root/autodl-tmp/.cache/huggingface
export HF_DATASETS_CACHE=/root/autodl-tmp/.cache/huggingface
```

## 📊 监控训练
```bash
# 查看训练日志
tail -f /root/autodl-tmp/MyFirstTransformer/logs/train.log

# 恢复screen会话
screen -r training

# 查看GPU使用情况
nvidia-smi

# 查看磁盘使用情况
df -h
du -sh /root/autodl-tmp/*
```

## ⚠️ 注意事项
1. 确保使用数据盘(/root/autodl-tmp)存储大文件
2. 定期清理系统盘空间
3. 使用screen或tmux运行长时间任务
4. 定期保存检查点
EOF

# 6. 创建压缩包
echo "🗜️  创建压缩包..."
tar -czf ${PACK_DIR}.tar.gz $PACK_DIR/

echo "✅ 打包完成！"
echo ""
echo "📦 打包文件: ${PACK_DIR}.tar.gz"
echo "📁 打包目录: $PACK_DIR/"
echo ""
echo "🚀 下一步："
echo "1. 将 ${PACK_DIR}.tar.gz 上传到AutoDL服务器"
echo "2. 或者直接使用Git克隆项目（推荐）"
echo "3. 在AutoDL上运行: bash autodl_deploy.sh"

# 7. 显示文件大小
echo ""
echo "📊 文件信息："
ls -lh ${PACK_DIR}.tar.gz
echo "📂 目录内容："
ls -la $PACK_DIR/