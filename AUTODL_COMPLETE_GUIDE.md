# 🚀 AutoDL完整操作指南

## 📋 目录
1. [连接AutoDL服务器](#1-连接autodl服务器)
2. [Screen后台运行详解](#2-screen后台运行详解)
3. [环境配置流程](#3-环境配置流程)
4. [IDE配置说明](#4-ide配置说明)
5. [完整部署流程](#5-完整部署流程)
6. [常见问题解答](#6-常见问题解答)

---

## 1. 连接AutoDL服务器

### 🔌 获取连接信息
1. 登录AutoDL控制台：https://www.autodl.com/
2. 找到你租用的实例
3. 点击「JupyterLab」或查看「连接信息」
4. 记录SSH连接命令，格式如：
   ```
   ssh root@connect.bjb1.seetacloud.com -p 12345
   ```

### 💻 连接方式选择

#### 方式一：JupyterLab（推荐新手）
1. 点击AutoDL控制台的「JupyterLab」按钮
2. 在浏览器中打开JupyterLab界面
3. 点击左侧「Terminal」图标打开终端
4. 直接在终端中操作（无需SSH）

#### 方式二：SSH连接
```bash
# Windows用户（使用PowerShell或CMD）
ssh root@connect.bjb1.seetacloud.com -p [你的端口号]

# 输入密码（通常是你设置的密码或默认密码）
```

---

## 2. Screen后台运行详解

### 🛡️ 为什么需要Screen？
- SSH连接断开时，普通命令会停止运行
- Screen创建虚拟终端，即使SSH断开也继续运行
- 训练可能需要几小时甚至几天，必须使用后台运行

### 📖 Screen基础操作

#### 安装Screen（如果没有）
```bash
apt-get update && apt-get install -y screen
```

#### 创建新会话
```bash
# 创建名为"training"的会话
screen -S training

# 会进入一个新的终端界面
# 在这个界面中运行的命令会受到保护
```

#### 离开会话（重要！）
```bash
# 按键组合：Ctrl + A，然后按 D
# 看到 [detached] 表示成功离开
# 此时训练继续在后台运行
```

#### 恢复会话
```bash
# 查看所有会话
screen -ls

# 恢复到training会话
screen -r training

# 如果只有一个会话，直接用
screen -r
```

#### 终止会话
```bash
# 在screen会话内按 Ctrl + D
# 或者输入 exit
```

---

## 3. 环境配置流程

### 🐍 Conda环境管理

#### 检查现有环境
```bash
# 查看所有conda环境
conda env list

# 查看当前环境
conda info --envs
```

#### 创建新环境（推荐）
```bash
# 创建名为transformer的Python 3.9环境
conda create -n transformer python=3.9 -y

# 激活环境
conda activate transformer

# 验证环境
which python
python --version
```

#### 使用现有环境
```bash
# 如果已有合适的环境，直接激活
conda activate [环境名]
```

### 📦 依赖安装
```bash
# 确保在正确的conda环境中
conda activate transformer

# 安装项目依赖
pip install -r requirements.txt

# 验证关键库
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
```

---

## 4. IDE配置说明

### 🔧 右下角Interpreter设置

#### 如果使用本地IDE连接AutoDL：
1. **不需要更改本地IDE的interpreter**
2. 本地IDE的interpreter仍然指向本地环境
3. 代码在AutoDL服务器上运行，使用服务器的环境

#### 如果使用JupyterLab：
1. 在JupyterLab中，kernel会自动使用当前激活的conda环境
2. 可以在Notebook中切换kernel到transformer环境

#### 如果使用VSCode Remote SSH：
1. 安装Remote SSH插件
2. 连接到AutoDL服务器
3. 在服务器上选择正确的Python解释器：
   ```
   /root/miniconda3/envs/transformer/bin/python
   ```

---

## 5. 完整部署流程

### 🎯 Step-by-Step操作

```bash
# === 第1步：连接服务器 ===
# 使用JupyterLab或SSH连接到AutoDL

# === 第2步：进入数据盘 ===
cd /root/autodl-tmp
pwd  # 确认在数据盘目录

# === 第3步：克隆项目 ===
git clone https://github.com/BikiniBottomPatric/MyFirstTransformer.git
cd MyFirstTransformer
ls -la  # 查看项目文件

# === 第4步：运行部署脚本 ===
bash autodl_deploy.sh
# 脚本会自动：
# - 创建conda环境
# - 安装依赖
# - 创建必要目录
# - 设置环境变量
# - 验证配置

# === 第5步：创建Screen会话 ===
screen -S training

# === 第6步：激活环境并开始训练 ===
conda activate transformer
python train.py > logs/train.log 2>&1

# === 第7步：离开Screen会话 ===
# 按 Ctrl+A，然后按 D

# === 第8步：监控训练（可选） ===
# 查看日志
tail -f /root/autodl-tmp/MyFirstTransformer/logs/train.log

# 查看GPU使用情况
nvidia-smi

# 恢复训练会话
screen -r training
```

---

## 6. 常见问题解答

### ❓ Q1: SSH连接总是断开怎么办？
**A:** 这是正常现象，使用Screen或JupyterLab终端可以解决：
```bash
# 使用Screen
screen -S training
# 在screen中运行训练
# Ctrl+A+D离开，训练继续
```

### ❓ Q2: 需要重新创建conda环境吗？
**A:** 建议创建新环境，避免冲突：
```bash
conda create -n transformer python=3.9 -y
conda activate transformer
```

### ❓ Q3: 本地IDE的interpreter要改吗？
**A:** 不需要！本地IDE保持不变，代码在服务器运行。

### ❓ Q4: 如何确认训练在运行？
**A:** 多种方法检查：
```bash
# 查看进程
ps aux | grep python

# 查看GPU使用
nvidia-smi

# 查看日志
tail -f logs/train.log

# 恢复screen会话
screen -r training
```

### ❓ Q5: 系统盘空间不足怎么办？
**A:** 清理系统盘：
```bash
# 清理conda缓存
rm -rf /root/miniconda3/pkgs/*

# 清理回收站
rm -rf /root/.local/share/Trash/*

# 检查空间使用
df -h
du -sh /root/autodl-tmp/*
```

### ❓ Q6: 如何停止训练？
**A:** 
```bash
# 恢复screen会话
screen -r training

# 按 Ctrl+C 停止训练
# 按 Ctrl+D 退出screen会话
```

---

## 🎉 快速启动命令

```bash
# 一键启动（复制粘贴即可）
cd /root/autodl-tmp && \
git clone https://github.com/BikiniBottomPatric/MyFirstTransformer.git && \
cd MyFirstTransformer && \
bash autodl_deploy.sh && \
screen -S training && \
conda activate transformer && \
python train.py > logs/train.log 2>&1
```

记住：**Ctrl+A+D** 离开screen，**screen -r training** 恢复会话！