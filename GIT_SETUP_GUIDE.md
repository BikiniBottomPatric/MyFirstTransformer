# 🚀 Git & GitHub 配置指南

## 📋 当前状态检查

你的Git已经基本配置完成：
- ✅ 用户名: `BikiniBottomPatric`
- ✅ 邮箱: `2453780@tongji.edu.cn`
- ✅ 本地仓库已初始化
- ❌ 尚未连接到GitHub远程仓库

---

## 🔧 完整配置步骤

### 1. 生成SSH密钥（推荐）

```bash
# 生成SSH密钥
ssh-keygen -t ed25519 -C "2453780@tongji.edu.cn"

# 启动ssh-agent
eval "$(ssh-agent -s)"

# 添加SSH密钥到ssh-agent
ssh-add ~/.ssh/id_ed25519

# 复制公钥到剪贴板
cat ~/.ssh/id_ed25519.pub
```

### 2. 在GitHub上添加SSH密钥

1. 登录 [GitHub](https://github.com)
2. 点击右上角头像 → Settings
3. 左侧菜单选择 "SSH and GPG keys"
4. 点击 "New SSH key"
5. 粘贴刚才复制的公钥内容
6. 给密钥起个名字（如："RTX4060-Workstation"）
7. 点击 "Add SSH key"

### 3. 测试SSH连接

```bash
# 测试GitHub连接
ssh -T git@github.com
```

应该看到类似输出：
```
Hi BikiniBottomPatric! You've successfully authenticated, but GitHub does not provide shell access.
```

### 4. 创建GitHub仓库

1. 在GitHub上点击 "New repository"
2. 仓库名建议：`transformer-wmt14-reproduction`
3. 描述：`Transformer model reproduction for WMT14 EN-DE translation (BLEU ≥ 25.0 target)`
4. 选择 Public（如果你想开源）或 Private
5. **不要**勾选 "Initialize with README"（因为本地已有文件）
6. 点击 "Create repository"

### 5. 连接本地仓库到GitHub

```bash
# 添加远程仓库（替换为你的GitHub用户名）
git remote add origin git@github.com:BikiniBottomPatric/transformer-wmt14-reproduction.git

# 验证远程仓库
git remote -v
```

---

## 📦 首次提交和推送

### 1. 整理文件

```bash
# 添加重要文件
git add README.md
git add config.py
git add model.py
git add train.py
git add data_utils.py
git add preprocess.py
git add beam_search.py
git add checkpoint_averaging.py
git add requirements.txt
git add EXPERIMENT_LOG.md
git add GIT_SETUP_GUIDE.md

# 添加文档文件
git add ADVANCED_TRAINING_FEATURES.md
git add FIXES_SUMMARY.md
git add TRAINING_OPTIMIZATIONS.md

# 删除不需要的文件
git rm data_multi30k/train.pt
git rm data_multi30k/validation.pt
git rm data_multi30k/vocab_src.pt
git rm data_multi30k/vocab_tgt.pt
git rm data_utils_multi30k.py
git rm debug_train.py
git rm preprocess_multi30k.py
git rm train_multi30k.py
git rm transformer_env_list.txt
```

### 2. 创建 .gitignore

```bash
# 查看当前.gitignore内容
cat .gitignore
```

### 3. 提交更改

```bash
# 提交当前状态
git commit -m "🎯 Initial commit: Transformer WMT14 reproduction project

✨ Features:
- Complete Transformer implementation following 'Attention is All You Need'
- WMT14 EN-DE translation with BPE preprocessing
- Target: BLEU ≥ 25.0
- RTX 4060 8GB optimized
- Comprehensive training pipeline with:
  - Dynamic batching
  - Mixed precision training
  - Beam search decoding
  - Checkpoint averaging
  - Early stopping
  - TensorBoard monitoring

📊 Project Status:
- Model: Transformer Base (~65M parameters)
- Data: WMT14 with BPE tokenization
- Hardware: RTX 4060 8GB optimized
- Monitoring: TensorBoard + detailed logging"
```

### 4. 推送到GitHub

```bash
# 首次推送（设置上游分支）
git push -u origin master
```

---

## 🌿 分支管理策略

### 主要分支
- `master/main`: 稳定版本，经过测试的代码
- `develop`: 开发分支，集成新功能
- `experiment/*`: 实验分支，用于测试新想法

### 创建实验分支

```bash
# 创建并切换到实验分支
git checkout -b experiment/label-smoothing

# 进行实验...
# 修改代码，测试结果

# 提交实验结果
git add .
git commit -m "🧪 Experiment: Label smoothing (ε=0.1)

Results:
- Validation BLEU: XX.XX
- Test BLEU: XX.XX
- Training time: X hours

Observations:
- Improved convergence stability
- Slight BLEU improvement (+0.5)"

# 推送实验分支
git push -u origin experiment/label-smoothing
```

### 合并成功的实验

```bash
# 切换回主分支
git checkout master

# 合并实验分支
git merge experiment/label-smoothing

# 推送更新
git push origin master

# 删除已合并的实验分支（可选）
git branch -d experiment/label-smoothing
git push origin --delete experiment/label-smoothing
```

---

## 📊 实验记录最佳实践

### 1. 提交信息规范

```bash
# 功能提交
git commit -m "✨ Add beam search decoder with length penalty"

# 实验提交
git commit -m "🧪 Experiment #003: Increased learning rate

Config changes:
- Learning rate: 1e-4 → 2e-4
- Warmup steps: 4000

Results:
- Best validation BLEU: 23.45
- Test BLEU: 22.89
- Training time: 4.2 hours

Observations:
- Faster convergence but less stable
- Slight overfitting observed

Next steps:
- Try learning rate scheduling
- Add more regularization"

# 修复提交
git commit -m "🐛 Fix memory leak in data loader"

# 文档提交
git commit -m "📝 Update experiment log with latest results"
```

### 2. 标签管理

```bash
# 为重要里程碑创建标签
git tag -a v1.0-baseline -m "Baseline model: BLEU 20.5"
git tag -a v1.1-optimized -m "Optimized model: BLEU 24.2"
git tag -a v2.0-target -m "Target achieved: BLEU 25.1"

# 推送标签
git push origin --tags
```

### 3. 实验数据管理

```bash
# 不要提交大文件到Git
echo "checkpoints/*.pt" >> .gitignore
echo "logs/" >> .gitignore
echo "data_bpe_original/" >> .gitignore
echo "processed_data/" >> .gitignore

# 但要记录实验配置和结果
git add EXPERIMENT_LOG.md
git commit -m "📊 Update experiment log: BLEU 24.8 achieved"
```

---

## 🔄 日常工作流程

### 开始新实验

```bash
# 1. 确保主分支是最新的
git checkout master
git pull origin master

# 2. 创建实验分支
git checkout -b experiment/new-feature

# 3. 记录实验计划
echo "### 实验 #XXX - 新功能测试" >> EXPERIMENT_LOG.md
git add EXPERIMENT_LOG.md
git commit -m "📋 Plan experiment #XXX: New feature test"
```

### 实验过程中

```bash
# 定期提交进度
git add .
git commit -m "🚧 WIP: Implementing new feature - checkpoint 1"

# 推送到远程（备份）
git push origin experiment/new-feature
```

### 实验完成

```bash
# 更新实验记录
vim EXPERIMENT_LOG.md  # 记录结果

# 提交最终结果
git add .
git commit -m "🧪 Complete experiment #XXX: New feature

Results:
- Validation BLEU: XX.XX
- Test BLEU: XX.XX
- Improvement: +X.X BLEU

Conclusion: [Success/Failure] - [Reason]"

# 推送结果
git push origin experiment/new-feature
```

---

## 🛠️ 有用的Git命令

```bash
# 查看提交历史
git log --oneline --graph

# 查看文件变更
git diff
git diff --staged

# 撤销更改
git restore <file>          # 撤销工作区更改
git restore --staged <file> # 撤销暂存区更改
git reset HEAD~1            # 撤销最后一次提交

# 查看分支
git branch -a               # 查看所有分支
git branch -r               # 查看远程分支

# 同步远程分支
git fetch origin
git pull origin master

# 清理
git clean -fd               # 删除未跟踪文件
git gc                      # 垃圾回收
```

---

## 🎯 下一步行动

1. **立即执行**:
   - [ ] 生成SSH密钥
   - [ ] 在GitHub添加SSH密钥
   - [ ] 创建GitHub仓库
   - [ ] 连接本地仓库到GitHub

2. **整理项目**:
   - [ ] 更新.gitignore
   - [ ] 首次提交和推送
   - [ ] 创建项目标签

3. **建立工作流**:
   - [ ] 开始使用实验分支
   - [ ] 定期更新实验记录
   - [ ] 为重要里程碑创建标签

---

## 💡 专业建议

1. **提交频率**: 每个小功能或实验阶段都要提交
2. **分支策略**: 用分支隔离实验，避免影响主线
3. **文档先行**: 先记录实验计划，再开始编码
4. **结果追踪**: 每次实验都要记录详细结果
5. **备份重要**: 定期推送到GitHub，避免数据丢失

记住：**好的版本控制习惯是成功研究的基础！** 🚀