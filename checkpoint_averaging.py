#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
模型平均 (Checkpoint Averaging) 脚本

这个脚本将最后N个检查点的权重进行平均，生成一个更鲁棒的模型。
模型平均是一个几乎零成本但通常能带来0.5-1.5个BLEU点提升的技巧。
"""

import torch
import os
import glob
import argparse
from collections import OrderedDict
import logging
from pathlib import Path

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_checkpoint(checkpoint_path):
    """加载检查点"""
    try:
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        if 'model_state_dict' in checkpoint:
            return checkpoint['model_state_dict']
        else:
            # 如果直接是state_dict
            return checkpoint
    except Exception as e:
        logger.error(f"加载检查点失败 {checkpoint_path}: {e}")
        return None

def get_latest_checkpoints(checkpoint_dir, num_checkpoints=5, pattern="*.pt"):
    """获取最新的N个检查点文件"""
    checkpoint_files = glob.glob(os.path.join(checkpoint_dir, pattern))
    
    # 按修改时间排序，最新的在前
    checkpoint_files.sort(key=lambda x: os.path.getmtime(x), reverse=True)
    
    # 过滤掉平均模型文件（避免重复平均）
    checkpoint_files = [f for f in checkpoint_files if 'averaged' not in f and 'average' not in f]
    
    logger.info(f"找到 {len(checkpoint_files)} 个检查点文件")
    
    # 返回最新的N个
    selected_files = checkpoint_files[:num_checkpoints]
    logger.info(f"选择最新的 {len(selected_files)} 个检查点进行平均:")
    for i, f in enumerate(selected_files):
        logger.info(f"  {i+1}. {os.path.basename(f)} (修改时间: {os.path.getmtime(f)})")
    
    return selected_files

def average_checkpoints(checkpoint_paths, output_path):
    """对多个检查点进行权重平均"""
    logger.info(f"开始平均 {len(checkpoint_paths)} 个检查点...")
    
    # 加载第一个检查点作为基础
    first_checkpoint = load_checkpoint(checkpoint_paths[0])
    if first_checkpoint is None:
        raise ValueError(f"无法加载第一个检查点: {checkpoint_paths[0]}")
    
    # 初始化平均权重字典
    averaged_state_dict = OrderedDict()
    for key, value in first_checkpoint.items():
        averaged_state_dict[key] = value.clone().float()  # 转换为float32进行精确计算
    
    logger.info(f"✅ 加载基础检查点: {os.path.basename(checkpoint_paths[0])}")
    
    # 累加其他检查点的权重
    valid_checkpoints = 1
    for i, checkpoint_path in enumerate(checkpoint_paths[1:], 1):
        checkpoint = load_checkpoint(checkpoint_path)
        if checkpoint is None:
            logger.warning(f"跳过无效检查点: {checkpoint_path}")
            continue
        
        # 检查键是否匹配
        if set(checkpoint.keys()) != set(averaged_state_dict.keys()):
            logger.warning(f"检查点 {checkpoint_path} 的键不匹配，跳过")
            continue
        
        # 累加权重
        for key in averaged_state_dict.keys():
            averaged_state_dict[key] += checkpoint[key].float()
        
        valid_checkpoints += 1
        logger.info(f"✅ 累加检查点 {i+1}: {os.path.basename(checkpoint_path)}")
    
    # 计算平均值
    logger.info(f"计算平均权重 (有效检查点数: {valid_checkpoints})...")
    for key in averaged_state_dict.keys():
        averaged_state_dict[key] /= valid_checkpoints
    
    # 保存平均模型
    logger.info(f"保存平均模型到: {output_path}")
    
    # 创建完整的检查点格式
    averaged_checkpoint = {
        'model_state_dict': averaged_state_dict,
        'averaged_from': [os.path.basename(p) for p in checkpoint_paths[:valid_checkpoints]],
        'num_averaged': valid_checkpoints,
        'averaging_info': {
            'method': 'simple_average',
            'precision': 'float32',
            'timestamp': str(torch.get_default_dtype())
        }
    }
    
    torch.save(averaged_checkpoint, output_path)
    logger.info(f"🎉 模型平均完成！平均了 {valid_checkpoints} 个检查点")
    
    return valid_checkpoints

def main():
    parser = argparse.ArgumentParser(description='模型检查点平均脚本')
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints',
                       help='检查点目录路径 (默认: ./checkpoints)')
    parser.add_argument('--num_checkpoints', type=int, default=5,
                       help='要平均的检查点数量 (默认: 5)')
    parser.add_argument('--output_name', type=str, default='averaged_model.pt',
                       help='输出的平均模型文件名 (默认: averaged_model.pt)')
    parser.add_argument('--pattern', type=str, default='*.pt',
                       help='检查点文件匹配模式 (默认: *.pt)')
    
    args = parser.parse_args()
    
    # 检查检查点目录
    if not os.path.exists(args.checkpoint_dir):
        logger.error(f"检查点目录不存在: {args.checkpoint_dir}")
        return
    
    # 获取最新的检查点
    checkpoint_paths = get_latest_checkpoints(
        args.checkpoint_dir, 
        args.num_checkpoints, 
        args.pattern
    )
    
    if len(checkpoint_paths) < 2:
        logger.error(f"找到的检查点数量不足 ({len(checkpoint_paths)})，至少需要2个")
        return
    
    # 设置输出路径
    output_path = os.path.join(args.checkpoint_dir, args.output_name)
    
    try:
        # 执行模型平均
        num_averaged = average_checkpoints(checkpoint_paths, output_path)
        
        logger.info("\n" + "="*50)
        logger.info("🎯 模型平均总结:")
        logger.info(f"📁 检查点目录: {args.checkpoint_dir}")
        logger.info(f"📊 平均检查点数: {num_averaged}")
        logger.info(f"💾 输出文件: {output_path}")
        logger.info(f"📈 预期BLEU提升: 0.5 - 1.5 点")
        logger.info("="*50)
        
        # 提供使用建议
        logger.info("\n💡 使用建议:")
        logger.info(f"1. 使用平均模型进行测试: python train.py --test --checkpoint {output_path}")
        logger.info("2. 比较平均模型与最佳单一模型的BLEU分数")
        logger.info("3. 如果平均模型效果更好，可以用它作为最终模型")
        
    except Exception as e:
        logger.error(f"模型平均失败: {e}")
        raise

if __name__ == "__main__":
    main()