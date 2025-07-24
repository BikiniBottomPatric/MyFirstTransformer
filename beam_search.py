# beam_search.py (重构版)

import torch
import torch.nn.functional as F
from dataclasses import dataclass
from typing import List, Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)

@dataclass
class BeamHypothesis:
    """Beam Search假设"""
    tokens: List[int]
    score: float
    
    def normalized_score(self, length_penalty: float = 1.0) -> float:
        """计算长度归一化分数"""
        length = len(self.tokens)
        if length == 0:
            return self.score
        return self.score / (length ** length_penalty)

class BeamSearchDecoder:
    """Beam Search解码器（重构版）"""
    
    def __init__(self, 
                 model,
                 tokenizer=None,
                 beam_size: int = 4,
                 max_length: int = 100,
                 length_penalty: float = 1.0,
                 early_stopping: bool = True,
                 bos_token_id: int = 1,
                 eos_token_id: int = 2,
                 unk_token_id: int = 0):
        self.model = model
        self.tokenizer = tokenizer
        self.beam_size = beam_size
        self.max_length = max_length
        self.length_penalty = length_penalty
        self.early_stopping = early_stopping
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        self.unk_token_id = unk_token_id

    @torch.no_grad()
    def search(self, 
               src: torch.Tensor, 
               src_mask: Optional[torch.Tensor] = None, 
               verbose: bool = True) -> Dict[str, Any]:
        
        device = src.device
        batch_size = src.size(0)
        assert batch_size == 1, "Beam Search目前只支持batch_size=1"

        vocab_size = self.model.decoder.fc_out.out_features
        # 编码
        memory = self.model.encode(src, src_mask)  # [1, src_len, d_model]
        
        # 初始化
        # beam_scores: [beam_size], 记录每个beam的累积log-prob
        beam_scores = torch.zeros(self.beam_size, device=device)
        beam_scores[1:] = -1e9  # 只有第一个beam是活跃的
        
        # input_ids: [beam_size, 1], 每个beam的当前序列
        input_ids = torch.full((self.beam_size, 1), self.bos_token_id, dtype=torch.long, device=device)
        
        # 完成的假设
        finished_hypotheses = []

        for step in range(self.max_length):
            # 1. 前向传播
            # 扩展memory以匹配beam_size
            expanded_memory = memory.expand(self.beam_size, -1, -1)
            
            # 创建causal mask
            tgt_mask = self.model.create_causal_mask(input_ids.size(1)).to(device)
            
            # 解码
            logits = self.model.decode(input_ids, expanded_memory, tgt_mask=tgt_mask)
            next_token_logits = logits[:, -1, :]  # [beam_size, vocab_size]

            # 2. 计算下一个token的概率
            log_probs = F.log_softmax(next_token_logits, dim=-1)

            # 3. 计算新候选的分数
            # next_scores: [beam_size, vocab_size]
            next_scores = log_probs + beam_scores.unsqueeze(1).expand_as(log_probs)

            # 4. 选择top-k候选
            # next_scores: [beam_size * vocab_size]
            next_scores = next_scores.view(-1)
            # topk返回 (values, indices)
            top_scores, top_indices = torch.topk(next_scores, 2 * self.beam_size)
            
            # 将一维索引解码为 beam_idx 和 token_id
            beam_indices = top_indices // vocab_size
            token_ids = top_indices % vocab_size

            # 5. 更新
            next_beam_scores = torch.zeros(self.beam_size, device=device)
            next_input_ids = torch.zeros(self.beam_size, input_ids.size(1) + 1, dtype=torch.long, device=device)
            
            beam_count = 0
            for score, beam_idx, token_id in zip(top_scores, beam_indices, token_ids):
                if beam_count >= self.beam_size:
                    break
                
                # 获取父beam的序列
                prev_sequence = input_ids[beam_idx]
                new_sequence = torch.cat([prev_sequence, token_id.unsqueeze(0)])
                
                if token_id.item() == self.eos_token_id:
                    # 序列完成
                    finished_hypotheses.append(
                        BeamHypothesis(tokens=new_sequence.tolist(), score=score.item())
                    )
                else:
                    # 添加到下一个活跃beam
                    next_beam_scores[beam_count] = score
                    next_input_ids[beam_count] = new_sequence
                    beam_count += 1
            
            if beam_count == 0:  # 所有beam都已完成
                break
                
            input_ids = next_input_ids[:beam_count]
            beam_scores = next_beam_scores[:beam_count]
            
            # 如果完成的假设足够多，可以早停
            if len(finished_hypotheses) >= self.beam_size and self.early_stopping:
                break
        
        # 添加未完成的假设
        for i in range(input_ids.size(0)):
            finished_hypotheses.append(
                BeamHypothesis(tokens=input_ids[i].tolist(), score=beam_scores[i].item())
            )
        
        # 按归一化分数排序
        finished_hypotheses.sort(
            key=lambda h: h.normalized_score(self.length_penalty),
            reverse=True
        )
        
        # 准备返回结果
        result = {'sequences': [], 'scores': [], 'raw_scores': [], 'lengths': []}
        for h in finished_hypotheses[:self.beam_size]:
            tokens = h.tokens[1:]  # 移除BOS
            if tokens and tokens[-1] == self.eos_token_id:
                tokens = tokens[:-1]
            if not tokens:  # 处理空序列
                tokens = [self.unk_token_id]

            result['sequences'].append(tokens)
            result['scores'].append(h.normalized_score(self.length_penalty))
            result['raw_scores'].append(h.score)
            result['lengths'].append(len(tokens))
            
        return result

    def generate(self, src: torch.Tensor, **kwargs) -> Dict[str, Any]:
        """简化接口"""
        return self.search(src, **kwargs)

def beam_search_decode(model, src, tokenizer=None, beam_size=4, max_length=100, **kwargs):
    """兼容旧版本的函数接口"""
    decoder = BeamSearchDecoder(
        model=model,
        tokenizer=tokenizer,
        beam_size=beam_size,
        max_length=max_length,
        **kwargs
    )
    return decoder.search(src)

def create_beam_search_decoder(model, tokenizer=None, beam_size=4, length_penalty=0.6, **kwargs):
    """创建Beam Search解码器"""
    return BeamSearchDecoder(
        model=model,
        tokenizer=tokenizer,
        beam_size=beam_size,
        length_penalty=length_penalty,
        **kwargs
    )

if __name__ == "__main__":
    print("Beam Search解码器 (重构版) - 简化且高效")
    print("特点:")
    print("- 解决批处理效率问题")
    print("- 简化逻辑复杂性")
    print("- 目前只支持batch_size=1")
    print("- 优化的内存使用")