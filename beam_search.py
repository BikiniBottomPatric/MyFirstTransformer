#!/usr/bin/env python3
# beam_search.py (真正的最终正确版)

import torch
import torch.nn.functional as F
import torch.nn as nn
from dataclasses import dataclass
from typing import List, Dict, Any, Optional
import logging
import config

logger = logging.getLogger(__name__)

@dataclass
class BeamHypothesis:
    """Beam Search假设 - 只存储原始数据"""
    tokens: List[int]  # 包含BOS和可能的EOS
    score: float       # 累积的原始log-probability

    def __len__(self):
        # 实际生成长度，不包括BOS
        return len(self.tokens) - 1

    def normalized_score(self, length_penalty: float) -> float:
        """只在最后排序时调用，用于计算长度归一化分数"""
        lp = ((5.0 + len(self)) / 6.0) ** length_penalty
        return self.score / lp

class BeamSearchDecoder:
    """Beam Search解码器（最终正确版）"""
    def __init__(self, model, beam_size: int, max_length: int,
                 length_penalty: float, early_stopping: bool,
                 bos_token_id: int, eos_token_id: int, unk_token_id: int):
        self.model = model
        self.beam_size = beam_size
        self.max_length = max_length
        self.length_penalty = length_penalty
        self.early_stopping = early_stopping
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        self.unk_token_id = unk_token_id

    @torch.no_grad()
    def search(self, src: torch.Tensor, src_mask: Optional[torch.Tensor] = None, verbose: bool = True) -> Dict[str, Any]:
        device = src.device
        batch_size = src.size(0)
        assert batch_size == 1, "Beam Search目前只支持batch_size=1"

        vocab_size = self.model.output_projection.out_features
        memory = self.model.encode(src, src_mask)

        # beam_scores: [beam_size], input_ids: [beam_size, seq_len]
        beam_scores = torch.zeros(self.beam_size, device=device)
        beam_scores[1:] = -1e9
        input_ids = torch.full((self.beam_size, 1), self.bos_token_id, dtype=torch.long, device=device)
        
        finished_hypotheses = []

        for step in range(self.max_length):
            num_active_beams = input_ids.size(0)
            if num_active_beams == 0:
                break

            expanded_memory = memory.expand(num_active_beams, -1, -1)
            tgt_mask = self.model.create_causal_mask(input_ids.size(1)).to(device)
            
            logits = self.model.decode(input_ids, expanded_memory, tgt_mask=tgt_mask)
            log_probs = F.log_softmax(logits[:, -1, :], dim=-1)
            
            next_scores = log_probs + beam_scores.unsqueeze(1)
            next_scores = next_scores.view(-1)
            
            top_scores, top_indices = torch.topk(next_scores, k=self.beam_size * 2)
            
            beam_indices = top_indices // vocab_size
            token_ids = top_indices % vocab_size
            
            next_input_ids_list = []
            next_beam_scores_list = []
            
            for score, beam_idx, token_id in zip(top_scores, beam_indices, token_ids):
                if token_id.item() == self.eos_token_id:
                    tokens_list = input_ids[beam_idx].tolist() + [token_id.item()]
                    finished_hypotheses.append(BeamHypothesis(tokens=tokens_list, score=score.item()))
                else:
                    next_input_ids_list.append(torch.cat([input_ids[beam_idx], token_id.unsqueeze(0)]))
                    next_beam_scores_list.append(score)

                if len(next_input_ids_list) == self.beam_size:
                    break
            
            if not next_input_ids_list:
                break
                
            beam_scores = torch.stack(next_beam_scores_list)
            input_ids = torch.stack(next_input_ids_list)

            if self.early_stopping and len(finished_hypotheses) >= self.beam_size:
                break
        
        # 将剩余的活跃beam也加入完成列表
        for i in range(input_ids.size(0)):
            finished_hypotheses.append(BeamHypothesis(tokens=input_ids[i].tolist(), score=beam_scores[i].item()))
        
        finished_hypotheses.sort(key=lambda h: h.normalized_score(self.length_penalty), reverse=True)
        
        result = {'sequences': [], 'scores': [], 'raw_scores': [], 'lengths': []}
        for h in finished_hypotheses[:self.beam_size]:
            tokens = h.tokens[1:] # 移除BOS
            if tokens and tokens[-1] == self.eos_token_id:
                tokens = tokens[:-1]
            if not tokens:
                tokens = [self.unk_token_id]

            result['sequences'].append(tokens)
            result['scores'].append(h.normalized_score(self.length_penalty))
            result['raw_scores'].append(h.score)
            result['lengths'].append(len(tokens))
            
        return result

def create_beam_search_decoder(model: nn.Module) -> BeamSearchDecoder:
    return BeamSearchDecoder(
        model=model,
        beam_size=config.BEAM_SIZE,
        max_length=config.MAX_DECODE_LENGTH,
        length_penalty=config.LENGTH_PENALTY,
        early_stopping=config.EARLY_STOPPING,
        bos_token_id=config.BOS_IDX,
        eos_token_id=config.EOS_IDX,
        unk_token_id=config.UNK_IDX
    )

if __name__ == "__main__":
    print("🚀 Beam Search解码器 (最终正确版)")