import os
import torch

import torch

def AUC(pos_logits, pos_labels, neg_logits=None, neg_labels=None):
    """
    计算AUC指标（不考虑相同分数）
    
    参数：
        pos_logits: 正样本预测分数 [pos_len]
        pos_labels: 正样本标签 [pos_len]
        neg_logits: 负样本预测分数 [neg_len] (可选)
        neg_labels: 负样本标签 [neg_len] (可选)
    
    返回：
        AUC值 (标量)
    """
    if neg_logits is not None and neg_labels is not None:
        logits = torch.cat([pos_logits, neg_logits], dim=0)
        labels = torch.cat([pos_labels, neg_labels], dim=0)
    else:
        logits = pos_logits
        labels = pos_labels
    
    sorted_logits, sorted_indices = torch.sort(logits, descending=False)
    sorted_labels = labels[sorted_indices]
    
    ranks = torch.arange(1, len(sorted_logits) + 1, device=logits.device, dtype=torch.float)
    n_pos = torch.sum(sorted_labels == 1)
    n_neg = torch.sum(sorted_labels == 0)
    
    if n_pos == 0 or n_neg == 0:
        return 0.5
    
    rank_sum = torch.sum(ranks[sorted_labels == 1]).item()
    auc = (rank_sum - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)
    
    return auc





def GAUC(pos_logits, pos_labels, neg_logits=None, neg_labels=None, next_token_type=None):
    """
    计算分组AUC（按用户分组）
    
    参数：
        pos_logits: 正样本预测分数 [B, pos_len]
        pos_labels: 正样本标签 [B, pos_len]
        neg_logits: 负样本预测分数 [B, neg_len]
        neg_labels: 负样本标签 [B, neg_len]
        indices: 元组 (row_indices, col_indices) 
                指定哪些位置是正样本（如torch.where(next_token_type==1))
    
    返回：
        GAUC值 (所有用户AUC的平均值)
    """
    # 获取行索引（用户索引）和列索引

    if next_token_type is not None:
        indices = torch.where(next_token_type==1)
        row_indices, col_indices = indices
    
    unique_rows = torch.unique(row_indices)
    user_aucs = []
    
    user_pos_logits = None
    user_pos_labels = None
    user_neg_logits = None
    user_neg_labels = None

    for row in unique_rows:
        if next_token_type is None:
            user_pos_logits = pos_logits[row]
            user_pos_labels = pos_labels[row]
        else:
            mask = (row_indices == row)
            user_cols = col_indices[mask]
            
            user_pos_logits = pos_logits[row][user_cols]
            user_pos_labels = pos_labels[row][user_cols]    
            if neg_logits is not None and neg_labels is not None:
                user_neg_logits = neg_logits[row][user_cols]
                user_neg_labels = neg_labels[row][user_cols]
        user_auc = AUC(
            user_pos_logits, 
            user_pos_labels,
            user_neg_logits,
            user_neg_labels
        )
   
        user_aucs.append(user_auc)
    return torch.tensor(user_aucs).mean().item()
    

def HitRate(pos_logits, neg_logits, next_token_type, next_action_type, topK=10):
    """
    计算HitRate@K指标
    
    参数：
        pos_logits: 正样本预测分数 [B, L]
        neg_logits: 负样本预测分数 [B, N]
        next_token_type: token类型 [B, L]
        next_action_type: action类型 [B, L] (标签)
        topK: 计算Top-K的K值
    
    返回：
        HitRate值 (标量)
    """
    assert pos_logits.shape == next_action_type.shape and pos_logits.shape == next_token_type.shape
    if neg_logits is not None:
        logits = torch.cat([pos_logits, neg_logits], dim=-1)
    else:
        logits = pos_logits
    
    
    next_token_type = torch.cat([next_token_type, torch.zeros((neg_logits.shape), device=next_token_type.device, dtype=next_token_type.dtype)], dim=-1)
    next_action_type = torch.cat([next_action_type, torch.zeros((neg_logits.shape), device=next_action_type.device, dtype=next_action_type.dtype)], dim=-1)
    
    hit_labels = (next_token_type == 1) & (next_action_type == 1)
    
    _, sorted_indices = torch.sort(logits, dim=-1, descending=True)
    
    topk_indices = sorted_indices[:, :topK]
    topk_labels = torch.gather(hit_labels, 1, topk_indices)
    
    if hit_labels.sum().item() > 0:
        return topk_labels.sum().item() / hit_labels.sum().item()
    else:
        return 0.0 

def NDCG(pos_logits, neg_logits, next_token_type, next_action_type=None, topK=10):
    """
    计算NDCG指标
    inputs:
        pos_logits: 正样本预测分数 [B, L]
        neg_logits: 负样本预测分数 [B, N]
        next_token_type: token类型 [B, L]   (位置标签)
        next_action_type: action类型 [B, L] (行为标签)
    outputs:
        NDCG值
    """
    if next_action_type is None:
        next_action_type = torch.zeros_like(next_token_type)

    if neg_logits is not None:
        logits = torch.cat([pos_logits, neg_logits], dim=-1)   
        next_token_type = torch.cat([next_token_type, torch.zeros_like(next_token_type)], dim=-1) 
        next_action_type = torch.cat([next_action_type, torch.zeros_like(next_action_type)], dim=-1) 
    else:
        logits = pos_logits

    _, sorted_indices = torch.sort(logits, dim=-1, descending=True)
    relevance = ((next_token_type == 1) & (next_action_type == 1)).float()
    discount = 1.0 / torch.log2(torch.arange(2, topK + 2, device=logits.device, dtype=torch.float))

    ndcg_scores = []
    for i in range(logits.size(0)):
        user_sorted_rel = relevance[i, sorted_indices[i]]
        
        # 计算DCG: ∑(rel_i / log2(排名+1))
        k = min(topK, len(user_sorted_rel))
        dcg = (user_sorted_rel[:k] * discount[:k]).sum()
        
        # 计算IDCG (理想排序的DCG)
        ideal_rel_sorted, _ = torch.sort(relevance[i], descending=True)
        idcg = (ideal_rel_sorted[:k] * discount[:k]).sum()
        
        ndcg = dcg / idcg if idcg > 0 else torch.tensor(0.0, device=logits.device)
        ndcg_scores.append(ndcg)
    
    return torch.stack(ndcg_scores).mean().item()

"""
example in main:
auc = AUC(pos_logits[indices], pos_labels[indices], neg_logits[indices], neg_labels[indices])
gauc = GAUC(pos_logits, pos_labels, neg_logits, neg_labels, next_token_type)
hitrate = HitRate(pos_logits, neg_logits, next_token_type, next_action_type)
ndcg = NDCG(pos_logits, neg_logits, next_token_type, next_action_type)
"""