# utils.py

import torch
import numpy as np
import random
import logging
from typing import List, Dict, Any, Tuple

logger = logging.getLogger(__name__)

# --- توابع عمومی ---

def set_seed(seed: int = 42):
    """تنظیم Seed برای تکرارپذیری."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    logger.info(f"Global seed set to: {seed}")

def build_adjacency_list(edge_index: torch.Tensor, num_nodes: int) -> List[List[int]]:
    """
    تبدیل edge_index به لیست مجاورت (برای دسترسی سریع همسایگان در Random Walk).
    این یک ابزار بهینه‌سازی عمومی است.
    """
    adj_list = [[] for _ in range(num_nodes)]
    rows = edge_index[0].cpu().numpy()
    cols = edge_index[1].cpu().numpy()
    
    for r, c in zip(rows, cols):
        adj_list[r].append(c)
    
    return adj_list

# توابع مربوط به محاسبه AUC-ROC و AUC-PR در اینجا قرار می‌گیرند (بعداً اضافه خواهیم کرد)
# def calculate_metrics(...):
#     ...
