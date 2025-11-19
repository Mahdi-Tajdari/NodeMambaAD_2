# mamba_dataset.py

import torch
from torch.utils.data import Dataset, DataLoader
from typing import List, Tuple
import random
import logging

# توابع عمومی مورد نیاز را از utils فراخوانی می‌کنیم
from utils import build_adjacency_list 
# داده خام را از data_loader فراخوانی می‌کنیم
from data_loader import load_yelpchi 

logger = logging.getLogger(__name__)

# #####################################################################
#                 توابع محلی: Random Walk و Masking
# #####################################################################

def random_walk_sampling(start_node: int, adj_list: List[List[int]], walk_length: int) -> List[int]:
    """
    اجرای Random Walk برای تولید دنباله ورودی Mamba.
    """
    walk = [start_node]
    curr = start_node
    
    for _ in range(walk_length - 1):
        neighbors = adj_list[curr]
        if len(neighbors) > 0:
            # گام تصادفی
            curr = random.choice(neighbors)
            walk.append(curr)
        else:
            # درجا زدن در صورت عدم وجود همسایه
            walk.append(curr)
            
    return walk

def mask_feature_sequence(sequence_features: torch.Tensor, mask_token: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    اعمال Masking روی نود هدف (اولین عضو دنباله).
    """
    masked_seq = sequence_features.clone()
    # نود هدف همیشه اولین نود در دنباله است
    target_feat = sequence_features[0].clone()
    
    # ماسک کردن نود هدف با توکن ماسک (بردار صفر)
    masked_seq[0] = mask_token
        
    return masked_seq, target_feat

# #####################################################################
#                   کلاس اصلی: MambaGraphDataset
# #####################################################################

class MambaGraphDataset(Dataset):
    
    def __init__(self, data_dir: str = "data/raw", walk_length: int = 64):
        
        # ۱. لود داده خام
        self.data = load_yelpchi(data_dir=data_dir)
        self.x = self.data.x  # ویژگی‌های گره (Node Features)
        self.y = self.data.y  # لیبل‌های ناهنجاری (Anomaly Labels)
        self.num_nodes = self.x.size(0)
        self.walk_length = walk_length
        
        # ۲. بهینه‌سازی لیست مجاورت (برای Random Walk)
        self.adj_list = build_adjacency_list(self.data.edge_index, self.num_nodes)
        
        # ۳. توکن ماسک (بردار صفر) با همان ابعاد ویژگی
        self.mask_token = torch.full((self.x.size(1),), 0.0)

    def __len__(self):
        # بازگرداندن تعداد کل گره‌ها
        return self.num_nodes

    def __getitem__(self, idx: int):
        
        # الف. ساخت دنباله ایندکس‌ها با شروع از نود idx
        node_indices_walk = random_walk_sampling(
            start_node=idx, 
            adj_list=self.adj_list, 
            walk_length=self.walk_length
        )
        
        # ب. تبدیل ایندکس‌ها به ویژگی‌ها [L, D]
        sequence_features = self.x[node_indices_walk]
        
        # ج. اعمال Masking برای فاز بازسازی
        masked_seq, target_feat = mask_feature_sequence(
            sequence_features=sequence_features, 
            mask_token=self.mask_token
        )
        
        # بازگرداندن: دنباله ماسک شده، ویژگی هدف، لیبل ناهنجاری
        return masked_seq, target_feat, self.y[idx]


def get_mamba_dataloader(walk_length: int, 
                         batch_size: int = 32, 
                         data_dir: str = "data/raw", 
                         shuffle: bool = True, # 👈 اصلاح: اضافه کردن پارامتر shuffle
                         **kwargs) -> DataLoader:
    """
    تابع اصلی برای تولید DataLoader.
    """
    dataset = MambaGraphDataset(data_dir=data_dir, walk_length=walk_length)
    
    return DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=shuffle, # 👈 اصلاح: استفاده از پارامتر ورودی
        drop_last=True,
        **kwargs
    )
