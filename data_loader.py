# data_loader.py

import torch
import scipy.io as sio
import numpy as np
from torch_geometric.data import Data
from pathlib import Path
import logging
import argparse
import sys

# --- ماژول‌های لازم برای دانلود ---
import kagglehub
import shutil
import os
# ---------------------------------

logger = logging.getLogger(__name__)

# برای جلوگیری از تداخل لاگ‌گیری در محیط‌های مختلف
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# --- تنظیمات عمومی داده ---
FILE_NAME = "YelpChi.mat"
DATASET_ID = "wangkezju/graphdata"
FILE_PATH_IN_DATASET = f"data/{FILE_NAME}"


def download_yelpchi(data_dir: str = "data/raw"):
    """
    دانلود فایل YelpChi.mat از Kaggle Hub به پوشه محلی.
    """
    DATA_DIR = Path(data_dir)
    FINAL_FILE_PATH = DATA_DIR / FILE_NAME
    
    # ۱. ایجاد مسیر نهایی
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    if FINAL_FILE_PATH.exists():
        logger.info(f"✅ File already exists: {FINAL_FILE_PATH}")
        return
    else:
        try:
            logger.info(f"⬇️ Downloading '{FILE_NAME}' from Kaggle Hub...")
            
            # ۲. دانلود/دسترسی به فایل در پوشه کش (مسیر فقط خواندنی)
            # توجه: KaggleHub ابتدا به پوشه کش نگاه می‌کند
            local_cache_path = kagglehub.dataset_download(DATASET_ID, path=FILE_PATH_IN_DATASET)
            
            logger.info(f"📂 File path in read-only system cache: {local_cache_path}")
            
            # ۳. کپی کردن به مسیر نهایی پروژه
            logger.info(f"➡️ Copying file to final writable path: {FINAL_FILE_PATH}...")
            shutil.copy(local_cache_path, FINAL_FILE_PATH)
            
            logger.info(f"🎉 Download and copy complete. File available at **{FINAL_FILE_PATH}**")
            
        except Exception as e:
            logger.error(f"\n❌ FATAL: Error during download or copy. Check your Kaggle permissions/internet connection: {e}")
            sys.exit(1)


def load_yelpchi(data_dir: str = "data/raw") -> Data:
    """
    وظیفه: دانلود دیتاست (در صورت لزوم) و بارگذاری آن در آبجکت PyG Data.
    """
    # گام جدید: قبل از تلاش برای خواندن، دانلود را اجرا کن
    download_yelpchi(data_dir)
    
    file_path = Path(data_dir) / FILE_NAME
    
    # --- ادامه منطق قبلی برای خواندن فایل ---
    
    logger.info(f"Loading raw data from {file_path}...")
    # فرض می‌کنیم فایل اکنون وجود دارد و در صورت عدم وجود، تابع download_yelpchi ارور داده است.
    mat = sio.loadmat(str(file_path))

    # --- Processing logic (تبدیل به Undirected, حذف Self-loops) ---
    adj = mat["homo"].tocoo()
    row, col = adj.row.astype(np.int64), adj.col.astype(np.int64)
    
    if len(row) < 4_000_000: 
        row, col = np.concatenate([row, col]), np.concatenate([col, row])
        
    edges = np.unique(np.column_stack([row, col])[np.column_stack([row, col])[:, 0] != np.column_stack([row, col])[:, 1]], axis=0)
    edge_index = torch.tensor(edges.T, dtype=torch.long)
    
    # Load Features
    x = torch.tensor(mat["features"].toarray(), dtype=torch.float) if "features" in mat else torch.eye(adj.shape[0], dtype=torch.float)
    y = torch.tensor(mat["label"].flatten(), dtype=torch.long)
    
    logger.info(f"Raw Data Loaded. Nodes: {x.size(0):,} | Features: {x.size(1)} | Anomaly Rate: {y.float().mean().item():.4f}")
    
    return Data(x=x, edge_index=edge_index, y=y)
