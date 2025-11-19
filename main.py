# main.py

import torch
import logging
import argparse
import sys
from pathlib import Path

# فراخوانی ماژول‌های پروژه
from utils import set_seed
from mamba_dataset import get_mamba_dataloader
from models import MambaAnomalyDetector 

# تنظیمات لاگر
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("MAIN")


def arg_parse():
    """تعریف و پارس کردن آرگومان‌های خط فرمان پروژه."""
    parser = argparse.ArgumentParser(description="Mamba-GNN Anomaly Detection Training and Verification.")
    
    # --- پارامترهای داده و بارگذاری ---
    parser.add_argument('--data_dir', type=str, default='data/raw',
                        help='مسیر دایرکتوری حاوی فایل YelpChi.mat (پیش‌فرض: data/raw)')
    parser.add_argument('--walk_length', type=int, default=32,
                        help='طول مسیر تصادفی (Sequence Length) برای ورودی Mamba (پیش‌فرض: 32)')
    parser.add_argument('--batch_size', type=int, default=16,
                        help='اندازه بچ (Batch Size) برای DataLoader (پیش‌فرض: 16)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Seed برای تضمین تکرارپذیری نتایج (پیش‌فرض: 42)')
                        
    # --- پارامترهای مدل Mamba ---
    parser.add_argument('--d_model', type=int, default=128, 
                        help='بعد داخلی (Embedding) مدل Mamba (پیش‌فرض: 128)')
    parser.add_argument('--n_layer', type=int, default=4, 
                        help='تعداد بلوک‌های Mamba (پیش‌فرض: 4)')

    return parser.parse_args()


def verify_pipeline(args):
    """
    اجرای کامل پایپ‌لاین داده و تست مدل با استفاده از آرگومان‌های ورودی.
    """
    
    logger.info("--- Starting Pipeline and Model Verification (Smoke Test) ---")
    
    # --- تعریف دستگاه (Device) ---
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using Device: {device}")
    
    # 1. تنظیم Seed
    set_seed(args.seed)
    
    # 2. ساخت DataLoader
    logger.info(f"Initializing DataLoader with BATCH_SIZE={args.batch_size}, WALK_LENGTH={args.walk_length}")
    try:
        dataloader = get_mamba_dataloader(
            walk_length=args.walk_length, 
            batch_size=args.batch_size, 
            data_dir=args.data_dir
        )
    except Exception as e:
        logger.error(f"FATAL: Error during DataLoader initialization. Check download logs. Error: {e}")
        sys.exit(1)
        
    # 3. گرفتن اولین بچ و تست مدل
    try:
        masked_seq, target_feat, anomaly_label = next(iter(dataloader))
        
        # 3.1. انتقال داده به دستگاه
        masked_seq = masked_seq.to(device)
        target_feat = target_feat.to(device)
        
        # استخراج بعد ویژگی‌ها
        feat_dim = target_feat.shape[1]
        
        # --- تست مدل ---
        logger.info(f"Initializing Mamba model (Input Dim: {feat_dim}, d_model: {args.d_model}).")
        model = MambaAnomalyDetector(input_dim=feat_dim, 
                                     d_model=args.d_model, 
                                     n_layer=args.n_layer)
        
        # 3.2. انتقال مدل به دستگاه
        model.to(device)
        
        # اجرای Forward Pass
        model.eval() 
        with torch.no_grad():
            reconstruction_output = model(masked_seq)
            
        # 4. اعتبارسنجی شکل خروجی
        expected_output_shape = target_feat.shape 
        assert reconstruction_output.shape == expected_output_shape
        
        logger.info("\n✅ SUCCESS: Full Pipeline (Data Loading + Model Forward Pass) successful.")
        logger.info(f"  Input Shape to Mamba (B, L, D_in): {masked_seq.shape}")
        logger.info(f"  Target Shape (B, D_in): {target_feat.shape}")
        logger.info(f"  Model Output Shape (Reconstruction): {reconstruction_output.shape}")
        
    except Exception as e:
        logger.error(f"FATAL: Error during batch fetching or model forward pass: {e}")
        sys.exit(1)

# #####################################################################

if __name__ == "__main__":
    args = arg_parse()
    verify_pipeline(args)
