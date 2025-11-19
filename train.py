# train.py

import torch
import torch.nn as nn
import torch.optim as optim
import logging
import argparse
import sys
from tqdm import tqdm # برای نمایش نوار پیشرفت

# فراخوانی ماژول‌های پروژه
from utils import set_seed
from mamba_dataset import get_mamba_dataloader
from models import MambaAnomalyDetector

logger = logging.getLogger("TRAIN")
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')


def train_model(args):
    """
    تعریف حلقه آموزش و اجرای آن.
    """
    
    logger.info("--- Starting Mamba Anomaly Detector Training ---")
    
    # 1. تنظیم دستگاه و Seed
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using Device: {device}")
    set_seed(args.seed)
    
    # 2. لود داده و DataLoader
    dataloader = get_mamba_dataloader(
        walk_length=args.walk_length, 
        batch_size=args.batch_size, 
        data_dir=args.data_dir
    )
    
    # فرض بر این است که داده‌ها لود شده و شکل Feature مشخص است
    first_batch = next(iter(dataloader))
    feat_dim = first_batch[1].shape[1] 
    
    # 3. تعریف مدل، تابع زیان و بهینه‌ساز
    model = MambaAnomalyDetector(
        input_dim=feat_dim, 
        d_model=args.d_model, 
        n_layer=args.n_layer
    ).to(device)
    
    # تابع زیان: Mean Squared Error (MSE) برای بازسازی ویژگی‌ها
    criterion = nn.MSELoss() 
    
    # بهینه‌ساز: Adam
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    
    logger.info(f"Model initialized. Optimizer: Adam (LR={args.learning_rate}), Loss: MSE")
    
    # 4. حلقه آموزش (Training Loop)
    model.train() # مدل را در حالت آموزش قرار می‌دهیم
    
    for epoch in range(args.epochs):
        total_loss = 0
        
        # استفاده از tqdm برای نمایش نوار پیشرفت
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{args.epochs}", leave=False)
        
        for batch_idx, (masked_seq, target_feat, _) in enumerate(progress_bar):
            
            # انتقال داده به دستگاه (GPU)
            masked_seq = masked_seq.to(device)
            target_feat = target_feat.to(device)
            
            # صفر کردن گرادیان‌ها
            optimizer.zero_grad()
            
            # Forward Pass: بازسازی ویژگی هدف
            reconstructed_feat = model(masked_seq)
            
            # محاسبه Loss
            loss = criterion(reconstructed_feat, target_feat)
            
            # Backward Pass: محاسبه گرادیان‌ها
            loss.backward()
            
            # به‌روزرسانی وزن‌ها
            optimizer.step()
            
            total_loss += loss.item()
            
            # به‌روزرسانی نوار پیشرفت با نمایش Loss فعلی
            progress_bar.set_postfix({'Loss': f'{loss.item():.6f}'})

        avg_loss = total_loss / len(dataloader)
        logger.info(f"Epoch {epoch+1}/{args.epochs} Completed. Average Loss: {avg_loss:.6f}")

    logger.info("Training finished successfully.")
    
    # 5. ذخیره مدل
    torch.save(model.state_dict(), args.model_path)
    logger.info(f"Model weights saved to: {args.model_path}")

# #####################################################################

def arg_parse():
    """تعریف و پارس کردن آرگومان‌های خط فرمان پروژه."""
    parser = argparse.ArgumentParser(description="Mamba-GNN Anomaly Detection Training.")
    
    # --- پارامترهای مدل و آموزش ---
    parser.add_argument('--model_path', type=str, default='best_mamba_model.pt',
                        help='مسیر ذخیره فایل مدل (پیش‌فرض: best_mamba_model.pt)')
    parser.add_argument('--epochs', type=int, default=10,
                        help='تعداد دوره‌های آموزش (پیش‌فرض: 10)')
    parser.add_argument('--learning_rate', type=float, default=1e-3,
                        help='نرخ یادگیری بهینه‌ساز Adam (پیش‌فرض: 0.001)')
    parser.add_argument('--d_model', type=int, default=128, 
                        help='بعد داخلی (Embedding) مدل Mamba (پیش‌فرض: 128)')
    parser.add_argument('--n_layer', type=int, default=4, 
                        help='تعداد بلوک‌های Mamba (پیش‌فرض: 4)')

    # --- پارامترهای داده (همانند main.py) ---
    parser.add_argument('--data_dir', type=str, default='data/raw',
                        help='مسیر دایرکتوری حاوی فایل YelpChi.mat (پیش‌فرض: data/raw)')
    parser.add_argument('--walk_length', type=int, default=32,
                        help='طول مسیر تصادفی (Sequence Length) برای ورودی Mamba (پیش‌فرض: 32)')
    parser.add_argument('--batch_size', type=int, default=16,
                        help='اندازه بچ (Batch Size) برای DataLoader (پیش‌فرض: 16)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Seed برای تضمین تکرارپذیری نتایج (پیش‌فرض: 42)')

    return parser.parse_args()


if __name__ == "__main__":
    # ⚠️ توجه: مطمئن شوید پکیج tqdm نصب شده است
    # !conda run -n old_env pip install tqdm 
    args = arg_parse()
    train_model(args)
