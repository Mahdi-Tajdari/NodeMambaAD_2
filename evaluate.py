# evaluate.py

import torch
import torch.nn as nn
import logging
import argparse
import sys
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score
from tqdm import tqdm

# فراخوانی ماژول‌های پروژه
from utils import set_seed
from mamba_dataset import get_mamba_dataloader
from models import MambaAnomalyDetector

logger = logging.getLogger("EVAL")
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

def calculate_anomaly_scores(model, dataloader, device):
    """
    محاسبه خطای بازسازی (Anomaly Score) برای هر گره در DataLoader.
    """
    model.eval() # مدل را در حالت ارزیابی قرار می‌دهیم
    
    # لیست‌هایی برای ذخیره نتایج
    all_scores = [] # امتیاز ناهنجاری (خطای بازسازی)
    all_labels = [] # لیبل‌های واقعی (0: نرمال، 1: ناهنجاری)

    # 💡 تابع زیان: MSE برای محاسبه خطای بازسازی
    reconstruction_loss_fn = nn.MSELoss(reduction='none') # reduction='none' برای گرفتن خطای هر نمونه

    with torch.no_grad(): # در ارزیابی، نیازی به محاسبه گرادیان نیست
        for masked_seq, target_feat, anomaly_label in tqdm(dataloader, desc="Calculating Anomaly Scores"):
            
            # انتقال داده به دستگاه (GPU)
            masked_seq = masked_seq.to(device)
            target_feat = target_feat.to(device)
            
            # Forward Pass: بازسازی ویژگی هدف
            reconstructed_feat = model(masked_seq)
            
            # محاسبه خطای بازسازی (MSE Loss) برای هر نمونه
            # خروجی: [Batch_Size, Feature_Dim]
            reconstruction_error_per_feature = reconstruction_loss_fn(reconstructed_feat, target_feat)
            
            # جمع خطاهای ویژگی‌ها برای گرفتن یک "امتیاز" نهایی برای هر نود در Batch
            # خروجی: [Batch_Size]
            node_anomaly_score = reconstruction_error_per_feature.mean(dim=1)
            
            # ذخیره نتایج
            all_scores.append(node_anomaly_score.cpu().numpy())
            all_labels.append(anomaly_label.cpu().numpy())

    # ترکیب تمام بچ‌ها
    scores = np.concatenate(all_scores)
    labels = np.concatenate(all_labels)
    
    return scores, labels


def evaluate_model(args):
    """
    اجرای ارزیابی مدل و محاسبه معیارهای دقت.
    """
    
    logger.info("--- Starting Model Evaluation ---")
    
    # 1. تنظیم دستگاه و Seed
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using Device: {device}")
    set_seed(args.seed)
    
    # 2. لود داده و DataLoader (استفاده از تمام داده‌ها برای ارزیابی)
    dataloader = get_mamba_dataloader(
        walk_length=args.walk_length, 
        batch_size=args.batch_size, 
        data_dir=args.data_dir,
        shuffle=False # برای ارزیابی نباید داده‌ها را به هم بزنیم
    )
    
    # 3. تعریف و بارگذاری مدل آموزش دیده
    first_batch = next(iter(dataloader))
    feat_dim = first_batch[1].shape[1] 
    
    model = MambaAnomalyDetector(
        input_dim=feat_dim, 
        d_model=args.d_model, 
        n_layer=args.n_layer
    ).to(device)
    
    try:
        model.load_state_dict(torch.load(args.model_path, map_location=device))
        logger.info(f"Model weights successfully loaded from: {args.model_path}")
    except FileNotFoundError:
        logger.error(f"FATAL: Model file not found at {args.model_path}. Please run train.py first.")
        sys.exit(1)
        
    # 4. محاسبه امتیازات ناهنجاری
    scores, labels = calculate_anomaly_scores(model, dataloader, device)
    
    # 5. محاسبه معیارهای ارزیابی (Metrics)
    
    # 5.1. AUC-ROC (Area Under the Receiver Operating Characteristic Curve)
    # این معیار نشان می‌دهد مدل چقدر توانایی تفکیک ناهنجاری‌ها از داده‌های نرمال را دارد.
    auc_roc = roc_auc_score(labels, scores)
    
    # 5.2. AUC-PR (Area Under the Precision-Recall Curve)
    # در مسائل ناهنجاری که لیبل‌های مثبت (ناهنجاری‌ها) بسیار کم هستند، AUC-PR معیار مهم‌تری است.
    auc_pr = average_precision_score(labels, scores)
    
    # 6. نمایش نتایج
    logger.info("--- Evaluation Results ---")
    logger.info(f"Total Nodes Evaluated: {len(labels)}")
    logger.info(f"True Anomaly Nodes: {np.sum(labels)}")
    logger.info(f"AUC-ROC Score: {auc_roc:.4f}")
    logger.info(f"AUC-PR Score: {auc_pr:.4f}")
    logger.info("--------------------------")
    
# #####################################################################

def arg_parse():
    """تعریف و پارس کردن آرگومان‌های خط فرمان پروژه."""
    parser = argparse.ArgumentParser(description="Mamba Anomaly Detector Evaluation.")
    
    # --- پارامترهای مدل و مسیر ---
    parser.add_argument('--model_path', type=str, default='best_mamba_model.pt',
                        help='مسیر فایل وزن مدل (پیش‌فرض: best_mamba_model.pt)')
    parser.add_argument('--d_model', type=int, default=128, 
                        help='بعد داخلی (Embedding) مدل Mamba (پیش‌فرض: 128)')
    parser.add_argument('--n_layer', type=int, default=4, 
                        help='تعداد بلوک‌های Mamba (پیش‌فرض: 4)')

    # --- پارامترهای داده ---
    parser.add_argument('--data_dir', type=str, default='data/raw',
                        help='مسیر دایرکتوری حاوی فایل YelpChi.mat (پیش‌فرض: data/raw)')
    parser.add_argument('--walk_length', type=int, default=32,
                        help='طول مسیر تصادفی (Sequence Length) برای ورودی Mamba (پیش‌فرض: 32)')
    parser.add_argument('--batch_size', type=int, default=512, # اندازه بچ بزرگتر برای ارزیابی
                        help='اندازه بچ (Batch Size) برای DataLoader (پیش‌فرض: 512)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Seed برای تضمین تکرارپذیری نتایج (پیش‌فرض: 42)')

    return parser.parse_args()


if __name__ == "__main__":
    args = arg_parse()
    # ⚠️ اطمینان حاصل کنید که پکیج scikit-learn نصب شده باشد
    # !conda run -n old_env pip install scikit-learn
    
    # ما از tqdm.write استفاده نمیکنیم زیرا در evaluate، لاگ‌ها معمولاً تداخلی ایجاد نمی‌کنند
    evaluate_model(args)
