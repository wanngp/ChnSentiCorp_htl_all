import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import jieba
import re
import random
import json
from sklearn.model_selection import train_test_split
# 修改点 1: 导入 accuracy_score
from sklearn.metrics import f1_score, confusion_matrix, accuracy_score
from tqdm import tqdm
from typing import List, Dict, Tuple

# --- 复用训练代码中的必要类和函数 (保持不变) ---
# ... (从 set_seed 到 EnhancedTextDataset 的所有类和函数定义保持原样) ...
def set_seed(seed=42):
    """设置随机种子以保证可复现性"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)

class AdvancedTextAugmentation:
    """高级文本增强类 (预测时基本不用，但 EnhancedTextDataset 依赖它)"""
    def __init__(self):
        self.synonym_dict = {
            '好': ['棒', '不错', '优秀', '很好', '赞', '给力'], '差': ['糟糕', '不好', '很差', '糟', '烂', '垃圾'],
            '喜欢': ['爱', '钟爱', '偏爱', '中意'], '讨厌': ['厌恶', '反感', '不喜欢', '恶心'],
            '满意': ['满足', '认可', '赞同', '称心'], '失望': ['沮丧', '不满', '遗憾', '郁闷'],
            '服务': ['态度', '接待', '待遇'], '环境': ['氛围', '条件', '场所'], '价格': ['费用', '收费', '价位']
        }
        self.negative_enhancers = ['非常', '特别', '极其', '相当', '十分', '很', '超级', '巨']
        self.positive_words = ['好', '棒', '不错', '满意', '喜欢', '推荐']
        self.negative_words = ['差', '糟', '不好', '失望', '讨厌', '垃圾', '恶心', '糟糕', '烂', '坑', '骗人', '后悔']
    def synonym_replacement(self, text: str, prob: float = 0.3) -> str:
        words = list(jieba.cut(text)); new_words = [];
        for word in words:
            if random.random() < prob and word in self.synonym_dict: new_words.append(random.choice(self.synonym_dict[word]))
            else: new_words.append(word)
        return ''.join(new_words)
    def enhance_sentiment_words(self, text: str, sentiment: str) -> str:
        if sentiment == 'negative':
            for neg_word in self.negative_words:
                if neg_word in text and random.random() < 0.4: text = text.replace(neg_word, f'{random.choice(self.negative_enhancers)}{neg_word}', 1)
        return text
    def insert_negation(self, text: str) -> str:
        words = list(jieba.cut(text))
        if len(words) > 2:
            for i, word in enumerate(words):
                if word in self.positive_words and random.random() < 0.3: words.insert(i, '不'); break
        return ''.join(words)
    def tone_adjustment(self, text: str) -> str:
        tone_words = ['真的', '实在', '确实', '简直']
        words = list(jieba.cut(text))
        if len(words) > 1 and random.random() < 0.3:
            insert_pos = random.randint(0, len(words)//2); words.insert(insert_pos, random.choice(tone_words))
        return ''.join(words)
    def augment_negative_samples(self, text: str, num_aug: int = 3) -> List[str]:
        augmented = []; augmented.append(self.synonym_replacement(text, 0.3)); augmented.append(self.enhance_sentiment_words(text, 'negative'))
        augmented.append(self.insert_negation(text)); augmented.append(self.tone_adjustment(text))
        combined = self.enhance_sentiment_words(self.synonym_replacement(text, 0.2), 'negative'); augmented.append(combined)
        return random.sample(augmented, min(num_aug, len(augmented)))

class ResidualBlock(nn.Module):
    """残差连接块"""
    def __init__(self, dim, dropout=0.1):
        super().__init__(); self.linear1 = nn.Linear(dim, dim); self.linear2 = nn.Linear(dim, dim); self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim); self.dropout = nn.Dropout(dropout); self.activation = nn.GELU()
    def forward(self, x):
        residual = x; x = self.norm1(x); x = self.linear1(x); x = self.activation(x); x = self.dropout(x); x = self.linear2(x)
        x = self.dropout(x); return x + residual

class EnhancedTransformerEncoder(nn.Module):
    """增强的Transformer编码器 (结构必须与训练时完全一致)"""
    def __init__(self, vocab_size, d_model=256, nhead=8, num_layers=4, max_len=512, dropout=0.1):
        super().__init__(); self.d_model = d_model; self.max_length = max_len; self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.pos_embedding = nn.Embedding(max_len, d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=d_model * 4, dropout=dropout, activation='gelu', norm_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers); self.attention_pooling = nn.MultiheadAttention(d_model, nhead//2, batch_first=True)
        self.class_attention = nn.MultiheadAttention(d_model, nhead//2, dropout=0.1); self.class_embedding = nn.Embedding(2, d_model)
        self.feature_projection = nn.Linear(d_model * 2, d_model)
        self.negative_branch = nn.Sequential(ResidualBlock(d_model, dropout=0.2), nn.Linear(d_model, d_model//2), nn.LayerNorm(d_model//2), nn.GELU(), nn.Dropout(0.3), ResidualBlock(d_model//2, dropout=0.2), nn.Linear(d_model//2, 1))
        self.positive_branch = nn.Sequential(ResidualBlock(d_model, dropout=0.2), nn.Linear(d_model, d_model//2), nn.LayerNorm(d_model//2), nn.GELU(), nn.Dropout(0.3), ResidualBlock(d_model//2, dropout=0.2), nn.Linear(d_model//2, 1))
        self.global_feature = nn.Sequential(nn.Linear(d_model * 2, d_model), nn.LayerNorm(d_model), nn.GELU(), nn.Dropout(0.2))
    def forward(self, x):
        batch_size, seq_len = x.size(); positions = torch.arange(seq_len, device=x.device).unsqueeze(0).expand(batch_size, -1)
        x = self.embedding(x) + self.pos_embedding(positions); padding_mask = (x.sum(dim=-1) == 0); x = x.transpose(0, 1)
        encoded = self.transformer(x, src_key_padding_mask=padding_mask); encoded = encoded.transpose(0, 1)
        query = encoded.mean(dim=1, keepdim=True); pooled, _ = self.attention_pooling(query, encoded, encoded, key_padding_mask=padding_mask)
        pooled = pooled.squeeze(1); class_embeds = self.class_embedding(torch.tensor([0, 1], device=x.device))
        class_embeds = class_embeds.mean(dim=0, keepdim=True).repeat(pooled.size(0), 1)
        combined_features = torch.cat([pooled, class_embeds], dim=-1); global_features = self.global_feature(combined_features)
        neg_logit = self.negative_branch(global_features); pos_logit = self.positive_branch(global_features)
        return torch.cat([neg_logit, pos_logit], dim=1)

def advanced_text_cleaning(text):
    """高级文本清理函数 (必须与训练时完全一致)"""
    if pd.isna(text): return ""; text = str(text); text = re.sub(r'[！!]{2,}', '!', text); text = re.sub(r'[？?]{2,}', '?', text)
    text = re.sub(r'[。.]{2,}', '。', text); text = re.sub(r'(.)\1{3,}', r'\1\1', text)
    text = re.sub(r'[^\u4e00-\u9fa5a-zA-Z0-9\s！？。，、；：""''（）!?.,;:""''()]', '', text); text = re.sub(r'\s+', '', text)
    return text.strip()

class EnhancedTextDataset(Dataset):
    """增强的文本数据集 (预测时关闭数据增强)"""
    def __init__(self, texts, labels, vocab, max_length=128, augment_negative=False): # 预测时关闭增强
        self.texts = texts; self.labels = labels; self.vocab = vocab; self.max_length = max_length
        self.augment_negative = augment_negative; self.augmenter = AdvancedTextAugmentation()
    def advanced_text_cleaning(self, text):
        if pd.isna(text): return ""
        text = str(text); text = re.sub(r'[！!]{2,}', '!', text); text = re.sub(r'[？?]{2,}', '?', text); text = re.sub(r'[。.]{2,}', '。', text)
        text = re.sub(r'(.)\1{2,}', r'\1\1', text); text = re.sub(r'[^\u4e00-\u9fa5a-zA-Z0-9\s！？。，、；：""''（）!?.,;:""''()]', '', text)
        text = re.sub(r'\s+', '', text); return text.strip()
    def text_to_indices(self, text):
        text = self.advanced_text_cleaning(text); words = list(jieba.cut(text))
        indices = [self.vocab.get(word, self.vocab['<UNK>']) for word in words]
        if len(indices) > self.max_length: indices = indices[:self.max_length]
        else: indices.extend([self.vocab['<PAD>']] * (self.max_length - len(indices)))
        return torch.tensor(indices, dtype=torch.long)
    def __len__(self): return len(self.texts)
    def __getitem__(self, idx):
        text = self.texts[idx]; label = self.labels[idx]; indices = self.text_to_indices(text)
        return indices, torch.tensor(label, dtype=torch.long)


# --- 预测与分析核心函数 ---

def optimize_threshold(model, val_loader, device):
    """在验证集上优化分类阈值"""
    print("\nStep 1: 在验证集上优化分类阈值...")
    model.eval()
    all_probs, all_labels = [], []
    with torch.no_grad():
        for data, target in tqdm(val_loader, desc="优化阈值"):
            data = data.to(device)
            output = model(data)
            prob = F.softmax(output, dim=1)
            all_probs.extend(prob[:, 1].cpu().numpy())
            all_labels.extend(target.cpu().numpy())
    
    best_f1, best_threshold = 0, 0.5
    thresholds = np.arange(0.3, 0.8, 0.01)
    
    for threshold in thresholds:
        preds = (np.array(all_probs) >= threshold).astype(int)
        f1 = f1_score(all_labels, preds, average='macro')
        if f1 > best_f1:
            best_f1, best_threshold = f1, threshold
            
    print(f"完成! 最优阈值: {best_threshold:.3f}, 对应F1分数: {best_f1:.4f}")
    return best_threshold

def evaluate_and_analyze(
    model, dataloader, original_texts, device, threshold, split_name
) -> Tuple[Dict, List]:
    """
    在给定的数据集上进行全面评估。
    
    功能:
    1. 计算 Accuracy, F1 score, TP, TN, FP, FN。
    2. 收集所有错误分类的样本及其详细信息。
    
    返回:
    - 一个包含各项性能指标的字典。
    - 一个包含所有错误分类样本信息的列表。
    """
    model.eval()
    misclassified_samples = []
    all_preds, all_labels = [], []
    
    with torch.no_grad():
        for i, (data, target) in enumerate(tqdm(dataloader, desc=f"在 {split_name} 上评估与分析")):
            data, target = data.to(device), target.to(device)
            output = model(data)
            probs = F.softmax(output, dim=1)
            preds = (probs[:, 1] >= threshold).long()
            
            all_labels.extend(target.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            
            misclassified_indices = (preds != target).nonzero(as_tuple=False).squeeze(1)
            
            if misclassified_indices.numel() > 0:
                for idx_in_batch in misclassified_indices:
                    global_idx = i * dataloader.batch_size + idx_in_batch.item()
                    
                    if global_idx < len(original_texts):
                        true_label = target[idx_in_batch].item()
                        pred_label = preds[idx_in_batch].item()
                        confidence_on_true_class = probs[idx_in_batch, true_label].item()
                        
                        sample_info = {
                            "original_text": original_texts[global_idx],
                            "dataset_split": split_name,
                            "true_label": true_label,
                            "predicted_label": pred_label,
                            "confidence_on_true_class": f"{confidence_on_true_class:.4f}"
                        }
                        misclassified_samples.append(sample_info)

    # 修改点 2: 在此处计算所有指标，包括 accuracy
    tn, fp, fn, tp = confusion_matrix(all_labels, all_preds, labels=[0, 1]).ravel()
    
    metrics = {
        "accuracy": accuracy_score(all_labels, all_preds),
        "f1_score": f1_score(all_labels, all_preds, average='macro'),
        "TP": tp,
        "TN": tn,
        "FP": fp,
        "FN": fn
    }
    
    print(f"{split_name} 中找到 {len(misclassified_samples)} 个错误分类样本。")
    return metrics, misclassified_samples


# 修改点 3: 更新打印函数以包含 accuracy
def print_metrics(split_name: str, metrics: Dict):
    """格式化打印性能指标"""
    print(f"\n--- {split_name} 上的性能指标 ---")
    print(f"Accuracy:         {metrics['accuracy']:.4f}")
    print(f"F1 Score (Macro): {metrics['f1_score']:.4f}")
    print(f"混淆矩阵:")
    print(f"  - True Positives (TP):  {metrics['TP']:<5} (正样本被正确预测为正)")
    print(f"  - True Negatives (TN):  {metrics['TN']:<5} (负样本被正确预测为负)")
    print(f"  - False Positives (FP): {metrics['FP']:<5} (负样本被错误预测为正) '误报'")
    print(f"  - False Negatives (FN): {metrics['FN']:<5} (正样本被错误预测为负) '漏报'")
    print("---------------------------------")
    acc_T=(metrics['TP'])/(metrics['TP']+metrics['FN'])
    acc_F=(metrics['TN'])/(metrics['TN']+metrics['FP'])
    print(f"正样本准确率：{acc_T}")
    print(f"负样本准确率：{acc_F}")


def main():
    """主执行函数"""
    # --- 配置参数 ---
    MODEL_PATH = 'best_enhanced_model.pth'
    VOCAB_PATH = 'enhanced_vocab.json'
    DATA_PATH = 'ChnSentiCorp_htl_all.csv'
    OUTPUT_FN_CSV = 'misclassified_FN.csv'
    OUTPUT_FP_CSV = 'misclassified_FP.csv'

    # --- 模型和数据加载参数 ---
    D_MODEL = 512
    N_HEAD = 8
    NUM_LAYERS = 6
    MAX_LEN = 400
    DATASET_MAX_LEN = 128
    
    # ... (从加载模型到创建 DataLoader 的代码保持不变) ...
    print("--- 正在加载模型和数据 ---")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")

    with open(VOCAB_PATH, 'r', encoding='utf-8') as f:
        vocab = json.load(f)
    
    model = EnhancedTransformerEncoder(
        vocab_size=len(vocab), d_model=D_MODEL, nhead=N_HEAD,
        num_layers=NUM_LAYERS, max_len=MAX_LEN, dropout=0.1
    ).to(device)
    
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    print("模型和词汇表加载成功。")

    print("\n--- 正在重新划分数据集以确保一致性 ---")
    df = pd.read_csv(DATA_PATH)
    df['review'] = df['review'].apply(advanced_text_cleaning)
    neg_df = df[df['label'] == 0].drop_duplicates(subset=['review'])
    pos_df = df[df['label'] == 1]
    df = pd.concat([neg_df, pos_df], ignore_index=True)
    
    texts = df['review'].tolist()
    labels = df['label'].tolist()

    X_train, X_temp, y_train, y_temp = train_test_split(
        texts, labels, test_size=0.2, stratify=labels, random_state=42
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=42
    )

    train_dataset = EnhancedTextDataset(X_train, y_train, vocab, max_length=DATASET_MAX_LEN, augment_negative=False)
    val_dataset = EnhancedTextDataset(X_val, y_val, vocab, max_length=DATASET_MAX_LEN, augment_negative=False)
    test_dataset = EnhancedTextDataset(X_test, y_test, vocab, max_length=DATASET_MAX_LEN, augment_negative=False)

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=False)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
    print("数据集创建完成。")
    print(f"训练集: {len(X_train)}, 验证集: {len(X_val)}, 测试集: {len(X_test)}")

    # 4. 优化阈值
    best_threshold = optimize_threshold(model, val_loader, device)

    # 5. 评估并打印结果
    print("\nStep 2: 在所有数据子集上评估性能并收集错误样本...")
    
    train_metrics, misclassified_train = evaluate_and_analyze(model, train_loader, X_train, device, best_threshold, '训练集')
    val_metrics, misclassified_val = evaluate_and_analyze(model, val_loader, X_val, device, best_threshold, '验证集')
    test_metrics, misclassified_test = evaluate_and_analyze(model, test_loader, X_test, device, best_threshold, '测试集')

    print_metrics('训练集', train_metrics)
    print_metrics('验证集', val_metrics)
    print_metrics('测试集', test_metrics)
    
    # 6. 保存错误样本 (逻辑不变)
    all_misclassified = misclassified_train + misclassified_val + misclassified_test
    
    if not all_misclassified:
        print("\n恭喜！模型在所有数据集上没有发现任何错误分类的样本。")
        return

    print("\nStep 3: 将错误样本分类并保存到CSV文件...")
    df_misclassified = pd.DataFrame(all_misclassified)
    
    df_fn = df_misclassified[df_misclassified['true_label'] == 1].copy()
    df_fn.rename(columns={'confidence_on_true_class': 'confidence_on_positive'}, inplace=True)
    
    df_fp = df_misclassified[df_misclassified['true_label'] == 0].copy()
    df_fp.rename(columns={'confidence_on_true_class': 'confidence_on_negative'}, inplace=True)

    if not df_fn.empty:
        df_fn.to_csv(OUTPUT_FN_CSV, index=False, encoding='utf-8-sig')
        print(f"成功保存 {len(df_fn)} 个“正面->负面”(FN)的错误样本到 {OUTPUT_FN_CSV}")
    else:
        print("模型没有将任何正面样本错误预测为负面。")

    if not df_fp.empty:
        df_fp.to_csv(OUTPUT_FP_CSV, index=False, encoding='utf-8-sig')
        print(f"成功保存 {len(df_fp)} 个“负面->正面”(FP)的错误样本到 {OUTPUT_FP_CSV}")
    else:
        print("模型没有将任何负面样本错误预测为正面。")
        
    print("\n分析完成！")


if __name__ == "__main__":
    main()