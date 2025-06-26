import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torch.nn.utils.rnn import pad_sequence
import pandas as pd
import numpy as np
import jieba
import re
import random
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import warnings
import math
import json
from typing import List

warnings.filterwarnings('ignore')
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)

class AdvancedTextAugmentation:
    """高级文本增强类，专门针对负面样本进行多样化增强"""
    def __init__(self):
        self.synonym_dict = {
            '好': ['棒', '不错', '优秀', '很好', '赞', '给力'],
            '差': ['糟糕', '不好', '很差', '糟', '烂', '垃圾'],
            '喜欢': ['爱', '钟爱', '偏爱', '中意'],
            '讨厌': ['厌恶', '反感', '不喜欢', '恶心'],
            '满意': ['满足', '认可', '赞同', '称心'],
            '失望': ['沮丧', '不满', '遗憾', '郁闷'],
            '服务': ['态度', '接待', '待遇'],
            '环境': ['氛围', '条件', '场所'],
            '价格': ['费用', '收费', '价位']
        }
        self.negative_enhancers = ['非常', '特别', '极其', '相当', '十分', '很', '超级', '巨']
        self.positive_words = ['好', '棒', '不错', '满意', '喜欢', '推荐']
        self.negative_words = ['差', '糟', '不好', '失望', '讨厌', '垃圾', '恶心', '糟糕', '烂', '坑', '骗人', '后悔']
    
    def synonym_replacement(self, text: str, prob: float = 0.3) -> str:
        """同义词替换"""
        words = list(jieba.cut(text))
        new_words = []
        for word in words:
            if random.random() < prob and word in self.synonym_dict:
                new_words.append(random.choice(self.synonym_dict[word]))
            else:
                new_words.append(word)
        return ''.join(new_words)
    
    def enhance_sentiment_words(self, text: str, sentiment: str) -> str:
        """增强情感词汇"""
        if sentiment == 'negative':
            for neg_word in self.negative_words:
                if neg_word in text and random.random() < 0.4:
                    enhancer = random.choice(self.negative_enhancers)
                    text = text.replace(neg_word, f'{enhancer}{neg_word}', 1)
        return text
    
    def random_deletion(self, text: str, prob: float = 0.1) -> str:
        """随机删除词汇"""
        words = list(jieba.cut(text))
        if len(words) <= 3:
            return text
        new_words = []
        for word in words:
            if random.random() > prob:
                new_words.append(word)
        return ''.join(new_words) if new_words else text
    
    def random_swap(self, text: str, n: int = 1) -> str:
        """随机交换词汇位置"""
        words = list(jieba.cut(text))
        if len(words) < 2:
            return text
        for _ in range(n):
            idx1, idx2 = random.sample(range(len(words)), 2)
            words[idx1], words[idx2] = words[idx2], words[idx1]
        return ''.join(words)
    
    def insert_negation(self, text: str) -> str:
        """插入否定词"""
        words = list(jieba.cut(text))
        if len(words) > 2:
            # 在正面词汇前插入否定词
            for i, word in enumerate(words):
                if word in self.positive_words and random.random() < 0.3:
                    words.insert(i, '不')
                    break
        return ''.join(words)
    
    def tone_adjustment(self, text: str) -> str:
        """语调调整"""
        # 添加语气词增强负面情感
        tone_words = ['真的', '实在', '确实', '简直']
        words = list(jieba.cut(text))
        if len(words) > 1 and random.random() < 0.3:
            insert_pos = random.randint(0, len(words)//2)
            words.insert(insert_pos, random.choice(tone_words))
        return ''.join(words)
    
    def augment_negative_samples(self, text: str, num_aug: int = 3) -> List[str]:
        """负面样本增强主函数"""
        augmented = []
        
        # 应用不同的增强策略
        augmented.append(self.synonym_replacement(text, 0.3))
        augmented.append(self.enhance_sentiment_words(text, 'negative'))
        augmented.append(self.random_deletion(text, 0.1))
        augmented.append(self.random_swap(text, 1))
        augmented.append(self.insert_negation(text))
        augmented.append(self.tone_adjustment(text))
        
        # 组合增强
        combined = self.enhance_sentiment_words(
            self.synonym_replacement(text, 0.2), 'negative'
        )
        augmented.append(combined)
        
        # 返回指定数量的增强样本
        return random.sample(augmented, min(num_aug, len(augmented)))

class AdaptiveFocalLoss(nn.Module):
    """自适应Focal Loss，动态调整alpha参数"""
    def __init__(self, gamma=2.0, class_weights=None, label_smoothing=0.1):
        super().__init__()
        self.gamma = gamma
        self.class_weights = class_weights
        self.label_smoothing = label_smoothing
    
    def forward(self, outputs, targets):
        device = outputs.device
        class_weights = self.class_weights.to(device) if self.class_weights is not None else None
        
        # 计算基础交叉熵损失
        ce_loss = F.cross_entropy(outputs, targets, weight=class_weights, reduction='none')
        pt = torch.exp(-ce_loss)
        
        # 动态调整alpha
        probs = F.softmax(outputs, dim=1)
        target_probs = probs.gather(1, targets.unsqueeze(1)).squeeze(1)
        
        # 根据预测置信度动态调整alpha
        alpha = torch.where(targets == 0, 0.75, 0.25)  # 负类更高权重
        alpha = alpha * (1 - target_probs)  # 困难样本获得更高权重
        
        # Focal Loss
        focal_loss = alpha * (1 - pt) ** self.gamma * ce_loss
        
        # Label Smoothing
        smooth_loss = F.cross_entropy(outputs, targets, weight=class_weights, 
                                    label_smoothing=self.label_smoothing)
        
        return 0.8 * focal_loss.mean() + 0.2 * smooth_loss

class ResidualBlock(nn.Module):
    """残差连接块"""
    def __init__(self, dim, dropout=0.1):
        super().__init__()
        self.linear1 = nn.Linear(dim, dim)
        self.linear2 = nn.Linear(dim, dim)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.GELU()
    
    def forward(self, x):
        residual = x
        x = self.norm1(x)
        x = self.linear1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.linear2(x)
        x = self.dropout(x)
        return x + residual

class EnhancedTransformerEncoder(nn.Module):
    """增强的Transformer编码器"""
    def __init__(self, vocab_size, d_model=256, nhead=8, num_layers=4, max_len=512, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.max_length = max_len
        
        # 嵌入层
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.pos_embedding = nn.Embedding(max_len, d_model)
        
        # Transformer编码器
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,  # 增加前馈网络维度
            dropout=dropout,
            activation='gelu',
            norm_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # 多头注意力池化
        self.attention_pooling = nn.MultiheadAttention(d_model, nhead//2, batch_first=True)
        
        # 类别感知注意力
        self.class_attention = nn.MultiheadAttention(d_model, nhead//2, dropout=0.1)
        self.class_embedding = nn.Embedding(2, d_model)
        
        # 增强的分类头 - 使用残差连接
        self.feature_projection = nn.Linear(d_model * 2, d_model)
        
        # 负面分类分支
        self.negative_branch = nn.Sequential(
            ResidualBlock(d_model, dropout=0.2),
            nn.Linear(d_model, d_model//2),
            nn.LayerNorm(d_model//2),
            nn.GELU(),
            nn.Dropout(0.3),
            ResidualBlock(d_model//2, dropout=0.2),
            nn.Linear(d_model//2, 1)
        )
        
        # 正面分类分支
        self.positive_branch = nn.Sequential(
            ResidualBlock(d_model, dropout=0.2),
            nn.Linear(d_model, d_model//2),
            nn.LayerNorm(d_model//2),
            nn.GELU(),
            nn.Dropout(0.3),
            ResidualBlock(d_model//2, dropout=0.2),
            nn.Linear(d_model//2, 1)
        )
        
        # 全局特征提取器
        self.global_feature = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(0.2)
        )
        
        self._init_weights()
    
    def _init_weights(self):
        """权重初始化"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0, std=0.02)
    
    def forward(self, x):
        batch_size, seq_len = x.size()
        
        # 位置编码
        positions = torch.arange(seq_len, device=x.device).unsqueeze(0).expand(batch_size, -1)
        x = self.embedding(x) + self.pos_embedding(positions)
        
        # 创建padding mask
        padding_mask = (x.sum(dim=-1) == 0)
        
        # Transformer编码
        x = x.transpose(0, 1)  # (seq_len, batch_size, d_model)
        encoded = self.transformer(x, src_key_padding_mask=padding_mask)
        encoded = encoded.transpose(0, 1)  # (batch_size, seq_len, d_model)
        
        # 注意力池化
        query = encoded.mean(dim=1, keepdim=True)
        pooled, attention_weights = self.attention_pooling(
            query, encoded, encoded, key_padding_mask=padding_mask
        )
        pooled = pooled.squeeze(1)
        
        # 类别感知特征
        class_embeds = self.class_embedding(torch.tensor([0, 1], device=x.device))
        class_embeds = class_embeds.mean(dim=0, keepdim=True).repeat(pooled.size(0), 1)
        
        # 特征融合
        combined_features = torch.cat([pooled, class_embeds], dim=-1)
        global_features = self.global_feature(combined_features)
        
        # 分别计算正负面分数
        neg_logit = self.negative_branch(global_features)
        pos_logit = self.positive_branch(global_features)
        
        return torch.cat([neg_logit, pos_logit], dim=1)

class EnhancedTextDataset(Dataset):
    """增强的文本数据集"""
    def __init__(self, texts, labels, vocab, max_length=128, augment_negative=True, augment_ratio=1.5):
        self.texts = texts
        self.labels = labels
        self.vocab = vocab
        self.max_length = max_length
        self.augment_negative = augment_negative
        self.augment_ratio = augment_ratio
        self.augmenter = AdvancedTextAugmentation()
        
        if augment_negative:
            self._augment_data()
    
    def _augment_data(self):
        """数据增强"""
        augmented_texts = []
        augmented_labels = []
        
        negative_count = sum(1 for label in self.labels if label == 0)
        positive_count = sum(1 for label in self.labels if label == 1)
        
        # 计算需要增强的负面样本数量
        target_negative = int(positive_count / self.augment_ratio)
        need_augment = max(0, target_negative - negative_count)
        
        negative_indices = [i for i, label in enumerate(self.labels) if label == 0]
        
        print(f"原始负面样本: {negative_count}, 正面样本: {positive_count}")
        print(f"需要增强负面样本: {need_augment}")
        
        # 增强负面样本
        for _ in range(need_augment):
            idx = random.choice(negative_indices)
            original_text = self.texts[idx]
            aug_texts = self.augmenter.augment_negative_samples(original_text, 2)
            
            for aug in aug_texts:
                if len(aug.strip()) > 3:  # 确保增强后的文本有意义
                    augmented_texts.append(aug)
                    augmented_labels.append(0)
        
        self.texts.extend(augmented_texts)
        self.labels.extend(augmented_labels)
        
        print(f"增强后总样本: {len(self.texts)}, 负面: {sum(1 for l in self.labels if l == 0)}, 正面: {sum(1 for l in self.labels if l == 1)}")
    
    def advanced_text_cleaning(self, text):
        """高级文本清理"""
        if pd.isna(text):
            return ""
        
        text = str(text)
        
        # 标准化标点符号
        text = re.sub(r'[！!]{2,}', '!', text)
        text = re.sub(r'[？?]{2,}', '?', text)
        text = re.sub(r'[。.]{2,}', '。', text)
        
        # 处理重复字符
        text = re.sub(r'(.)\1{2,}', r'\1\1', text)
        
        # 保留情感相关的符号
        text = re.sub(r'[^\u4e00-\u9fa5a-zA-Z0-9\s！？。，、；：""''（）!?.,;:""''()]', '', text)
        
        # 清理多余空格
        text = re.sub(r'\s+', '', text)
        
        return text.strip()
    
    def text_to_indices(self, text):
        """文本转索引"""
        text = self.advanced_text_cleaning(text)
        words = list(jieba.cut(text))
        indices = [self.vocab.get(word, self.vocab['<UNK>']) for word in words]
        
        if len(indices) > self.max_length:
            indices = indices[:self.max_length]
        else:
            indices.extend([self.vocab['<PAD>']] * (self.max_length - len(indices)))
        
        return torch.tensor(indices, dtype=torch.long)
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        
        # 训练时随机应用增强
        if self.augment_negative and random.random() < 0.15:
            if label == 0:  # 只对负面样本进行实时增强
                aug_texts = self.augmenter.augment_negative_samples(text, 1)
                if aug_texts:
                    text = aug_texts[0]
        
        indices = self.text_to_indices(text)
        return indices, torch.tensor(label, dtype=torch.long)

class ProgressiveTrainer:
    """渐进式训练器"""
    def __init__(self, model, device, class_weights=None):
        self.model = model
        self.device = device
        self.class_weights = class_weights
        
        # 使用自适应Focal Loss
        self.criterion = AdaptiveFocalLoss(
            gamma=2.0, 
            class_weights=class_weights, 
            label_smoothing=0.1
        )
        
        self.best_f1 = 0
        self.patience_counter = 0
        self.training_history = {
            'train_losses': [],
            'val_accuracies': [],
            'val_f1_scores': [],
            'learning_rates': []
        }
    
    def progressive_unfreezing(self, epoch, total_epochs):
        """渐进式解冻"""
        if epoch < total_epochs * 0.3:
            # 前30%的epoch只训练分类头
            for param in self.model.embedding.parameters():
                param.requires_grad = False
            for param in self.model.transformer.parameters():
                param.requires_grad = False
        elif epoch < total_epochs * 0.6:
            # 中间30%的epoch解冻部分transformer层
            for param in self.model.embedding.parameters():
                param.requires_grad = True
            for i, layer in enumerate(self.model.transformer.layers):
                if i >= len(self.model.transformer.layers) // 2:
                    for param in layer.parameters():
                        param.requires_grad = True
        else:
            # 最后40%的epoch全部解冻
            for param in self.model.parameters():
                param.requires_grad = True
    
    def train_phase1(self, train_loader, val_loader, epochs=8, lr=1e-3):
        """阶段1：预训练分类头"""
        print("\n=== 阶段1：预训练分类头 ===")
        
        # 冻结主要参数
        for param in self.model.embedding.parameters():
            param.requires_grad = False
        for param in self.model.pos_embedding.parameters():
            param.requires_grad = False
        for param in self.model.transformer.parameters():
            param.requires_grad = False
        for param in self.model.attention_pooling.parameters():
            param.requires_grad = False
        
        optimizer = optim.AdamW(
            filter(lambda p: p.requires_grad, self.model.parameters()), 
            lr=lr, weight_decay=0.01
        )
        
        for epoch in range(epochs):
            self.model.train()
            total_loss = 0
            
            for batch_idx, (data, target) in enumerate(tqdm(train_loader, desc=f"Phase1 Epoch {epoch+1}")):
                data, target = data.to(self.device), target.to(self.device)
                
                optimizer.zero_grad()
                output = self.model(data)
                loss = self.criterion(output, target)
                loss.backward()
                
                # 梯度裁剪
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                optimizer.step()
                
                total_loss += loss.item()
            
            val_acc, val_f1 = self.evaluate(val_loader)
            avg_loss = total_loss / len(train_loader)
            
            print(f"Phase1 Epoch {epoch+1}: Loss={avg_loss:.4f}, Val_Acc={val_acc:.4f}, Val_F1={val_f1:.4f}")
    
    def train_phase2(self, train_loader, val_loader, epochs=30, lr=1e-4, patience=10):
        """阶段2：微调全模型"""
        print("\n=== 阶段2：微调全模型 ===")
        
        # 解冻所有参数
        for param in self.model.parameters():
            param.requires_grad = True
        
        optimizer = optim.AdamW(self.model.parameters(), lr=lr, weight_decay=0.02)
        
        # 使用余弦退火学习率调度
        scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=10, T_mult=2, eta_min=1e-6
        )
        
        for epoch in range(epochs):
            # 渐进式解冻
            self.progressive_unfreezing(epoch, epochs)
            
            self.model.train()
            total_loss = 0
            
            for batch_idx, (data, target) in enumerate(tqdm(train_loader, desc=f"Phase2 Epoch {epoch+1}")):
                data, target = data.to(self.device), target.to(self.device)
                
                optimizer.zero_grad()
                output = self.model(data)
                loss = self.criterion(output, target)
                loss.backward()
                
                # 梯度裁剪
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                optimizer.step()
                
                total_loss += loss.item()
            
            scheduler.step()
            
            val_acc, val_f1 = self.evaluate(val_loader)
            avg_loss = total_loss / len(train_loader)
            current_lr = optimizer.param_groups[0]['lr']
            
            # 记录训练历史
            self.training_history['train_losses'].append(avg_loss)
            self.training_history['val_accuracies'].append(val_acc)
            self.training_history['val_f1_scores'].append(val_f1)
            self.training_history['learning_rates'].append(current_lr)
            
            print(f"Phase2 Epoch {epoch+1}: Loss={avg_loss:.4f}, Val_Acc={val_acc:.4f}, Val_F1={val_f1:.4f}, LR={current_lr:.2e}")
            
            # 早停机制
            if val_f1 > self.best_f1:
                self.best_f1 = val_f1
                self.patience_counter = 0
                torch.save(self.model.state_dict(), 'best_enhanced_model.pth')
                print(f"新的最佳F1分数: {self.best_f1:.4f}")
            else:
                self.patience_counter += 1
                if self.patience_counter >= patience:
                    print(f"早停触发，最佳F1分数: {self.best_f1:.4f}")
                    break
        
        return self.training_history
    
    def evaluate(self, data_loader, threshold=0.5):
        """模型评估"""
        self.model.eval()
        all_preds = []
        all_labels = []
        all_probs = []
        
        with torch.no_grad():
            for data, target in data_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                prob = F.softmax(output, dim=1)
                pred = (prob[:, 1] >= threshold).long()
                
                all_preds.extend(pred.cpu().numpy())
                all_labels.extend(target.cpu().numpy())
                all_probs.extend(prob[:, 1].cpu().numpy())
        
        accuracy = accuracy_score(all_labels, all_preds)
        f1 = f1_score(all_labels, all_preds, average='macro')
        
        return accuracy, f1
    
    def optimize_threshold(self, val_loader):
        """优化分类阈值"""
        print("\n=== 优化分类阈值 ===")
        self.model.eval()
        all_probs, all_labels = [], []
        
        with torch.no_grad():
            for data, target in val_loader:
                data = data.to(self.device)
                output = self.model(data)
                prob = F.softmax(output, dim=1)
                all_probs.extend(prob[:, 1].cpu().numpy())
                all_labels.extend(target.cpu().numpy())
        
        best_f1, best_threshold = 0, 0.5
        thresholds = np.arange(0.3, 0.8, 0.02)
        
        for threshold in thresholds:
            preds = (np.array(all_probs) >= threshold).astype(int)
            f1 = f1_score(all_labels, preds, average='macro')
            if f1 > best_f1:
                best_f1, best_threshold = f1, threshold
        
        print(f"最优阈值: {best_threshold:.3f}, 对应F1: {best_f1:.4f}")
        return best_threshold

def advanced_text_cleaning(text):
    """高级文本清理函数"""
    if pd.isna(text):
        return ""
    
    text = str(text)
    
    # 标准化标点符号
    text = re.sub(r'[！!]{2,}', '!', text)
    text = re.sub(r'[？?]{2,}', '?', text)
    text = re.sub(r'[。.]{2,}', '。', text)
    
    # 处理重复字符
    text = re.sub(r'(.)\1{3,}', r'\1\1', text)
    
    # 保留情感相关的符号和表情
    text = re.sub(r'[^\u4e00-\u9fa5a-zA-Z0-9\s！？。，、；：""''（）!?.,;:""''()]', '', text)
    
    # 清理多余空格
    text = re.sub(r'\s+', '', text)
    
    return text.strip()

def build_enhanced_vocab(texts, min_freq=2, max_vocab_size=12000):
    """构建增强词表"""
    word_freq = Counter()
    
    for text in texts:
        # 使用清理后的文本构建词表
        cleaned_text = advanced_text_cleaning(text)
        word_freq.update(jieba.cut(cleaned_text))
    
    # 按频率排序
    sorted_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
    
    # 过滤低频词
    vocab_words = [w for w, f in sorted_words if f >= min_freq and len(w.strip()) > 0][:max_vocab_size-4]
    
    # 构建词表
    vocab = {'<PAD>': 0, '<UNK>': 1, '<CLS>': 2, '<SEP>': 3}
    for idx, word in enumerate(vocab_words, start=4):
        vocab[word] = idx
    
    print(f"词表大小: {len(vocab)}")
    return vocab

def plot_enhanced_training_curves(history):
    """绘制增强的训练曲线"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # 训练损失
    axes[0, 0].plot(history['train_losses'])
    axes[0, 0].set_title('训练损失')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].grid(True)
    
    # 验证准确率
    axes[0, 1].plot(history['val_accuracies'])
    axes[0, 1].set_title('验证准确率')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Accuracy')
    axes[0, 1].grid(True)
    
    # 验证F1分数
    axes[1, 0].plot(history['val_f1_scores'])
    axes[1, 0].set_title('验证F1分数')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('F1 Score')
    axes[1, 0].grid(True)
    
    # 学习率
    axes[1, 1].plot(history['learning_rates'])
    axes[1, 1].set_title('学习率变化')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Learning Rate')
    axes[1, 1].set_yscale('log')
    axes[1, 1].grid(True)
    
    plt.tight_layout()
    plt.savefig('enhanced_training_curves.png', dpi=300, bbox_inches='tight')
    plt.close()

def main():
    print("=== 增强版情感分类模型训练 ===")
    
    # 读取数据
    df = pd.read_csv('ChnSentiCorp_htl_all.csv')
    print(f"原始数据量: {len(df)}")
    
    # 高级文本清理
    df['review'] = df['review'].apply(advanced_text_cleaning)
    
    # 过滤过短的文本
    df = df[df['review'].str.len() > 5].reset_index(drop=True)
    
    # 负面样本质量提升
    neg_df = df[df['label'] == 0]
    neg_df = neg_df.drop_duplicates(subset=['review'])
    
    # # 过滤无意义的负面评论
    # meaningless_patterns = ['差评', '无语', '垃圾', '呵呵', '不评价', '啥说的', '一般般', '还行']
    # for pattern in meaningless_patterns:
    #     neg_df = neg_df[~neg_df['review'].str.contains(pattern, na=False)]
    
    pos_df = df[df['label'] == 1]
    df = pd.concat([neg_df, pos_df], ignore_index=True)
    
    texts = df['review'].tolist()
    labels = df['label'].tolist()
    
    print(f"清理后数据量: {len(texts)}")
    print(f"正面样本: {sum(labels)}, 负面样本: {len(labels) - sum(labels)}")
    print(f"类别比例: {sum(labels) / len(labels):.3f}")
    
    # 构建增强词表
    vocab = build_enhanced_vocab(texts, min_freq=2, max_vocab_size=12000)
    
    # 数据集划分
    X_train, X_temp, y_train, y_temp = train_test_split(
        texts, labels, test_size=0.2, stratify=labels, random_state=42
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=42
    )
    
    print(f"训练集: {len(X_train)}, 验证集: {len(X_val)}, 测试集: {len(X_test)}")
    
    # 创建数据集
    train_dataset = EnhancedTextDataset(
        X_train, y_train, vocab, 
        augment_negative=True, augment_ratio=1.2
    )
    val_dataset = EnhancedTextDataset(X_val, y_val, vocab, augment_negative=False)
    test_dataset = EnhancedTextDataset(X_test, y_test, vocab, augment_negative=False)
    
    # 创建数据加载器
    # 调整采样权重
    sample_weights = [2.5 if l == 0 else 1.0 for l in train_dataset.labels]
    sampler = WeightedRandomSampler(
        sample_weights, num_samples=len(sample_weights), replacement=True
    )
    
    train_loader = DataLoader(train_dataset, batch_size=32, sampler=sampler, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=64)
    test_loader = DataLoader(test_dataset, batch_size=64)
    
    # 计算类别权重
    neg_count = sum(1 for l in train_dataset.labels if l == 0)
    pos_count = sum(1 for l in train_dataset.labels if l == 1)
    total = neg_count + pos_count
    
    class_weights = torch.tensor([
        total / (2 * neg_count),  # 负类权重
        total / (2 * pos_count)   # 正类权重
    ], dtype=torch.float)
    
    print(f"类别权重: 负类={class_weights[0]:.3f}, 正类={class_weights[1]:.3f}")
    
    # 创建模型
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    model = EnhancedTransformerEncoder(
        vocab_size=len(vocab),
        d_model=512,
        nhead=8,
        num_layers=6,  # 增加层数
        max_len=200,
        dropout=0.1
    ).to(device)
    
    # 创建训练器
    trainer = ProgressiveTrainer(model, device, class_weights=class_weights)
    
    # 阶段1：预训练分类头
    trainer.train_phase1(train_loader, val_loader, epochs=8, lr=1e-3)
    
    # 阶段2：微调全模型
    history = trainer.train_phase2(train_loader, val_loader, epochs=30, lr=1e-4, patience=12)
    
    # 绘制训练曲线
    plot_enhanced_training_curves(history)
    
    # 优化阈值
    best_threshold = trainer.optimize_threshold(val_loader)
    
    # 加载最佳模型进行测试
    model.load_state_dict(torch.load('best_enhanced_model.pth', map_location=device))
    
    # 最终测试
    print("\n=== 最终测试结果 ===")
    model.eval()
    all_preds, all_labels, all_probs = [], [], []
    
    with torch.no_grad():
        for data, target in test_loader:
            data = data.to(device)
            output = model(data)
            prob = F.softmax(output, dim=1)
            pred = (prob[:, 1] >= best_threshold).long()
            
            all_preds.extend(pred.cpu().numpy())
            all_labels.extend(target.numpy())
            all_probs.extend(prob[:, 1].cpu().numpy())
    
    # 计算指标
    acc = accuracy_score(all_labels, all_preds)
    f1_macro = f1_score(all_labels, all_preds, average='macro')
    f1_weighted = f1_score(all_labels, all_preds, average='weighted')
    
    print(f"测试集准确率: {acc:.4f}")
    print(f"宏平均F1: {f1_macro:.4f}")
    print(f"加权平均F1: {f1_weighted:.4f}")
    print(f"使用阈值: {best_threshold:.3f}")
    
    # 详细分类报告
    print("\n详细分类报告:")
    print(classification_report(all_labels, all_preds, digits=4, 
                              target_names=['负面', '正面']))
    
    # 绘制混淆矩阵
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['负面', '正面'], yticklabels=['负面', '正面'])
    plt.xlabel('预测标签')
    plt.ylabel('真实标签')
    plt.title('增强版情感分类混淆矩阵')
    plt.savefig('enhanced_confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 保存模型和词表
    torch.save(model.state_dict(), 'final_enhanced_model.pth')
    with open('enhanced_vocab.json', 'w', encoding='utf-8') as f:
        json.dump(vocab, f, ensure_ascii=False, indent=2)
    
    print("\n训练完成！模型和词表已保存。")
    print(f"最佳模型: best_enhanced_model.pth")
    print(f"最终模型: final_enhanced_model.pth")
    print(f"词表文件: enhanced_vocab.json")

if __name__ == "__main__":
    main()