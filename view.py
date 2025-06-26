# analyze_all.py

import pandas as pd
import jieba
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import os
import warnings
from gensim.models import Word2Vec
from sklearn.manifold import TSNE

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

def visualize_hotel_reviews(file_path='ChnSentiCorp_htl_all.csv'):
    """
    Loads, processes, and creates a full suite of visualizations 
    for the Chinese hotel review dataset.
    
    All generated images will be saved in a new folder named 'view'.
    """
    # --- 0. Setup Environment ---
    print("--- Starting Full Analysis Pipeline ---")
    output_folder = 'view'
    os.makedirs(output_folder, exist_ok=True)
    print(f"All images will be saved to the '{output_folder}/' folder.")
    
    # !!! IMPORTANT !!!
    # You MUST specify a font path that exists on your system and supports Chinese.
    # Examples:
    # Windows: font_path = 'C:/Windows/Fonts/msyh.ttc' (Microsoft YaHei)
    # macOS:   font_path = '/System/Library/Fonts/PingFang.ttc'
    # Linux:   Find a font like 'wqy-zenhei.ttc' or 'NotoSansCJK-Regular.ttc'
    font_path = 'msyh.ttc' # <-- CHANGE THIS PATH IF NEEDED
    
    # Set a Chinese font for all matplotlib plots
    try:
        plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'PingFang TC', 'WenQuanYi Zen Hei']
        plt.rcParams['axes.unicode_minus'] = False
    except:
        print("Could not set Chinese font for plots. Labels may not display correctly.")

    # --- 1. Load Data ---
    print("\n[1/6] Loading data...")
    df = pd.read_csv(file_path)
    df.dropna(subset=['review'], inplace=True)
    df['label_desc'] = df['label'].apply(lambda x: 'Positive' if x == 1 else 'Negative')

    # --- 2. Basic Statistics Visualizations ---
    print("[2/6] Generating basic statistics plots...")
    
    # Label Distribution
    plt.figure(figsize=(8, 6))
    ax = sns.countplot(x='label_desc', data=df, order=['Positive', 'Negative'], palette=['#2ca02c', '#d62728'])
    ax.set_title('Distribution of Hotel Reviews', fontsize=16)
    for p in ax.patches:
        ax.annotate(f'{p.get_height()}', (p.get_x() + p.get_width() / 2., p.get_height()), ha='center', va='center', xytext=(0, 5), textcoords='offset points')
    plt.savefig(os.path.join(output_folder, "1_review_distribution.png"))
    
    # Review Length Distribution
    df['review_length'] = df['review'].str.len()
    plt.figure(figsize=(10, 7))
    sns.boxplot(x='label_desc', y='review_length', data=df, order=['Positive', 'Negative'], palette=['#2ca02c', '#d62728'])
    plt.title('Review Length Distribution by Sentiment', fontsize=16)
    plt.ylabel('Review Length (Characters)')
    plt.ylim(0, 600)
    plt.savefig(os.path.join(output_folder, "2_review_length_distribution.png"))
    print("   - Saved review distribution and length plots.")

    # --- 3. Text Processing ---
    print("[3/6] Processing text with jieba...")
    stopwords = set([
        "的", "了", "我", "是", "在", "也", "都", "就", "和", "跟", "我们", "你", "他", "她", "它", "但",
        "一个", "没有", "还", "有", "很", "个", "不", "吧", "被", "给", "说", "到", "去", "会", "着",
        "对", "得", "而", "能", "要", "多", "好", "来", "又", "为", "这个", "那个", "那", "这",
        "什么", "还有", "但是", "就是", "酒店", "房间", "饭店", "宾馆", "不错"
    ])

    def process_text(text):
        seg_list = jieba.cut(text, cut_all=False)
        return [word for word in seg_list if word not in stopwords and len(word) > 1]

    df['tokens'] = df['review'].apply(process_text)
    
    all_words = sum(df['tokens'].tolist(), [])
    positive_words = sum(df[df['label'] == 1]['tokens'].tolist(), [])
    negative_words = sum(df[df['label'] == 0]['tokens'].tolist(), [])

    # --- 4. Word Cloud Visualizations ---
    print("[4/6] Generating word clouds...")
    try:
        # Overall
        wc_all = WordCloud(font_path=font_path, width=1000, height=700, background_color='white').generate(" ".join(all_words))
        plt.figure(figsize=(10, 8)); plt.imshow(wc_all, interpolation='bilinear'); plt.axis("off"); plt.title("Overall Word Cloud", fontsize=16)
        plt.savefig(os.path.join(output_folder, "3a_wordcloud_overall.png"))

        # Positive
        wc_pos = WordCloud(font_path=font_path, width=800, height=600, background_color='white').generate(" ".join(positive_words))
        plt.figure(figsize=(10, 8)); plt.imshow(wc_pos, interpolation='bilinear'); plt.axis("off"); plt.title("Positive Reviews Word Cloud", fontsize=16)
        plt.savefig(os.path.join(output_folder, "3b_wordcloud_positive.png"))

        # Negative
        wc_neg = WordCloud(font_path=font_path, width=800, height=600, background_color='black', colormap='Reds').generate(" ".join(negative_words))
        plt.figure(figsize=(10, 8)); plt.imshow(wc_neg, interpolation='bilinear'); plt.axis("off"); plt.title("Negative Reviews Word Cloud", fontsize=16)
        plt.savefig(os.path.join(output_folder, "3c_wordcloud_negative.png"))
        print("   - Saved 3 word cloud images.")
    except Exception as e:
        print(f"   - Could not generate word clouds. Error: {e}. Check font path.")

    # --- 5. Word Frequency Visualizations ---
    print("[5/6] Generating word frequency bar charts...")
    pos_counts = Counter(positive_words).most_common(20)
    df_pos = pd.DataFrame(pos_counts, columns=['word', 'count'])
    
    plt.figure(figsize=(12, 8)); sns.barplot(x='count', y='word', data=df_pos, palette='Greens_r')
    plt.title('Top 20 Words in Positive Reviews', fontsize=16)
    plt.savefig(os.path.join(output_folder, "4a_top_positive_words.png"), bbox_inches='tight')

    neg_counts = Counter(negative_words).most_common(20)
    df_neg = pd.DataFrame(neg_counts, columns=['word', 'count'])
    
    plt.figure(figsize=(12, 8)); sns.barplot(x='count', y='word', data=df_neg, palette='Reds_r')
    plt.title('Top 20 Words in Negative Reviews', fontsize=16)
    plt.savefig(os.path.join(output_folder, "4b_top_negative_words.png"), bbox_inches='tight')
    print("   - Saved top word frequency charts.")

    # --- 6. Word Embedding (t-SNE) Visualization ---
    print("[6/6] Training Word2Vec and generating t-SNE plot...")
    try:
        # Train Word2Vec model
        w2v_model = Word2Vec(sentences=df['tokens'].tolist(), vector_size=100, window=5, min_count=5, workers=4)
        
        # Get vectors for the top N most frequent words
        top_words = [word for word, count in Counter(all_words).most_common(100)]
        vocab_vectors = w2v_model.wv[top_words]
        
        # Reduce dimensionality with t-SNE
        tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(vocab_vectors)-1))
        vectors_2d = tsne.fit_transform(vocab_vectors)
        
        # Create plot
        plt.figure(figsize=(16, 16))
        plt.scatter(vectors_2d[:, 0], vectors_2d[:, 1], c='steelblue', edgecolors='k')
        for i, word in enumerate(top_words):
            plt.annotate(word, (vectors_2d[i, 0], vectors_2d[i, 1]), fontsize=9)
        plt.title('t-SNE Visualization of Word Embeddings (Top 100 Words)', fontsize=18)
        plt.grid(True)
        plt.savefig(os.path.join(output_folder, "5_word_embedding_tsne.png"), bbox_inches='tight')
        print("   - Saved t-SNE visualization plot.")
    except Exception as e:
        print(f"   - Could not generate t-SNE plot. Error: {e}")

    print("\n--- Analysis Complete! ---")


if __name__ == '__main__':
    visualize_hotel_reviews()