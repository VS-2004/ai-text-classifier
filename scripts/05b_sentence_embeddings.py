import pandas as pd
import numpy as np
import os
from sentence_transformers import SentenceTransformer
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
df = pd.read_csv('outputs/figures/processed_features.csv')
df['y'] = (df['label'] == 'ai').astype(int)
struct_cols = ['word_count', 'avg_sentence_length', 'punct_density', 'exclaim_count', 'has_caps_word', 'has_emoji', 'has_casual_marker', 'avg_word_length', 'ttr', 'domain_academic', 'domain_news', 'domain_social']
emb_path = 'outputs/figures/sentence_embeddings.npy'
if os.path.exists(emb_path):
    embeddings = np.load(emb_path)
else:
    model = SentenceTransformer('all-MiniLM-L6-v2')
    embeddings = model.encode(df['text_clean'].tolist(), show_progress_bar=True)
    np.save(emb_path, embeddings)
idx_all = df.index.values
y_all = df['y'].values
gss = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=42)
train_idx, test_idx = next(gss.split(idx_all, y_all, groups=df['template_id'].values))
train_idx = idx_all[train_idx]
test_idx = idx_all[test_idx]
emb_train = embeddings[train_idx]
emb_test = embeddings[test_idx]
scaler = StandardScaler()
struct_train = scaler.fit_transform(df.loc[train_idx, struct_cols])
struct_test = scaler.transform(df.loc[test_idx, struct_cols])
X_train_combined = np.hstack([emb_train, struct_train])
X_test_combined = np.hstack([emb_test, struct_test])
y_train = df.loc[train_idx, 'y'].values
y_test = df.loc[test_idx, 'y'].values
model_combined = LogisticRegression(max_iter=2000, C=10, class_weight='balanced')
model_combined.fit(X_train_combined, y_train)
preds_combined = model_combined.predict(X_test_combined)
model_emb_only = LogisticRegression(max_iter=2000, C=10, class_weight='balanced')
model_emb_only.fit(emb_train, y_train)
preds_emb_only = model_emb_only.predict(emb_test)
results = pd.DataFrame([
{'model': 'Embeddings + Structural (LR)', 'accuracy': accuracy_score(y_test, preds_combined), 'precision_ai': precision_score(y_test, preds_combined), 'recall_ai': recall_score(y_test, preds_combined), 'f1_ai': f1_score(y_test, preds_combined)},
{'model': 'Embeddings only (LR)', 'accuracy': accuracy_score(y_test, preds_emb_only), 'precision_ai': precision_score(y_test, preds_emb_only), 'recall_ai': recall_score(y_test, preds_emb_only), 'f1_ai': f1_score(y_test, preds_emb_only)},
{'model': 'TF-IDF + Structural (Phase 4 baseline)', 'accuracy': 0.8772, 'precision_ai': 1.0, 'recall_ai': 0.6996, 'f1_ai': 0.8232}
])
print(results.round(4).to_string(index=False))
results.to_csv('outputs/figures/phase5_embedding_comparison.csv', index=False)
full_comparison = pd.DataFrame([
{'model': 'TF-IDF only', 'accuracy': 0.5912, 'precision_ai': 0.0, 'recall_ai': 0.0, 'f1_ai': 0.0},
{'model': 'Embeddings only', 'accuracy': accuracy_score(y_test, preds_emb_only), 'precision_ai': precision_score(y_test, preds_emb_only), 'recall_ai': recall_score(y_test, preds_emb_only), 'f1_ai': f1_score(y_test, preds_emb_only)},
{'model': 'Structural only', 'accuracy': 0.6404, 'precision_ai': 0.5654, 'recall_ai': 0.5193, 'f1_ai': 0.5414},
{'model': 'RandomForest (TF-IDF+structural)', 'accuracy': 0.642105, 'precision_ai': 1.0, 'recall_ai': 0.124464, 'f1_ai': 0.221374},
{'model': 'GradientBoosting (TF-IDF+structural)', 'accuracy': 0.721053, 'precision_ai': 1.0, 'recall_ai': 0.317597, 'f1_ai': 0.482085},
{'model': 'Embeddings + Structural (LR)', 'accuracy': accuracy_score(y_test, preds_combined), 'precision_ai': precision_score(y_test, preds_combined), 'recall_ai': recall_score(y_test, preds_combined), 'f1_ai': f1_score(y_test, preds_combined)},
{'model': 'TF-IDF + Structural (LR, tuned)', 'accuracy': 0.8772, 'precision_ai': 1.0, 'recall_ai': 0.6996, 'f1_ai': 0.8232}
])
full_comparison.to_csv('outputs/figures/phase5_full_comparison.csv', index=False)