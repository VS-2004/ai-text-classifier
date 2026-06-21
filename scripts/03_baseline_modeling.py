import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, GroupShuffleSplit
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, confusion_matrix
from scipy.sparse import hstack, csr_matrix
import matplotlib.pyplot as plt
import seaborn as sns
df = pd.read_csv('outputs/figures/processed_features.csv')
df['y'] = (df['label'] == 'ai').astype(int)
struct_cols = ['word_count', 'avg_sentence_length', 'punct_density', 'exclaim_count', 'has_caps_word', 'has_emoji', 'has_casual_marker', 'avg_word_length', 'ttr', 'domain_academic', 'domain_news', 'domain_social']
def build_features(train_idx, test_idx):
    vec = TfidfVectorizer(max_features=3000, ngram_range=(1, 2), min_df=2)
    X_tfidf_train = vec.fit_transform(df.loc[train_idx, 'text_clean'])
    X_tfidf_test = vec.transform(df.loc[test_idx, 'text_clean'])
    scaler = StandardScaler()
    X_struct_train = scaler.fit_transform(df.loc[train_idx, struct_cols])
    X_struct_test = scaler.transform(df.loc[test_idx, struct_cols])
    X_combined_train = hstack([X_tfidf_train, csr_matrix(X_struct_train)])
    X_combined_test = hstack([X_tfidf_test, csr_matrix(X_struct_test)])
    return X_tfidf_train, X_tfidf_test, X_struct_train, X_struct_test, X_combined_train, X_combined_test
def evaluate(model, X_train, y_train, X_test, y_test, name):
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    return {'experiment': name, 'accuracy': accuracy_score(y_test, preds), 'precision_ai': precision_score(y_test, preds, pos_label=1), 'recall_ai': recall_score(y_test, preds, pos_label=1), 'f1_ai': f1_score(y_test, preds, pos_label=1)}, preds
results = []
idx_all = df.index.values
y_all = df['y'].values
train_idx_naive, test_idx_naive = train_test_split(idx_all, test_size=0.25, stratify=y_all, random_state=42)
gss = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=42)
train_idx_grp, test_idx_grp = next(gss.split(idx_all, y_all, groups=df['template_id'].values))
train_idx_grp = idx_all[train_idx_grp]
test_idx_grp = idx_all[test_idx_grp]
overlap = set(df.loc[train_idx_grp, 'template_id']) & set(df.loc[test_idx_grp, 'template_id'])
print('template overlap between grouped train/test:', len(overlap))
splits = {'naive': (train_idx_naive, test_idx_naive), 'grouped': (train_idx_grp, test_idx_grp)}
all_preds = {}
for split_name, (tr_idx, te_idx) in splits.items():
    X_tfidf_tr, X_tfidf_te, X_struct_tr, X_struct_te, X_comb_tr, X_comb_te = build_features(tr_idx, te_idx)
    y_tr = df.loc[tr_idx, 'y'].values
    y_te = df.loc[te_idx, 'y'].values
    r, p = evaluate(LogisticRegression(max_iter=1000), X_tfidf_tr, y_tr, X_tfidf_te, y_te, f'{split_name}_tfidf_only_LR')
    results.append(r)
    r, p = evaluate(LogisticRegression(max_iter=1000), X_struct_tr, y_tr, X_struct_te, y_te, f'{split_name}_structural_only_LR')
    results.append(r)
    r, p = evaluate(LogisticRegression(max_iter=1000), X_comb_tr, y_tr, X_comb_te, y_te, f'{split_name}_combined_LR')
    results.append(r)
    all_preds[f'{split_name}_combined_LR'] = (y_te, p)
    r, p = evaluate(MultinomialNB(), X_tfidf_tr, y_tr, X_tfidf_te, y_te, f'{split_name}_tfidf_only_NB')
    results.append(r)
    r, p = evaluate(LinearSVC(max_iter=5000), X_comb_tr, y_tr, X_comb_te, y_te, f'{split_name}_combined_SVM')
    results.append(r)
res_df = pd.DataFrame(results)
print(res_df.round(4).to_string(index=False))
res_df.to_csv('outputs/figures/phase3_model_comparison.csv', index=False)
fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
for ax, key in zip(axes, ['naive_combined_LR', 'grouped_combined_LR']):
    y_te, p = all_preds[key]
    cm = confusion_matrix(y_te, p)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['human', 'ai'], yticklabels=['human', 'ai'], ax=ax)
    ax.set_title(key)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
plt.tight_layout()
plt.savefig('outputs/figures/phase3_confusion_matrices.png', dpi=130)
plt.close()
domain_test = df.loc[test_idx_grp]
y_te_grp = domain_test['y'].values
_, _, X_struct_tr_d, X_struct_te_d, X_comb_tr_d, X_comb_te_d = build_features(train_idx_grp, test_idx_grp)
model = LogisticRegression(max_iter=1000).fit(X_comb_tr_d, df.loc[train_idx_grp, 'y'].values)
domain_test = domain_test.copy()
domain_test['pred'] = model.predict(X_comb_te_d)
domain_report = domain_test.groupby('domain').apply(lambda g: pd.Series({'n': len(g), 'precision_ai': precision_score(g['y'], g['pred'], pos_label=1, zero_division=0), 'recall_ai': recall_score(g['y'], g['pred'], pos_label=1, zero_division=0)}))
print(domain_report.round(3))
domain_report.to_csv('outputs/figures/phase3_domain_breakdown.csv')