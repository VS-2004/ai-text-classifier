import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, GroupShuffleSplit, GroupKFold, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import precision_score, recall_score, f1_score, fbeta_score, accuracy_score, confusion_matrix, precision_recall_curve, make_scorer
import matplotlib.pyplot as plt
df = pd.read_csv('outputs/figures/processed_features.csv')
df['y'] = (df['label'] == 'ai').astype(int)
struct_cols = ['word_count', 'avg_sentence_length', 'punct_density', 'exclaim_count', 'has_caps_word', 'has_emoji', 'has_casual_marker', 'avg_word_length', 'ttr', 'domain_academic', 'domain_news', 'domain_social']
idx_all = df.index.values
y_all = df['y'].values
gss = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=42)
train_idx, test_idx = next(gss.split(idx_all, y_all, groups=df['template_id'].values))
train_idx = idx_all[train_idx]
test_idx = idx_all[test_idx]
X_train = df.loc[train_idx, ['text_clean'] + struct_cols]
X_test = df.loc[test_idx, ['text_clean'] + struct_cols]
y_train = df.loc[train_idx, 'y'].values
y_test = df.loc[test_idx, 'y'].values
groups_train = df.loc[train_idx, 'template_id'].values
preprocessor = ColumnTransformer([('tfidf', TfidfVectorizer(max_features=3000, ngram_range=(1, 2), min_df=2), 'text_clean'), ('struct', StandardScaler(), struct_cols)])
pipe = Pipeline([('prep', preprocessor), ('clf', LogisticRegression(max_iter=2000))])
param_grid = {'clf__C': [0.01, 0.1, 1, 10], 'clf__class_weight': [None, 'balanced']}
gkf = GroupKFold(n_splits=4)
cv_splits = list(gkf.split(X_train, y_train, groups=groups_train))
f05_scorer = make_scorer(fbeta_score, beta=0.5)
grid = GridSearchCV(pipe, param_grid, scoring=f05_scorer, cv=cv_splits, n_jobs=-1)
grid.fit(X_train, y_train)
print('best params:', grid.best_params_)
print('best cv f0.5:', round(grid.best_score_, 4))
best_model = grid.best_estimator_
preds = best_model.predict(X_test)
print('test accuracy:', round(accuracy_score(y_test, preds), 4))
print('test precision_ai:', round(precision_score(y_test, preds), 4))
print('test recall_ai:', round(recall_score(y_test, preds), 4))
print('test f1_ai:', round(f1_score(y_test, preds), 4))
cv_results = pd.DataFrame(grid.cv_results_)[['param_clf__C', 'param_clf__class_weight', 'mean_test_score']]
cv_results.to_csv('outputs/figures/phase4_grid_search_results.csv', index=False)
probs = best_model.predict_proba(X_test)[:, 1]
precisions, recalls, thresholds = precision_recall_curve(y_test, probs)
thresh_table = []
for target_p in [0.99, 0.95, 0.90, 0.85, 0.80]:
    valid = np.where(precisions[:-1] >= target_p)[0]
    if len(valid) > 0:
        best_idx = valid[np.argmax(recalls[:-1][valid])]
        thresh_table.append({'target_precision': target_p, 'threshold': thresholds[best_idx], 'achieved_precision': precisions[best_idx], 'achieved_recall': recalls[best_idx]})
thresh_df = pd.DataFrame(thresh_table)
print(thresh_df.round(4))
thresh_df.to_csv('outputs/figures/phase4_threshold_tuning.csv', index=False)
plt.figure(figsize=(6, 5))
plt.plot(recalls, precisions, marker='.')
plt.xlabel('Recall (ai)')
plt.ylabel('Precision (ai)')
plt.title('Precision-Recall Tradeoff (Tuned Model, Grouped Test Set)')
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('outputs/figures/phase4_precision_recall_curve.png', dpi=130)
plt.close()
cross_domain_results = []
for held_out_domain in ['academic', 'news', 'social']:
    train_mask = df['domain'] != held_out_domain
    test_mask = df['domain'] == held_out_domain
    Xd_train = df.loc[train_mask, ['text_clean'] + struct_cols]
    Xd_test = df.loc[test_mask, ['text_clean'] + struct_cols]
    yd_train = df.loc[train_mask, 'y'].values
    yd_test = df.loc[test_mask, 'y'].values
    model_d = Pipeline([('prep', ColumnTransformer([('tfidf', TfidfVectorizer(max_features=3000, ngram_range=(1, 2), min_df=2), 'text_clean'), ('struct', StandardScaler(), struct_cols)])), ('clf', LogisticRegression(max_iter=2000, C=grid.best_params_['clf__C'], class_weight=grid.best_params_['clf__class_weight']))])
    model_d.fit(Xd_train, yd_train)
    preds_d = model_d.predict(Xd_test)
    cross_domain_results.append({'held_out_domain': held_out_domain, 'n_test': test_mask.sum(), 'accuracy': accuracy_score(yd_test, preds_d), 'precision_ai': precision_score(yd_test, preds_d, zero_division=0), 'recall_ai': recall_score(yd_test, preds_d, zero_division=0)})
cross_df = pd.DataFrame(cross_domain_results)
print(cross_df.round(4))
cross_df.to_csv('outputs/figures/phase4_cross_domain_validation.csv', index=False)