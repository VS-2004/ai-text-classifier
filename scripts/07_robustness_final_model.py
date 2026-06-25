import pandas as pd
import numpy as np
import joblib
import json
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, precision_recall_curve, confusion_matrix
df = pd.read_csv('outputs/figures/processed_features.csv')
df['y'] = (df['label'] == 'ai').astype(int)
struct_cols = ['word_count', 'avg_sentence_length', 'punct_density', 'exclaim_count', 'has_caps_word', 'has_emoji', 'has_casual_marker', 'avg_word_length', 'ttr', 'domain_academic', 'domain_news', 'domain_social']
X_full = df[['text_clean'] + struct_cols]
y_full = df['y'].values
groups_full = df['template_id'].values
gkf = GroupKFold(n_splits=5)
fold_results = []
oof_probs = np.zeros(len(df))
oof_preds = np.zeros(len(df))
for fold_num, (train_idx, test_idx) in enumerate(gkf.split(X_full, y_full, groups=groups_full)):
    preprocessor = ColumnTransformer([('tfidf', TfidfVectorizer(max_features=3000, ngram_range=(1, 2), min_df=2), 'text_clean'), ('struct', StandardScaler(), struct_cols)])
    model = Pipeline([('prep', preprocessor), ('clf', LogisticRegression(max_iter=2000, C=10, class_weight='balanced'))])
    model.fit(X_full.iloc[train_idx], y_full[train_idx])
    probs = model.predict_proba(X_full.iloc[test_idx])[:, 1]
    preds = model.predict(X_full.iloc[test_idx])
    oof_probs[test_idx] = probs
    oof_preds[test_idx] = preds
    fold_results.append({'fold': fold_num, 'n_test': len(test_idx), 'accuracy': accuracy_score(y_full[test_idx], preds), 'precision_ai': precision_score(y_full[test_idx], preds, zero_division=0), 'recall_ai': recall_score(y_full[test_idx], preds, zero_division=0), 'f1_ai': f1_score(y_full[test_idx], preds, zero_division=0)})
fold_df = pd.DataFrame(fold_results)
print(fold_df.round(4).to_string(index=False))
print('mean +/- std across folds:')
summary = fold_df[['accuracy', 'precision_ai', 'recall_ai', 'f1_ai']].agg(['mean', 'std'])
print(summary.round(4))
fold_df.to_csv('outputs/figures/phase7_cv_robustness.csv', index=False)
pooled_cm = confusion_matrix(y_full, oof_preds)
print('pooled out-of-fold confusion matrix:')
print(pooled_cm)
precisions, recalls, thresholds = precision_recall_curve(y_full, oof_probs)
print('max achievable pooled precision at any threshold:', round(precisions[:-1].max(), 4))
target_options = [0.95, 0.90, 0.85, 0.80, 0.75, 0.70]
final_threshold = None
target_p = None
for tp in target_options:
    valid = np.where(precisions[:-1] >= tp)[0]
    if len(valid) > 0:
        best_idx = valid[np.argmax(recalls[:-1][valid])]
        final_threshold = thresholds[best_idx]
        target_p = tp
        break
if final_threshold is None:
    final_threshold = 0.5
    target_p = None
    print('no threshold met any target precision, defaulting to 0.5')
else:
    print('chosen final threshold for precision>=', target_p, ':', round(final_threshold, 4))
final_preds = (oof_probs >= final_threshold).astype(int)
print('pooled metrics at chosen threshold:')
print('accuracy', round(accuracy_score(y_full, final_preds), 4))
print('precision_ai', round(precision_score(y_full, final_preds), 4))
print('recall_ai', round(recall_score(y_full, final_preds), 4))
print('f1_ai', round(f1_score(y_full, final_preds), 4))
final_preprocessor = ColumnTransformer([('tfidf', TfidfVectorizer(max_features=3000, ngram_range=(1, 2), min_df=2), 'text_clean'), ('struct', StandardScaler(), struct_cols)])
final_model = Pipeline([('prep', final_preprocessor), ('clf', LogisticRegression(max_iter=2000, C=10, class_weight='balanced'))])
final_model.fit(X_full, y_full)
joblib.dump(final_model, 'outputs/figures/final_model.joblib')
with open('outputs/figures/final_model_config.json', 'w') as f:
    json.dump({'model': 'LogisticRegression', 'C': 10, 'class_weight': 'balanced', 'features': 'tfidf_3000_1-2gram_min_df2 + structural', 'decision_threshold': float(final_threshold), 'expected_precision_ai': float(precision_score(y_full, final_preds)), 'expected_recall_ai': float(recall_score(y_full, final_preds))}, f, indent=2)
print('final model and config saved')