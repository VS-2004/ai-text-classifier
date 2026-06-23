import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import GroupShuffleSplit, GroupKFold, GridSearchCV, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import precision_score, recall_score, f1_score, fbeta_score, accuracy_score, make_scorer
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
gkf = GroupKFold(n_splits=4)
cv_splits = list(gkf.split(X_train, y_train, groups=groups_train))
f05_scorer = make_scorer(fbeta_score, beta=0.5)
results = []
rf_pipe = Pipeline([('prep', preprocessor), ('clf', RandomForestClassifier(random_state=42, class_weight='balanced'))])
rf_grid = {'clf__n_estimators': [200, 400], 'clf__max_depth': [None, 20]}
rf_search = GridSearchCV(rf_pipe, rf_grid, scoring=f05_scorer, cv=cv_splits, n_jobs=-1)
rf_search.fit(X_train, y_train)
rf_preds = rf_search.predict(X_test)
results.append({'model': 'RandomForest', 'best_params': rf_search.best_params_, 'accuracy': accuracy_score(y_test, rf_preds), 'precision_ai': precision_score(y_test, rf_preds), 'recall_ai': recall_score(y_test, rf_preds), 'f1_ai': f1_score(y_test, rf_preds)})
gb_pipe = Pipeline([('prep', preprocessor), ('clf', GradientBoostingClassifier(random_state=42))])
gb_grid = {'clf__n_estimators': [150, 300], 'clf__max_depth': [2, 3]}
gb_search = GridSearchCV(gb_pipe, gb_grid, scoring=f05_scorer, cv=cv_splits, n_jobs=-1)
gb_search.fit(X_train, y_train)
gb_preds = gb_search.predict(X_test)
results.append({'model': 'GradientBoosting', 'best_params': gb_search.best_params_, 'accuracy': accuracy_score(y_test, gb_preds), 'precision_ai': precision_score(y_test, gb_preds), 'recall_ai': recall_score(y_test, gb_preds), 'f1_ai': f1_score(y_test, gb_preds)})
results.append({'model': 'LogisticRegression (Phase 4 baseline)', 'best_params': 'C=10, class_weight=balanced', 'accuracy': 0.8772, 'precision_ai': 1.0, 'recall_ai': 0.6996, 'f1_ai': 0.8232})
res_df = pd.DataFrame(results)
print(res_df.to_string(index=False))
res_df.to_csv('outputs/figures/phase5_tree_model_comparison.csv', index=False)
best_rf = rf_search.best_estimator_
feature_names = best_rf.named_steps['prep'].get_feature_names_out()
importances = best_rf.named_steps['clf'].feature_importances_
top_idx = np.argsort(importances)[-15:][::-1]
print('top 15 RF feature importances:')
for i in top_idx:
    print(feature_names[i], round(importances[i], 4))