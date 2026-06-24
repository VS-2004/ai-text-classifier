import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
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
preprocessor = ColumnTransformer([('tfidf', TfidfVectorizer(max_features=3000, ngram_range=(1, 2), min_df=2), 'text_clean'), ('struct', StandardScaler(), struct_cols)])
model = Pipeline([('prep', preprocessor), ('clf', LogisticRegression(max_iter=2000, C=10, class_weight='balanced'))])
model.fit(X_train, y_train)
preds = model.predict(X_test)
probs = model.predict_proba(X_test)[:, 1]
test_results = df.loc[test_idx, ['text_id', 'text_content', 'domain', 'template_id', 'label']].copy()
test_results['predicted'] = np.where(preds == 1, 'ai', 'human')
test_results['prob_ai'] = probs
false_positives = test_results[(test_results['label'] == 'human') & (test_results['predicted'] == 'ai')]
false_negatives = test_results[(test_results['label'] == 'ai') & (test_results['predicted'] == 'human')]
print('false positives (human flagged as ai):', len(false_positives))
print('false negatives (ai missed as human):', len(false_negatives))
print(false_positives[['domain', 'template_id', 'prob_ai', 'text_content']].head(10).to_string(index=False))
print(false_negatives[['domain', 'template_id', 'prob_ai', 'text_content']].head(10).to_string(index=False))
false_positives.to_csv('outputs/figures/phase6_false_positives.csv', index=False)
false_negatives.to_csv('outputs/figures/phase6_false_negatives.csv', index=False)
print('false negatives by domain:')
print(false_negatives['domain'].value_counts())
feature_names = model.named_steps['prep'].get_feature_names_out()
coefs = model.named_steps['clf'].coef_[0]
coef_df = pd.DataFrame({'feature': feature_names, 'coef': coefs}).sort_values('coef', ascending=False)
top_ai = coef_df.head(15)
top_human = coef_df.tail(15).sort_values('coef')
print('top 15 features pushing toward ai:')
print(top_ai.to_string(index=False))
print('top 15 features pushing toward human:')
print(top_human.to_string(index=False))
coef_df.to_csv('outputs/figures/phase6_model_coefficients.csv', index=False)
fig, ax = plt.subplots(figsize=(8, 7))
plot_df = pd.concat([top_human, top_ai]).sort_values('coef')
colors = ['#4C72B0' if c < 0 else '#DD8452' for c in plot_df['coef']]
ax.barh(plot_df['feature'], plot_df['coef'], color=colors)
ax.set_xlabel('Coefficient (negative = pushes toward human, positive = pushes toward ai)')
ax.set_title('Top 15 Features Driving Predictions in Each Direction')
plt.tight_layout()
plt.savefig('outputs/figures/phase6_coefficients_plot.png', dpi=130)
plt.close()