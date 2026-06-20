import pandas as pd
import re
df = pd.read_csv('data/ai_vs_human_text_2026.csv')
df['text_clean'] = df['text_content'].str.strip().str.replace(r'\s+', ' ', regex=True)
df['punct_count'] = df['text_content'].apply(lambda x: sum(1 for c in x if c in '.,!?;:'))
df['punct_density'] = df['punct_count'] / df['word_count']
df['exclaim_count'] = df['text_content'].str.count('!')
df['has_caps_word'] = df['text_content'].apply(lambda x: int(any(w.isupper() and len(w) > 1 for w in x.split())))
df['has_emoji'] = df['text_content'].apply(lambda x: int(any(ord(c) > 10000 for c in x)))
df['has_casual_marker'] = df['text_content'].str.contains(r'\blol\b|\bhonestly\b|\btbh\b|\bngl\b|\bidk\b', case=False, regex=True).astype(int)
df['avg_word_length'] = df['text_content'].apply(lambda x: sum(len(w) for w in x.split()) / len(x.split()))
df['ttr'] = df['text_content'].apply(lambda x: len(set(w.lower() for w in x.split())) / len(x.split()))
domain_dummies = pd.get_dummies(df['domain'], prefix='domain')
df = pd.concat([df, domain_dummies], axis=1)
def template_sig(row):
    return row['text_content'].replace(row['topic_hint'], '<TOPIC>')
df['template_sig'] = df.apply(template_sig, axis=1)
df['template_id'] = df['template_sig'].astype('category').cat.codes
feature_cols = ['word_count', 'avg_sentence_length', 'punct_density', 'exclaim_count', 'has_caps_word', 'has_emoji', 'has_casual_marker', 'avg_word_length', 'ttr', 'domain_academic', 'domain_news', 'domain_social']
print(df.groupby('label')[feature_cols[:9]].mean().round(3))
print('unique templates:', df['template_id'].nunique())
out_cols = ['text_id', 'text_content', 'text_clean', 'label', 'domain', 'template_id'] + feature_cols
df[out_cols].to_csv('outputs/figures/processed_features.csv', index=False)
print('saved processed_features.csv with', len(out_cols), 'columns and', len(df), 'rows')