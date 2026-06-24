# AI vs Human Text Classifier

A binary text classification pipeline that detects AI-generated vs human-written
content across three domains — social media, news, and academic writing — built
as an 8-week data science capstone project.

## Business Problem

A content platform hosting social posts, news articles, and academic summaries
needs to automatically flag AI-generated text before publication, without
falsely flagging genuine human writers. The model must generalize across three
very different writing styles, where simple heuristics like text length fail
because the AI-vs-human signal direction actually reverses between domains.

## Dataset

`ai_vs_human_text_2026.csv` - 2,000 labeled text samples (66.7% human, 33.3% ai),
balanced across academic, news, and social domains, with structural metadata
(word_count, avg_sentence_length, topic_hint, source_model).

## Project Structure

- data/ - raw dataset
- scripts/01_eda.py - exploratory analysis
- scripts/02_feature_engineering.py - stylistic features + template grouping
- scripts/03_baseline_modeling.py - model training and evaluation
- outputs/figures/ - generated plots and result tables

## Setup

pip install pandas matplotlib seaborn scikit-learn scipy

## Key Finding: Template Leakage

The dataset's AI samples are built from only 58 underlying sentence templates
with topic words swapped in. A random train/test split lets identical template
wording leak across both sides, inflating accuracy to 100% - a false signal.
This project evaluates every model two ways: a naive random split (inflated,
shown for contrast) and a template-grouped split (templates never shared
between train and test - the honest generalization test).

## Results (Template-Grouped Split)

| Features | Accuracy | Precision (ai) | Recall (ai) | F1 |
|---|---|---|---|---|
| TF-IDF only | 59.1% | 0.00 | 0.00 | 0.00 |
| Structural features only | 64.0% | 0.57 | 0.52 | 0.54 |
| TF-IDF + structural (combined) | 83.5-86.0% | 0.97 | 0.62-0.68 | 0.75-0.80 |

Pure TF-IDF collapses on unseen templates since it memorized exact wording,
not the concept of AI-ness. Combining it with stylistic features (punctuation
density, emoji/slang presence, word length, domain) generalizes far better -
0.97 precision means false alarms on genuine human writers are rare, at the
cost of missing roughly a third of AI text on unfamiliar phrasing.

### Domain Breakdown (combined model, grouped split)

| Domain | Precision (ai) | Recall (ai) |
|---|---|---|
| Academic | 1.00 | 0.49 |
| News | 0.89 | 1.00 |
| Social | 1.00 | 0.58 |

## Phases

- [x] Phase 1: EDA
- [x] Phase 2: Data Cleaning and Feature Engineering
- [x] Phase 3: Baseline Modeling
- [x] Phase 4: Tuning and Cross-Domain Validation
- [x] Phase 5: Advanced Modeling
- [x] Phase 6: Error Analysis and Interpretability
- [ ] Phase 7: Robustness and Final Model Selection
- [ ] Phase 8: Final Report

## Running the Pipeline

python scripts/01_eda.py
python scripts/02_feature_engineering.py
python scripts/03_baseline_modeling.py