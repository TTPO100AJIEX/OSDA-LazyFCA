# OSDA-LazyFCA

LazyFCA (Lazy Formal Concept Analysis) is an interpretable machine learning classifier based on Formal Concept Analysis principles. It provides transparent, rule-based predictions with built-in explanation capabilities for binary classification tasks.

## Features

- **Interpretable Classifications**: Generates human-readable hypotheses in the form of attribute ranges and binary features
- **Explainable Predictions**: Every prediction comes with positive and negative classifiers that justify the decision
- **Rich Quality Metrics**: Evaluates classifiers using 25+ quality metrics including precision, WRAcc, Matthews correlation, information gain, and more
- **Flexible Filtering**: Customize classifier selection using multiple ranking criteria and top-k filtering
- **Mixed Data Types**: Handles both binary and numeric features seamlessly
- **Scikit-learn Compatible**: Follows familiar `fit`/`predict` API patterns

## Installation

### Requirements

- Python 3.11+
- NumPy
- Pandas
- scikit-learn
- joblib
- tqdm

### Install Dependencies

```bash
pip install -r requirements.txt
```

## Quick Start

```python
import pandas as pd
from lazyfca import LazyFCA

# Prepare your data (binary features as bool, numeric as float)
X_train = pd.DataFrame({
    'feature1': [True, False, True, False],
    'feature2': [1.5, 2.3, 1.8, 2.7],
    'feature3': [True, True, False, False]
})
y_train = pd.Series([1, 0, 1, 0])

# Initialize and train the classifier
model = LazyFCA(
    pos_params=LazyFCA.Params(precision=0.7, support=0.1),
    neg_params=LazyFCA.Params(precision=0.7, support=0.1),
    pos_weight=1.0
)
model.fit(X_train, y_train)

# Make predictions
X_test = pd.DataFrame({
    'feature1': [True, False],
    'feature2': [1.6, 2.5],
    'feature3': [True, False]
})
predictions = model.predict(X_test)

# Get explanations
explanations = model.explain(X_test)
for explanation in explanations:
    print(explanation.display())
```

## Architecture

### Core Components

- **`LazyFCA`**: Main classifier interface with fit/predict/explain methods
- **`Classifier`**: Represents a single hypothesis with associated quality metrics
- **`Hypothesis`**: Encodes the logical rule (binary attributes + numeric intervals)
- **`Explanation`**: Contains positive and negative classifiers for a prediction
- **`Dataset`**: Manages training data split by class labels
- **`Metrics`**: Comprehensive evaluation metrics for classifier quality

### Key Parameters

#### Classifier Metrics Thresholds (`LazyFCA.Params`)

Control which classifiers are considered valid:

- `supporters_covered`: Minimum positive examples covered
- `opposers_covered`: Maximum negative examples allowed
- `precision`: Minimum precision threshold
- `support`: Minimum support threshold
- `wracc`: Minimum weighted relative accuracy
- And 20+ additional metrics...

#### Ranking and Selection

- `pos_rank_by` / `neg_rank_by`: Metric name to sort classifiers by (e.g., "precision", "wracc")
- `pos_top_k` / `neg_top_k`: Maximum number of classifiers to keep after ranking
- `pos_weight`: Weight for positive classifiers in voting

## Quality Metrics

LazyFCA evaluates each classifier using a comprehensive set of metrics:

### Coverage Metrics
- Supporters covered / Opposers covered
- Support / Error rate
- Supporter-to-opposer ratio

### Discrimination Metrics
- Precision, Recall, F1-Score
- Lift, WRAcc (Weighted Relative Accuracy)
- Youden's J statistic
- Matthews Correlation Coefficient

### Information-Theoretic Metrics
- Information Gain
- Gini Gain
- Log Odds Ratio
- Chi-Squared statistic
- G-test

### Interpretability Metrics
- Interval tightness
- Description volume
- Simplicity prior
- Query similarity metrics

### Robustness Metrics
- Stability
- Delta stability
- Robustness

## Project Structure

```
OSDA-LazyFCA/
├── lazyfca/                  # Core package
│   ├── __init__.py          # Package exports
│   ├── lazyfca.py           # Main LazyFCA classifier
│   ├── classifier.py        # Classifier and Hypothesis classes
│   ├── dataset.py           # Dataset handling
│   └── explanation.py       # Explanation wrapper
├── utils/                    # Utility functions
│   ├── estimate_quality.py  # Quality estimation utilities
│   └── __init__.py
├── experiments/              # Experimental notebooks
│   └── rice_dataset.ipynb   # Rice dataset experiments
├── 1__eda.ipynb             # Exploratory Data Analysis
├── 2__traditional.ipynb     # Traditional ML comparisons
├── 3__ips.ipynb             # IPS experiments
├── 4__interpretability.ipynb # Interpretability analysis
├── 5__metrics.ipynb         # Metrics evaluation
├── 6__datasets.ipynb        # Dataset experiments
├── 7__metrics.ipynb         # Additional metrics
├── requirements.txt         # Python dependencies
└── README.md                # This file
```

## Examples

### Custom Metric Thresholds

```python
model = LazyFCA(
    pos_params=LazyFCA.Params(
        precision=0.8,
        support=0.2,
        wracc=0.1,
        stability=0.5
    ),
    neg_params=LazyFCA.Params(
        precision=0.8,
        support=0.2
    )
)
```

### Ranked Classifier Selection

```python
# Select top 10 classifiers by WRAcc
model = LazyFCA(
    pos_rank_by="wracc",
    pos_top_k=10,
    neg_rank_by="wracc",
    neg_top_k=10
)
```

### Parallel Prediction

```python
# Use all CPU cores for faster predictions
predictions = model.predict(X_test, n_jobs=-1)
explanations = model.explain(X_test, n_jobs=-1)
```

### Accessing Individual Explanations

```python
explanation = model.explain_sample(X_test.iloc[0])

print(f"Positive classifiers: {len(explanation.positive_classifiers)}")
print(f"Negative classifiers: {len(explanation.negative_classifiers)}")

# View detailed metrics
df = explanation.display()
print(df)
```

## Use Cases

LazyFCA is particularly well-suited for:

- **Medical Diagnosis**: Transparent decision-making with clear justifications
- **Credit Scoring**: Explainable loan approval/rejection reasons
- **Fraud Detection**: Interpretable anomaly detection with evidence
- **Customer Churn**: Understanding why customers leave with actionable insights
- **Quality Control**: Rule-based defect detection with clear criteria

## Research & Development

This project implements concepts from Formal Concept Analysis adapted for lazy evaluation and machine learning. The notebooks in the repository demonstrate:

- Comparative analysis with traditional ML algorithms
- Interpretability metrics and evaluation
- Performance on various datasets
- Threshold tuning strategies

## Contributing

Contributions are welcome! Areas for improvement include:

- Additional quality metrics
- Performance optimizations
- Multi-class classification support
- Integration with explainability frameworks
- More comprehensive documentation

## License

...

## Citation

...

<!-- If you use this implementation in your research, please cite:

```bibtex
@software{osda_lazyfca,
  title = {OSDA-LazyFCA: Interpretable Classification via Lazy Formal Concept Analysis},
  author = {Raimova, A.L.},
  year = {2026},
  url = {https://github.com/yourusername/OSDA-LazyFCA}
}
``` -->

## Contact

For questions, issues, or collaboration opportunities, please open an issue on GitHub.
