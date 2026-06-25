# Customer Churn Prediction with Neural Networks

A Deep Learning coursework project (SGH Warsaw School of Economics, Winter 2024/25) comparing a feedforward neural network (MLP) against a Logistic Regression baseline for predicting customer churn on the IBM Telco Customer Churn dataset.

## Project context

This project was built as a final assignment for the **Deep Learning** course at SGH Warsaw School of Economics. It was developed as a guided, AI-assisted learning exercise — I worked through each stage (EDA, preprocessing, architecture design, hyperparameter tuning, evaluation) with explanations at every step, with the explicit goal of understanding *why* each decision was made, not just producing a working notebook. I'm including this context here in the interest of transparency rather than overstating independent authorship.

## What this project does

- Predicts whether a telecom customer will churn, using the [IBM Telco Customer Churn dataset](https://github.com/IBM/telco-customer-churn-on-icp4d) (~7,000 customers, 21 features)
- Builds a Logistic Regression baseline first, to honestly test whether a neural network is actually justified for this problem
- Builds and tunes a feedforward neural network (MLP) in Keras/TensorFlow
- Compares both models on accuracy, precision, recall, F1, and ROC-AUC (not just accuracy, since the dataset is moderately imbalanced — ~27% churn)
- Reports an honest final evaluation on a held-out test set that was never used during model selection

## Key findings

- The neural network slightly outperformed Logistic Regression (~2 points of accuracy/F1), with nearly identical AUC (~0.83 both) — a modest, honestly-reported improvement, not an inflated one.
- A deeper architecture (64→32→16) matched but did not meaningfully beat a simpler one (32→16) during hyperparameter tuning — evidence that bigger isn't automatically better on a dataset of this size.
- Strongest churn predictors found in EDA: short tenure, month-to-month contracts, high monthly charges, fiber internet, electronic check payment, and lack of tech support.

## Tech stack

- Python, Jupyter Notebook
- pandas, NumPy — data handling
- scikit-learn — preprocessing, Logistic Regression baseline, metrics
- TensorFlow / Keras — neural network
- matplotlib, seaborn — visualization

## Project structure

```
churn-prediction-nn/
├── 01_churn_project.ipynb     # Full notebook: EDA → preprocessing → modeling → evaluation
├── data/
│   └── Telco-Customer-Churn.csv
└── README.md
```

## Pipeline overview

1. **EDA** — class imbalance check, distribution analysis, churn-rate-by-category breakdown, correlation analysis
2. **Preprocessing** — categorical encoding (binary mapping + one-hot), `StandardScaler` fit only on training data (no leakage), stratified 70/15/15 train/val/test split, class weighting for imbalance
3. **Baseline model** — Logistic Regression with balanced class weights
4. **Neural network** — feedforward MLP (Dense → Dropout → Dense → Dropout → Sigmoid output), trained with Adam, binary cross-entropy, EarlyStopping
5. **Hyperparameter tuning** — controlled comparison across architecture depth/width, dropout rate, and learning rate, selected on validation AUC only
6. **Final evaluation** — confusion matrix, ROC curve, and full classification report on the untouched test set

## Honest limitations

- Single train/val/test split rather than k-fold cross-validation
- No customer behavior history over time — dataset is a single snapshot per customer
- Decision threshold fixed at 0.5 throughout; a real deployment would tune this against the actual business cost of false positives vs. false negatives

## What I'm still working on

I can explain the reasoning behind every step in this notebook, but I'm still building toward being able to write a pipeline like this unaided, from a blank notebook. This project reflects where I am in that process, not a finished skill.

## References

See the Bibliography section inside the notebook for full citations (Goodfellow et al., Chollet, scikit-learn, TensorFlow, Kingma & Ba, Srivastava et al., Saito & Rehmsmeier).
