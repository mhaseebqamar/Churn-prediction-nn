# Customer Churn Prediction with Neural Networks

A Deep Learning final project (SGH Warsaw School of Economics, MSc Advanced Analytics & Big Data, 2025/26) comparing a feedforward neural network (MLP) against a Logistic Regression baseline for predicting customer churn on the IBM Telco Customer Churn dataset.

## Project context

This was an individual final project for the Deep Learning course. I worked through each stage (EDA, preprocessing, baseline, architecture design, hyperparameter tuning, evaluation) with the reasoning written out at every step. The question the project asks is not only "how well can a neural network predict churn?" but also "is a neural network actually justified for this problem compared with a simple linear model?"

## What this project does

* Predicts whether a telecom customer will churn, using the [IBM Telco Customer Churn dataset](https://github.com/IBM/telco-customer-churn-on-icp4d) (7,043 customers of a fictional telecom provider, 21 columns, 26.5% churn)
* Builds a class-weighted Logistic Regression baseline first, to set a performance floor
* Builds a feedforward MLP in Keras/TensorFlow and tunes it with a controlled set of 7 configurations
* Compares models on accuracy, precision, recall, F1 and ROC-AUC rather than accuracy alone, because the classes are imbalanced
* Evaluates the selected model once on a held-out test set that was never used for any modelling decision

## Key findings

**Validation set: MLP vs Logistic Regression**

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|
| Logistic Regression (class-weighted) | 0.742 | 0.509 | 0.769 | 0.613 | 0.829 |
| MLP 32→16, dropout 0.3 | 0.756 | 0.529 | 0.747 | 0.619 | 0.830 |

* **The neural network did not meaningfully beat Logistic Regression.** The MLP was 1.4 points higher on accuracy and 0.6 points higher on F1, with practically the same AUC (0.830 vs 0.829). On this dataset the added complexity of a neural network buys very little.
* **Tuning confirmed the original architecture.** Across 7 configurations (deeper 64→32→16, smaller 16→8, dropout 0.1 / 0.3 / 0.5, learning rate 0.0005 / 0.001 / 0.005), validation AUC only ranged from 0.826 to 0.831. The best configuration was the original 32→16 network with dropout 0.3 and learning rate 0.001. A bigger network did not help.

**Final test set (selected MLP, evaluated once)**

| Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|
| 0.744 | 0.510 | 0.800 | 0.623 | 0.849 |

* The model catches 80% of churners (224 of 280) at the default 0.5 threshold, at the cost of 215 false alarms among 777 customers who stayed. With class weighting, the model leans towards recall, which suits a retention use case where missing a churner usually costs more than an unnecessary retention offer.

**Churn drivers found in EDA**

* Short tenure, month-to-month contracts, higher monthly charges, fiber optic internet, electronic check payment and no tech support are all associated with higher churn rates.
* `tenure` and `TotalCharges` are strongly correlated (about 0.83).

## Pipeline overview

1. **EDA**: class balance, numeric distributions by churn, churn rate by category, correlation analysis. Found 11 blank `TotalCharges` values, all for customers with zero tenure (new customers not yet billed), and set them to 0.
2. **Preprocessing**: dropped `customerID`; mapped binary columns to 0/1; one-hot encoded multi-category columns (`drop_first=True`), keeping "No internet service" as its own category; stratified 70/15/15 train/validation/test split; `StandardScaler` fitted on the training set only to avoid leakage; balanced class weights for the imbalance.
3. **Baseline**: Logistic Regression with `class_weight="balanced"`.
4. **Neural network**: Dense(32, ReLU) → Dropout(0.3) → Dense(16, ReLU) → Dropout(0.3) → Dense(1, sigmoid); 1,537 parameters; Adam optimiser, binary cross-entropy, class weights, EarlyStopping on validation loss with best weights restored.
5. **Hyperparameter tuning**: 7 configurations, each changing one factor from the baseline (depth/width, dropout, learning rate), ranked by validation AUC.
6. **Final evaluation**: classification report, confusion matrix and ROC curve on the untouched test set.

## Tech stack

* Python, Jupyter Notebook
* pandas, NumPy: data handling
* scikit-learn: preprocessing, Logistic Regression, class weights, metrics
* TensorFlow / Keras: neural network
* matplotlib, seaborn: visualisation

## Project structure

```
churn-prediction-nn/
├── 01_churn_project.ipynb     # Full notebook: EDA → preprocessing → modelling → evaluation
├── data/
│   └── Telco-Customer-Churn.csv
└── README.md
```

To run it, update the CSV path in the data-loading cell to `data/Telco-Customer-Churn.csv`.

## Honest limitations

* **Single train/validation/test split** rather than k-fold cross-validation, so the results carry some split-to-split variance.
* **Logistic Regression was compared on the validation set only.** It was not scored on the test set, so the test results above apply to the MLP alone.
* **Fixed 0.5 decision threshold.** A real deployment would tune the threshold against the actual cost of false positives and false negatives.
* **Snapshot data.** Each customer appears once, with no behaviour history over time.
