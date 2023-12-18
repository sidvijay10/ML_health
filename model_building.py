#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec  6 18:35:44 2023

@author: sidvijay
"""

import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import RFE, VarianceThreshold
from sklearn.linear_model import ElasticNet
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score, roc_auc_score
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
from sklearn.inspection import permutation_importance
import shap
import numpy as np
from sklearn.linear_model import LogisticRegression


df = pd.read_csv("survival_L_all_1.csv")
df.dropna(subset=['within_2yrs'], inplace=True)


y = df['within_2yrs'].astype(int).values
X = df.drop(columns=['coordinate_ID', 'dbscan_class.x', 'within_2yrs', 'within_5yrs', 'time.for.KM_final'])

# Split the data
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=17)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=17)

# Normalize features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)
X_test = scaler.transform(X_test)

models = {
    'LogisticRegression': LogisticRegression(max_iter=1000),
    'XGBoost': XGBClassifier(use_label_encoder=False, eval_metric='logloss', verbose = 3),
    'NeuralNetwork': MLPClassifier(max_iter=300)
}

# Hyperparameter grid
param_grid = {
    'LogisticRegression': {'penalty': ['l1', 'l2', 'elasticnet', None], 'C': np.logspace(-4, 4, 20)},
    'XGBoost': {'n_estimators': [100, 200], 'max_depth': [3, 5, 7], 'learning_rate': [0.01, 0.1]},
    'NeuralNetwork': {'hidden_layer_sizes': [(50,50), (100,)], 'activation': ['relu', 'tanh']}
}


# Evaluation function
def evaluate_model(model, X_test, y_test):
    print()
    y_pred = model.predict(X_test)
    print("YPRED")
    print(y_pred)
    print(y_test)
    threshold = 0.5
    y_pred = np.where(y_pred > threshold, 1, 0)
    print("NP WHERE")
    metrics = {
        'f1_score': f1_score(y_test, y_pred),
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
        'roc_auc': roc_auc_score(y_test, y_pred)
    }
    print(metrics)
    return metrics

# Training and evaluation
results = {}
for name, model in models.items():
    print(name)
    grid = GridSearchCV(model, param_grid[name], cv=5, scoring='f1', n_jobs=4)
    grid.fit(X_train, y_train)
    best_model = grid.best_estimator_
    results[name] = evaluate_model(best_model, X_test, y_test)



# SHAP and Permutation Feature Importance
model = RandomForestClassifier().fit(X_train, y_train)
shap_values = shap.TreeExplainer(model).shap_values(X_test)
perm_importance = permutation_importance(model, X_test, y_test, n_repeats=10)

print(results)
print("SHAP Values:", shap_values)
print("Permutation Importance:", perm_importance.importances_mean)
