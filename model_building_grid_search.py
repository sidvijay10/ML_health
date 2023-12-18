#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

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
#import shap
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier


identifier = "Meningioma"
matrix = pd.read_csv('final_matrix_' + identifier + '.csv', index_col='Gene')
final_matrix = matrix.T

X = final_matrix.values

# Formatting response data
response_data = pd.read_csv("Meningioma_response.csv")
patient_list = response_data.iloc[:, 0].values
y = response_data['Response'].values


# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=17)

# Normalize features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

models = {
    'LogisticRegression': LogisticRegression(max_iter=1000),
    #'XGBoost': XGBClassifier(use_label_encoder=False, eval_metric='logloss', verbose = 3),
    'SVM': SVC(probability=True),
    'DecisionTree': DecisionTreeClassifier()
}

# Hyperparameter grid
param_grid = {
    'LogisticRegression': {'penalty': ['l1', 'l2', 'elasticnet', None], 'C': np.logspace(-4, 4, 20)},
    #'XGBoost': {'n_estimators': [100, 200], 'max_depth': [3, 5, 7], 'learning_rate': [0.01, 0.1]},
    'SVM': {'C': [0.1, 1, 10, 100], 'gamma': [1, 0.1, 0.01, 0.001], 'kernel': ['rbf', 'poly', 'sigmoid']},
    'DecisionTree': {'max_depth': [None, 10, 20, 30, 40], 'min_samples_split': [2, 5, 10], 'min_samples_leaf': [1, 2, 4]}
    #'NeuralNetwork': {'hidden_layer_sizes': [(50,50), (100,)], 'activation': ['relu', 'tanh']}
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
    
    
import matplotlib.pyplot as plt

for model_name, metrics in results.items():
    plt.figure(figsize=(10, 6))
    plt.bar(metrics.keys(), metrics.values())
    plt.title(f'Performance Metrics for {model_name}')
    plt.ylabel('Score')
    plt.show()
    
    
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

def plot_roc_curve(model, X_test, y_test, name):
    y_scores = model.predict_proba(X_test)[:, 1]
    fpr, tpr, thresholds = roc_curve(y_test, y_scores)
    roc_auc = auc(fpr, tpr)

    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'Receiver Operating Characteristic: {name}')
    plt.legend(loc="lower right")
    plt.show()

#plot_roc_curve(best_model_logistic_regression, X_test, y_test, "Logistic Regression")


# Building Neural Network Model

from scikeras.wrappers import KerasClassifier
from keras.models import Sequential
from keras.layers import Dense, Dropout, PReLU
from sklearn.model_selection import GridSearchCV, RepeatedKFold

# Modify the build_classifier function
def build_classifier(meta, optimizer='adam', init='uniform', activation='relu', hl_1=64, hl_2=64, drop_percent=0.5, num_hidden_layers=1):
    classifier = Sequential()
    classifier.add(Dense(units=hl_1, kernel_initializer=init, activation=activation, input_dim=meta['n_features_in_']))
    classifier.add(Dropout(drop_percent))
    for _ in range(num_hidden_layers - 1):
        classifier.add(Dense(units=hl_2, kernel_initializer=init, activation=activation))
        classifier.add(Dropout(drop_percent))
    classifier.add(Dense(units=1, kernel_initializer=init, activation='sigmoid'))
    classifier.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
    return classifier

# Instantiate KerasClassifier
model = KerasClassifier(model=build_classifier, batch_size=8, epochs = 100)
param_grid = {
    'model__init': ['uniform'],
    'model__activation': ['relu'],
    'model__optimizer': ['adagrad', 'adamax', 'adam'],
    'model__drop_percent': [0],
    'model__num_hidden_layers': [2],
    'model__hl_1': [200],
    'model__hl_2': [200]
}

grid = GridSearchCV(estimator=model, param_grid=param_grid, cv=5, scoring=['f1', 'accuracy'], refit="f1")
grid_result = grid.fit(X_train, y_train)

# Print best score and parameters
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
means_f1 = grid_result.cv_results_['mean_test_f1']
means_acc = grid_result.cv_results_['mean_test_accuracy']
params = grid_result.cv_results_['params']
best_params = grid_result.best_params_


# After fitting GridSearchCV
best_model = grid_result.best_estimator_
y_pred = best_model.predict(X_test)

# Calculate additional metrics
roc_auc = roc_auc_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)

# Print metrics
print(f"ROC AUC: {roc_auc:.2f}")
print(f"Precision: {precision:.2f}")
print(f"Recall: {recall:.2f}")
