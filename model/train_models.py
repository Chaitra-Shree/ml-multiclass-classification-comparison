"""
ML Assignment 2 - Stramlit Web Application
Mushroom Classification
"""

import pandas as pd
import numpy as np
import pickle
import json
import os
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    confusion_matrix,
    classification_report
)
import warnings
warnings.filterwarnings('ignore')


def load_and_preprocess_data():
    """
    Load Mushroom dataset from local file and preprocess it
    Dataset source : Kaggle- Mushroom Classification Dataset
    """
    #Get the project root directory
    project_root = Path(__file__).parent.parent

    #column names for mushroom datset
    column_names = [
        'class',
        'cap-shape',
        'cap-surface',
        'cap-color',
        'bruises',
        'odor',
        'gill-attachment',
        'gill-spacing',
        'gill-size',
        'gill-color',
        'stalk-shape',
        'stalk-root',
        'stalk-surface-above-ring',
        'stalk-surface-below-ring',
        'stalk-color-above-ring',
        'stalk-color-below-ring',
        'veil-type',
        'veil-color',
        'ring-number',
        'ring-type',
        'spore-print-color',
        'population',
        'habitat'
    ]

    #Load dat from local file
    data_path = project_root / 'data' / 'agaricus-lepiota.data'
    if not data_path.exists():
        raise FileNotFoundError(f"Data file not found at: {data_path}")
    
    df = pd.read_csv(data_path, names=column_names)

    print(f"Dataset shape: {df.shape}")
    print(f"Number of features: {len(df.columns)-1}")
    print(f"Number of instances: {len(df)}")
    print(f"\nTarget distribution:\n{df['class'].value_counts()}")

    #Separate features and target
    X = df.drop('class', axis=1)
    y = df['class']


    #Encode Categorical features
    label_encoders = {}
    for column in X.columns:
        le = LabelEncoder()
        X[column] = le.fit_transform(X[column])
        label_encoders[column] = le

    #Encode target variable ( e = edible, p = poisonous)
    target_encoder = LabelEncoder()
    y = target_encoder.fit_transform(y)

    #Save encoders
    model_dir = project_root / 'model'
    model_dir.mkdir(exist_ok=True)

    with open(model_dir / 'label_encoders.pkl', 'wb') as f:
        pickle.dump(label_encoders, f)

    with open(model_dir / 'target_encoder.pkl', 'wb') as f:
        pickle.dump(target_encoder, f)

    #Split Data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    #Save test data for streamlit app(with encoded values)
    test_data = X_test.copy()
    test_data['class'] = y_test
    test_data.to_csv(model_dir / 'test_data.csv', index=False)

    print(f"\nTrain set size: {len(X_train)}")
    print(f"Test set size: {len(X_test)}")
    print(f"Test data file saved: model/test_data.csv ({len(X_test)} samples)")

    return X_train, X_test, y_train, y_test, X.columns.tolist()


def calculate_metrics(y_true, y_pred, y_pred_proba=None):
    """
    Calculating the evaluation metrics
    """
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, average='binary'),
        'recall': recall_score(y_true, y_pred, average='binary'),
        'f1': f1_score(y_true, y_pred, average='binary'),
        'mcc': matthews_corrcoef(y_true, y_pred)
    }

    #AUC score requires probability predictions
    if y_pred_proba is not None:
        if len(y_pred_proba.shape) > 1 and y_pred_proba.shape[1] > 1:
            metrics['auc'] = roc_auc_score(y_true, y_pred_proba[:, 1])
        else:
            metrics['auc'] = roc_auc_score(y_true, y_pred_proba)
    else:
        metrics['auc'] = 0.0

    return metrics


def train_and_evaluate_models(X_train, X_test, y_train, y_test):
    project_root = Path(__file__).parent.parent
    model_dir = project_root / 'model'
    model_dir.mkdir(exist_ok=True)

    results = {}

    #Define models
    models = {
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
        'Decision Tree': DecisionTreeClassifier(random_state=42, max_depth=10),
        'kNN': KNeighborsClassifier(n_neighbors=5),
        'Naive Bayes': GaussianNB(),
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42, max_depth=10),
        'XGBoost': XGBClassifier(n_estimators=100, random_state=42, max_depth=6, use_label_encoder=False, eval_metric='logloss')
    }

    print("\n" + "="*80)
    print("TRAINING AND EVALUATING MODELS")
    print("="*80)

    for model_name, model in models.items():
        print(f"\nTraining {model_name}...")

        #Train model
        model.fit(X_train, y_train)

        #Make Predictions
        y_pred = model.predict(X_test)

        #Get Probability predictions for AUC
        try:
            y_pred_proba = model.predict_proba(X_test)
        except:
            y_pred_proba = None
        
        #Calculate metrics
        metrics = calculate_metrics(y_test, y_pred, y_pred_proba)

        #Get confusion matrix and classification report
        cm = confusion_matrix(y_test, y_pred)
        cr = classification_report(y_test, y_pred, output_dict=True)

        #Store results
        results[model_name] = {
            'metrics': metrics,
            'confusion_matrix': cm.tolist(),
            'classification_report': cr
        }

        #Save model
        model_filename = model_dir / f"{model_name.lower().replace(' ', '_')}_model.pkl"
        with open(model_filename, 'wb') as f:
            pickle.dump(model, f)

            print(f"✓ {model_name} trained successfully")
            print(f"  Accuracy: {metrics['accuracy']:.4f}")
            print(f"  AUC: {metrics['auc']:.4f}")
            print(f"  F1 Score: {metrics['f1']:.4f}")
            print(f"  Accuracy: {metrics['mcc']:.4f}")

    #Save results
    with open(model_dir / 'results.json', 'w') as f:
        json.dump(results, f, indent=4)

    return results


def print_comparison_table(results):
    print("\n" + "="*80)
    print("MODEL COMPARISION TABLE")
    print("="*80)
    print(f"{'Model':<25} {'Accuracy':<10} {'AUC':<10} {'Precision':<10} {'Recall':<10} {'F1':<10} {'MCC':<10}")
    print("-"*80)

    for model_name, result in results.items():
        metrics = result['metrics']
        print(f"{model_name:<25} {metrics['accuracy']:<10.4f} {metrics['auc']:<10.4f} "
              f"{metrics['precision']:<10.4f} {metrics['recall']:<10.4f} "
              f"{metrics['f1']:<10.4f} {metrics['mcc']:<10.4f}")

    print("="*80)


def main():
    """
    Main function to run the entire pipeline
    """
    print("="*80)
    print("ML ASSIGNMENT 2 - MUSHOOM CLASSIFICATION")
    print("="*80)

    #Load and preprocess the data
    print("\n[Step 1] Loading and preprocessing data...")
    X_train, X_test, y_train, y_test, feature_name = load_and_preprocess_data()

    #Train and evaluate models
    print("\n[Step 2] Training and evaluating models...")
    results = train_and_evaluate_models(X_train, X_test, y_train, y_test)

    #Print comparision table
    print_comparison_table(results)

    print("\n✓ All models trained and saved successfully!")
    print("✓ Results saved to 'model/results.json'")
    print("✓ Test data saved to 'model/test_data.csv'")


if __name__ == "__main__":
    main()