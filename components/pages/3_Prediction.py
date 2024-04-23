import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.graph_objs as go
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier


def impute_missing_values(df):
    cat_features = df.select_dtypes(include=['object']).columns
    for cat_feature in cat_features:
        mode = df[cat_feature].mode()[0]
        df[cat_feature].fillna(mode, inplace=True)
    num_features = df.select_dtypes(include=['int64', 'float64']).columns
    for num_feature in num_features:
        mean = df[num_feature].mean()
        df[num_feature].fillna(mean, inplace=True)
    return df


def normalize(X):
    means = np.mean(X, axis=0)
    stds = np.std(X, axis=0)
    normalized_X = (X - means) / stds
    return normalized_X


def plot_confusion_matrix(conf_matrix, classes):
    plt.figure(figsize=(11, 3))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.title('Confusion Matrix')
    st.set_option('deprecation.showPyplotGlobalUse', False)
    st.pyplot()


def main():
    st.set_page_config(page_title="Prediction", page_icon="🔮", layout="wide")

    # Load data
    data = pd.read_csv(r'C:\Users\hp\Desktop\DataAnalysisApp\Data_Analysis_App\artifact\HousePriceData.csv')
    data = data.sample(200)

    # Impute missing values
    data = impute_missing_values(data)

    # Feature engineering and normalization
    X = data.drop(columns=['SalePrice'])  # Features
    Y = data['SalePrice']  # Target

    # One-hot encoding for categorical variables
    features = list(X.columns)
    numerical_vars = X.select_dtypes(include=['int64', 'float64']).columns
    cat_vars = list(set(features) - set(numerical_vars))
    for var in cat_vars:
        one_hot_df = pd.get_dummies(X[var], prefix=var)
        X = pd.concat([X, one_hot_df], axis=1)
        X.drop(var, axis=1, inplace=True)

    X = normalize(X)

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=.2, random_state=42)

    # Define classifiers
    classifiers = {
        'Logistic Regression': LogisticRegression(solver='lbfgs', penalty='l2', max_iter=1000000, multi_class='multinomial'),
        'Random Forest Classifier': RandomForestClassifier(n_estimators=100, random_state=123),
        'Ridge Classifier': RidgeClassifier(alpha=1.0, solver='auto', random_state=123),
        'Gradient Boosting Classifier': GradientBoostingClassifier(n_estimators=100, random_state=123),
        'Multi-layer Perceptron Classifier': MLPClassifier(hidden_layer_sizes=(15, 10), alpha=3, learning_rate='adaptive', max_iter=100000),
    }

    # Sidebar
    st.sidebar.title('Model Selection')
    selected_model = st.sidebar.selectbox('Choose a model', list(classifiers.keys()))

    # Model training and evaluation
    selected_clf = classifiers[selected_model]
    selected_clf.fit(X_train, y_train)
    y_pred = selected_clf.predict(X_test)

    # Evaluation metrics
    st.title('Alzheimer\'s Disease Prediction')
    st.write('-------')
    st.write('## 🔍 Model Evaluation:', selected_model)

    evaluation_metrics = {
        'Metric': ['Accuracy', 'Precision', 'Recall', 'F1 Score'],
        'Value': [
            accuracy_score(y_test, y_pred),
            precision_score(y_test, y_pred, average='weighted'),
            recall_score(y_test, y_pred, average='weighted'),
            f1_score(y_test, y_pred, average='weighted')
        ]
    }

    # Create a DataFrame from the evaluation metrics dictionary
    metrics_df = pd.DataFrame(evaluation_metrics)

    # Color-based formatting for metrics
    red_color = '#ff0000'  # Red color
    green_color = '#00ff00'  # Green color
    
    def apply_color(value):
        trend_float = float(value)
        if trend_float > 0.5:
            return f'background-color: {green_color}; color: black'
        elif trend_float < 0.5:
            return f'background-color: {red_color}; color: black'
        else:
            return ''

    styled_metrics_df = metrics_df.style.applymap(apply_color, subset=['Value'])

    # Display the styled DataFrame
    st.write(styled_metrics_df)

    # Confusion matrix
    st.write('### Confusion Matrix')
    conf_matrix = confusion_matrix(y_test, y_pred)
    plot_confusion_matrix(conf_matrix, selected_clf.classes_)


if __name__ == "__main__":
    main()
