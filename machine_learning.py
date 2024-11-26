# Step 1 import libraries
import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
import mlflow.xgboost

from sklearn.model_selection import train_test_split
from sklearn import svm, tree
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier

import matplotlib.pyplot as plt

# Step 2 read the csv files and create pandas dataframes
legitimate_df = pd.read_csv("structured_data_leg_2.csv")
phishing_df = pd.read_csv("structured_data_phishing_2.csv")


# Step 3 combine legitimate and phishing dataframes, and shuffle
df = pd.concat([legitimate_df, phishing_df], axis=0)
df = df.sample(frac=1)

# Step 4 remove 'url' and remove duplicates, then create X and Y for the models, Supervised Learning
df = df.drop('URL', axis=1)
df = df.drop_duplicates()

# Remove rows with missing values in X or Y
df = df.dropna()

X = df.drop('label', axis=1)
Y = df['label']

# Step 5 split data to train and test
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=10)

# Models to be trained
models = {
    "SVM": svm.LinearSVC(),
    "RandomForest": RandomForestClassifier(n_estimators=60),
    "DecisionTree": tree.DecisionTreeClassifier(),
    "AdaBoost": AdaBoostClassifier(),
    "NaiveBayes": GaussianNB(),
    "NeuralNetwork": MLPClassifier(alpha=1),
    "KNeighbors": KNeighborsClassifier(),
    "LogisticRegression": LogisticRegression(),
    "GradientBoosting": GradientBoostingClassifier(),
    "XGBoost": XGBClassifier()
}

# Set up MLflow experiment
mlflow.set_experiment("Phishing Detection Models")

# K-fold cross validation, and K = 5
K = 5
total = X.shape[0]
index = int(total / K)

# K-fold split
X_train_list, X_test_list = [], []
Y_train_list, Y_test_list = [], []

for i in range(K):
    X_test_split = X.iloc[i * index:(i + 1) * index]
    Y_test_split = Y.iloc[i * index:(i + 1) * index]

    X_train_split = X.iloc[np.r_[:i * index, (i + 1) * index:]]
    Y_train_split = Y.iloc[np.r_[:i * index, (i + 1) * index:]]

    X_train_list.append(X_train_split)
    X_test_list.append(X_test_split)
    Y_train_list.append(Y_train_split)
    Y_test_list.append(Y_test_split)

def calculate_measures(TN, TP, FN, FP):
    model_accuracy = (TP + TN) / (TP + TN + FN + FP)
    model_precision = TP / (TP + FP)
    model_recall = TP / (TP + FN)
    return model_accuracy, model_precision, model_recall

# Store the results
results = {
    "accuracy": [],
    "precision": [],
    "recall": []
}

model_names = []

# Function to train, predict and log results with MLflow
def train_and_log_model(model_name, model, X_train_list, X_test_list, Y_train_list, Y_test_list):
    accuracies, precisions, recalls = [], [], []

    for i in range(K):
        model.fit(X_train_list[i], Y_train_list[i])
        predictions = model.predict(X_test_list[i])
        tn, fp, fn, tp = confusion_matrix(y_true=Y_test_list[i], y_pred=predictions).ravel()
        accuracy, precision, recall = calculate_measures(tn, tp, fn, fp)
        accuracies.append(accuracy)
        precisions.append(precision)
        recalls.append(recall)

    avg_accuracy = sum(accuracies) / len(accuracies)
    avg_precision = sum(precisions) / len(precisions)
    avg_recall = sum(recalls) / len(recalls)
    print(f"Model: {model_name}")   
    print(f"Accuracy: {avg_accuracy}")
    print(f"Precision: {avg_precision}")
    print(f"Recall: {avg_recall}")  


    with mlflow.start_run(run_name=model_name):
        mlflow.log_param("model_name", model_name)
        mlflow.log_metric("accuracy", avg_accuracy)
        mlflow.log_metric("precision", avg_precision)
        mlflow.log_metric("recall", avg_recall)
        mlflow.sklearn.log_model(model, model_name)

    # Store results in dictionary
    results["accuracy"].append(avg_accuracy)
    results["precision"].append(avg_precision)
    results["recall"].append(avg_recall)
    model_names.append(model_name)

# Train and log all models
for model_name, model in models.items():
    train_and_log_model(model_name, model, X_train_list, X_test_list, Y_train_list, Y_test_list)
    print(train_and_log_model)

# Create DataFrame for the results
df_results = pd.DataFrame(data=results, index=model_names)
print(df_results)

# Visualize the results
ax = df_results.plot.bar(rot=0)
plt.show()
