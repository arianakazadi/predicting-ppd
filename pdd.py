
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, f1_score, classification_report
import matplotlib.pyplot as plt


#import data
df = pd.read_csv("/Users/arianakazadi/Desktop/DSM25/postnatal.data.csv")

#target mapping
target_mapping = {
    "Yes": 1,
    "No": 0,
    "Maybe": 1,
    "Sometimes": 1
}

df["Feeling anxious"] = df["Feeling anxious"].map(target_mapping)
df = df.dropna(subset=["Feeling anxious"])
df["Feeling anxious"] = df["Feeling anxious"].astype(int)

#predictors
feature_cols = [
    "Age",
    "Feeling sad or Tearful",
    "Irritable towards baby & partner",
    "Trouble sleeping at night",
    "Problems concentrating or making decision",
    "Overeating or loss of appetite",
    "Feeling of guilt",
    "Problems of bonding with baby",
    "Suicide attempt"
]

X = df[feature_cols]
y = df["Feeling anxious"]

#encode predictors
X_encoded = pd.get_dummies(X, drop_first=True)

print("Encoded feature columns:")
print(X_encoded.columns)

#train test split
X_train, X_test, y_train, y_test = train_test_split(
    X_encoded,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

#train models

#LR
log_reg = LogisticRegression(max_iter=1000)
log_reg.fit(X_train, y_train)
y_pred_log = log_reg.predict(X_test)

#SVM
svm_model = SVC(kernel="rbf")
svm_model.fit(X_train, y_train)
y_pred_svm = svm_model.predict(X_test)

#KNN
knn_model = KNeighborsClassifier(n_neighbors=5)
knn_model.fit(X_train, y_train)
y_pred_knn = knn_model.predict(X_test)

#evaluations
def evaluate_model(y_true, y_pred, model_name="Model"):
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    print(f"=== {model_name} ===")
    print(f"Accuracy : {accuracy:.3f}")
    print(f"Precision: {precision:.3f}")
    print(f"F1-score : {f1:.3f}")
    print()

    return accuracy, precision, f1

#evals of each model
metrics_log = evaluate_model(y_test, y_pred_log, "Logistic Regression")
metrics_svm = evaluate_model(y_test, y_pred_svm, "SVM")
metrics_knn = evaluate_model(y_test, y_pred_knn, "KNN")

#print("Classification report for Logistic Regression:")
#print(classification_report(y_test, y_pred_log, zero_division=0))

#compare models
results = pd.DataFrame({
    "Model": ["Logistic Regression", "SVM", "KNN"],
    "Accuracy": [metrics_log[0], metrics_svm[0], metrics_knn[0]],
    "Precision": [metrics_log[1], metrics_svm[1], metrics_knn[1]],
    "F1 Score": [metrics_log[2], metrics_svm[2], metrics_knn[2]],
})

#print("\n=== Model Comparison ===")
#print(results)

#plots
models = ["Logistic Regression", "SVM", "KNN"]

accuracy = [
    accuracy_score(y_test, y_pred_log),
    accuracy_score(y_test, y_pred_svm),
    accuracy_score(y_test, y_pred_knn)
]

precision = [
    precision_score(y_test, y_pred_log, zero_division=0),
    precision_score(y_test, y_pred_svm, zero_division=0),
    precision_score(y_test, y_pred_knn, zero_division=0)
]

f1 = [
    f1_score(y_test, y_pred_log, zero_division=0),
    f1_score(y_test, y_pred_svm, zero_division=0),
    f1_score(y_test, y_pred_knn, zero_division=0)
]

#F1 
plt.figure(figsize=(8,5))
plt.bar(models, f1, color="pink")
plt.title("F1 Score Comparison")
plt.ylabel("F1 Score")
plt.ylim(0, 1)
plt.show()

#precision 
plt.figure(figsize=(8,5))
plt.bar(models, precision, color="pink")
plt.title("Precision Comparison")
plt.ylabel("Precision")
plt.ylim(0, 1)
plt.show()

#accuracy
plt.figure(figsize=(8,5))
plt.bar(models, accuracy, color="pink")
plt.title("Accuracy Comparison")
plt.ylabel("Accuracy")
plt.ylim(0, 1)
plt.show()
