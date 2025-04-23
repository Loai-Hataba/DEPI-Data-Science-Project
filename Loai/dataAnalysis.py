import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report , confusion_matrix, ConfusionMatrixDisplay


data = pd.read_csv("schizophrenia_dataset.csv")

print(data.shape)

print(data.isnull().sum().sum())

print(data.describe())

print(data.duplicated().sum())

plt.figure(figsize=(10,6))
sns.heatmap(data.corr(), annot=True, cmap="coolwarm", fmt=".2f")
plt.show()

for column in data.columns:
    print(f"\n{data[column].value_counts()}")

for column in data.columns:
    print(f"Unique values in {column}: {data[column].unique()}")

# Check for class imbalance
print("\nTarget value distribution:")
print(data["Diagnosis"].value_counts(normalize=True)) 



# Drop Patient_ID as it is not a feature
data = data.drop(columns=["Patient_ID"])

# Split features and target variable
X = data.drop(columns=["Diagnosis"])
y = data["Diagnosis"]

# Add slight noise to make synthetic data more realistic
x_noisy = X.copy()
noise_factor = 0.01
for col in x_noisy.select_dtypes(include='number').columns:
    x_noisy[col] += noise_factor * np.random.normal(size=x_noisy.shape[0])

#  add noise to Y
y_noisy = y.copy()
flip_indices = np.random.choice(y.index, size=int(0.05 * len(y)), replace=False)
y_noisy[flip_indices] = 1 - y_noisy[flip_indices]


# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(x_noisy, y_noisy, test_size=0.2, random_state=42, stratify=y)

# Add class_weight='balanced' if class imbalance exists
model = RandomForestClassifier(n_estimators=100, max_depth=5,random_state=42, class_weight='balanced')
model.fit(X_train, y_train)


# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

print(f"Model Accuracy: {accuracy:.4f}")
print("Classification Report:\n", report)

# Confusion Matrix
ConfusionMatrixDisplay.from_estimator(model, X_test, y_test)
plt.title("Confusion Matrix")
plt.show()

# Feature Importances
importances = model.feature_importances_
indices = np.argsort(importances)[::-1]
features = x_noisy.columns
top_indices = np.argsort(importances)[::-1][:5]
top_features = x_noisy.columns[top_indices]
print("Top 5 Features:", list(top_features))

plt.figure(figsize=(10, 6))
sns.barplot(x=importances[indices], y=features[indices])
plt.title("Feature Importances")
plt.xlabel("Importance")
plt.ylabel("Feature")
plt.tight_layout()
plt.show()

# Cross-Validation Accuracy
cv_scores = cross_val_score(model, x_noisy, y_noisy, cv=5, scoring='accuracy')
print(f"\nCross-validated Accuracy (5-fold): {cv_scores.mean():.4f}")


# checking for data leakage
correlations = data.corr()
print(correlations["Diagnosis"].sort_values(ascending=False))

#Drop Diagnosis Column
data = data.drop(columns=["Diagnosis"])

#trying a simpler model
from sklearn.linear_model import LogisticRegression

model_lr = LogisticRegression(max_iter=1000)
model_lr.fit(X_train, y_train)
y_pred_lr = model_lr.predict(X_test)

print("\n--- Logistic Regression Results ---")
print(f"Accuracy: {accuracy_score(y_test, y_pred_lr):.4f}")
print("Classification Report:\n", classification_report(y_test, y_pred_lr))


# Learning curve
from sklearn.model_selection import learning_curve

train_sizes, train_scores, test_scores = learning_curve(
    model, x_noisy, y, cv=5, scoring='accuracy', n_jobs=-1,
    train_sizes=np.linspace(0.1, 1.0, 5)
)

train_mean = np.mean(train_scores, axis=1)
test_mean = np.mean(test_scores, axis=1)

plt.figure(figsize=(10, 6))
plt.plot(train_sizes, train_mean, label="Training Score")
plt.plot(train_sizes, test_mean, label="Validation Score")
plt.xlabel("Training Set Size")
plt.ylabel("Accuracy")
plt.title("Learning Curve")
plt.legend()
plt.grid()
plt.show()


train_sizes_lr, train_scores_lr, test_scores_lr = learning_curve(
    model_lr, x_noisy, y, cv=5, scoring='accuracy', n_jobs=-1,
    train_sizes=np.linspace(0.1, 1.0, 5)
)

train_mean_lr = np.mean(train_scores, axis=1)
test_mean_lr = np.mean(test_scores, axis=1)

plt.figure(figsize=(10, 6))
plt.plot(train_sizes_lr, train_mean_lr, label="Training Score")
plt.plot(train_sizes_lr, test_mean_lr, label="Validation Score")
plt.xlabel("Training Set Size")
plt.ylabel("Accuracy")
plt.title("Learning Curve")
plt.legend()
plt.grid()
plt.show()
