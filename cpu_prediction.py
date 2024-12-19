import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset (replace 'cpu_performance_dataset.csv' with your actual dataset file)
df = pd.read_csv('cpu_performance.csv')

# Check the first few rows of the dataset
print(df.head())

# Preprocessing
# Encode the target variable (Performance Class) using LabelEncoder
label_encoder = LabelEncoder()
df['Performance Class'] = label_encoder.fit_transform(df['Performance Class'])

# Split the dataset into features (X) and target variable (y)
X = df.drop(columns=['Performance Class'])
y = df['Performance Class']

# Split the dataset into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the classifiers
svm = SVC(kernel='linear', random_state=42)
rf = RandomForestClassifier(n_estimators=100, random_state=42)
knn = KNeighborsClassifier(n_neighbors=5)

# Train the classifiers
svm.fit(X_train, y_train)
rf.fit(X_train, y_train)
knn.fit(X_train, y_train)

# Make predictions
y_pred_svm = svm.predict(X_test)
y_pred_rf = rf.predict(X_test)
y_pred_knn = knn.predict(X_test)

# Evaluate the classifiers using accuracy
accuracy_svm = accuracy_score(y_test, y_pred_svm)
accuracy_rf = accuracy_score(y_test, y_pred_rf)
accuracy_knn = accuracy_score(y_test, y_pred_knn)

print(f"Accuracy of SVM: {accuracy_svm:.4f}")
print(f"Accuracy of Random Forest: {accuracy_rf:.4f}")
print(f"Accuracy of KNN: {accuracy_knn:.4f}")

# Print classification reports in the terminal
print("\nClassification Report for SVM:")
print(classification_report(y_test, y_pred_svm, target_names=label_encoder.classes_))

print("\nClassification Report for Random Forest:")
print(classification_report(y_test, y_pred_rf, target_names=label_encoder.classes_))

print("\nClassification Report for KNN:")
print(classification_report(y_test, y_pred_knn, target_names=label_encoder.classes_))

# Create a figure with subplots (2 rows, 2 columns) with a smaller size
fig, axs = plt.subplots(2, 2, figsize=(10, 8))

# Accuracy Comparison Plot (in the first subplot)
accuracies = [accuracy_svm, accuracy_rf, accuracy_knn]
models = ['SVM', 'Random Forest', 'KNN']

axs[0, 0].bar(models, accuracies, color=['skyblue', 'orange', 'green'])
axs[0, 0].set_title('Accuracy Comparison of Classifiers', fontsize=10)
axs[0, 0].set_xlabel('Models', fontsize=8)
axs[0, 0].set_ylabel('Accuracy', fontsize=8)
axs[0, 0].set_ylim(0, 1)

# Confusion Matrix Plotting Function (for each model)
def plot_confusion_matrix(y_true, y_pred, model_name, ax):
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_, ax=ax)
    ax.set_title(f'Confusion Matrix - {model_name}', fontsize=10)
    ax.set_xlabel('Predicted', fontsize=8)
    ax.set_ylabel('Actual', fontsize=8)

# Plot confusion matrices for each model in the remaining subplots
plot_confusion_matrix(y_test, y_pred_svm, 'SVM', axs[0, 1])
plot_confusion_matrix(y_test, y_pred_rf, 'Random Forest', axs[1, 0])
plot_confusion_matrix(y_test, y_pred_knn, 'KNN', axs[1, 1])

# Adjust layout for better spacing and smaller font sizes
plt.tight_layout(pad=3.0)

# Show the combined plot (Accuracy and Confusion Matrices)
plt.show()
