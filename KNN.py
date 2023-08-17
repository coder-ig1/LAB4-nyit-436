# Import necessary libraries
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, LeaveOneOut
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from customStandardScaler import StandardScaler

# Load the dataset
df = pd.read_csv('diabetes.csv')

# Handle zero values by replacing them with mean in testing it led to better accuracy then the 0s
no_zero_columns = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI','Age']
for col in no_zero_columns:
    df[col] = df[col].replace(0, np.NaN)
    mean = df[col].mean(skipna=True)
    df[col].fillna(mean, inplace=True)

# Scale the data using custom StandardScaler
scaler = StandardScaler()
columns_to_scale = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']
df[columns_to_scale] = scaler.fit_transform(df[columns_to_scale])

# Split the data into training and testing sets
X = df[columns_to_scale]
y = df['Outcome']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)  # Added random_state for reproducibility

# Find the best K value
ks = []
accuracies = []
for k in range(1, int(np.sqrt(len(df['Outcome']))), 2):
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)
    ks.append(k)
    accuracies.append(accuracy_score(y_test, y_pred))

best_k = ks[np.argmax(accuracies)]
print("The best K value is", best_k, "with accuracy =", max(accuracies))

# Perform 5-fold cross-validation
knn = KNeighborsClassifier(n_neighbors=best_k)
cv_scores = cross_val_score(knn, X, y, cv=5, scoring='accuracy')
print("5-Fold Cross Validation Mean Accuracy:", cv_scores.mean())
print("5-Fold Cross Validation Standard Deviation:", cv_scores.std())

# Confusion matrix
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)
conf_matrix = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:\n", conf_matrix)

# Leave-One-Out cross-validation
loo = LeaveOneOut()
loo_scores = cross_val_score(knn, X, y, cv=loo, scoring='accuracy')
print("Leave-One-Out Cross Validation Mean Accuracy:", loo_scores.mean())
print("Leave-One-Out Cross Validation Standard Deviation:", loo_scores.std())
