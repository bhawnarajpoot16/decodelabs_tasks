# AI Project 2 - Data Classification Using AI
# Dataset: Iris Dataset
# Algorithm: Decision Tree Classifier

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib

# -------------------------------
# Step 1: Load Dataset
# -------------------------------
iris = load_iris()

X = iris.data          # Features
y = iris.target        # Target Labels

print("Dataset Loaded Successfully")
print("Number of Samples :", len(X))
print("Number of Features :", X.shape[1])

# -------------------------------
# Step 2: Split Dataset
# -------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

print("\nDataset Split Completed")
print("Training Samples :", len(X_train))
print("Testing Samples :", len(X_test))

# -------------------------------
# Step 3: Create Model
# -------------------------------
model = DecisionTreeClassifier(random_state=42)

# -------------------------------
# Step 4: Train Model
# -------------------------------
model.fit(X_train, y_train)

print("\nModel Training Completed")

# -------------------------------
# Step 5: Predict
# -------------------------------
y_pred = model.predict(X_test)

# -------------------------------
# Step 6: Evaluate Model
# -------------------------------
accuracy = accuracy_score(y_test, y_pred)

print("\n========== RESULT ==========")
print("Accuracy :", round(accuracy * 100, 2), "%")

print("\nClassification Report")
print(classification_report(y_test, y_pred))

print("Confusion Matrix")
print(confusion_matrix(y_test, y_pred))

# -------------------------------
# Step 7: Save Model
# -------------------------------
joblib.dump(model, "iris_model.pkl")

print("\nModel Saved Successfully as iris_model.pkl")

# -------------------------------
# Step 8: Predict New Flower
# -------------------------------
sample = [[5.1, 3.5, 1.4, 0.2]]

prediction = model.predict(sample)

print("\nPrediction for Sample:", prediction[0])
print("Flower Name:", iris.target_names[prediction][0])
