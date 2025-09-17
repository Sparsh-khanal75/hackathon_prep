# train_model.py
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib

# Load Iris dataset
iris = load_iris()
X, y = iris.data, iris.target

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Test accuracy
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model accuracy: {accuracy:.2f}")

# Save the model
joblib.dump(model, 'iris_model.pkl')
print("Model saved as 'iris_model.pkl'")

# Save feature names and target names for reference
import json
model_info = {
    'feature_names': iris.feature_names,
    'target_names': iris.target_names.tolist(),
    'accuracy': accuracy
}
with open('model_info.json', 'w') as f:
    json.dump(model_info, f)