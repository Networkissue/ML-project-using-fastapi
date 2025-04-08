# retrain_model.py
import pickle
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier

# Load dataset
X, y = load_iris(return_X_y=True)

# Train a new model
model = DecisionTreeClassifier()
model.fit(X, y)

# Save the model to a file
with open('savedmodel.sav', 'wb') as f:
    pickle.dump(model, f)

print("✅ Model saved successfully.")
