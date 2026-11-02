# Ensemble Learning 
# A. Implement Random Forest Classifier model to predict the safety of the car. 
# Dataset link: https://www.kaggle.com/datasets/elikplim/car-evaluation-data-set


# Random Forest Classifier on Car Evaluation Dataset

import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

# Column names of Car Evaluation dataset
columns = [
    "buying", "maint", "doors",
    "persons", "lug_boot", "safety", "class"
]

# Load dataset
df = pd.read_csv("car_evaluation.csv")

# Check and correct column names
df.columns = df.columns.str.strip().str.lower()

# If headers are missing, reload with proper column names
if not set(columns).issubset(df.columns):
    df = pd.read_csv("car_evaluation.csv", header=None, names=columns)

# Separate input and output
X = df.drop("class", axis=1)

y = df["class"]

# Encode categorical input columns
encoders = {}

for col in X.columns:
    encoders[col] = LabelEncoder()
    X[col] = encoders[col].fit_transform(X[col].astype(str))

# Encode target column
target_encoder = LabelEncoder()
y = target_encoder.fit_transform(y)

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Create Random Forest model
model = RandomForestClassifier(
    n_estimators=100, random_state=42
)

# Train model
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Evaluate model
print("Accuracy:", accuracy_score(y_test, y_pred))

print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# Predict a new car
new_car = pd.DataFrame([{
    "buying": "low",
    "maint": "low",
    "doors": "4",
    "persons": "4",
    "lug_boot": "big",
    "safety": "high"
}])

# Apply the same encoders used during training
for col in new_car.columns:
    new_car[col] = encoders[col].transform(
        new_car[col].astype(str)
    )

prediction = model.predict(new_car)

print(
    "Predicted Car Class:",
    target_encoder.inverse_transform(prediction)[0]
)
