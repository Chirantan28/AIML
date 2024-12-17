import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
import joblib

# Load and preprocess the data
data = pd.read_csv("multi_modal_data.csv")

# Encode categorical variables
label_encoders = {}
for col in ["Gender", "Ethnicity"]:
    le = LabelEncoder()
    data[col] = le.fit_transform(data[col])
    label_encoders[col] = le

# Normalize numerical features
numerical_cols = [col for col in data.columns if data[col].dtype in ["float64", "int64"]]
scaler = StandardScaler()
data[numerical_cols] = scaler.fit_transform(data[numerical_cols])

# Prepare features (X) and target (y)
X = data.drop(columns=["Has_Diabetes"])
y = data["Has_Diabetes"]

# Ensure that the target is binary (0 or 1)
y = y.astype(int)

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the Random Forest model
rf_model = RandomForestClassifier(random_state=42)
rf_model.fit(X_train, y_train)

# Save the trained model and the feature matrix (X)
joblib.dump(rf_model, 'random_forest_model.pkl')
X.to_csv('features.csv', index=False)

# Evaluate the model
print("Training complete. Model and features saved.")
