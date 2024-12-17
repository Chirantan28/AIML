from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder

# Reload and preprocess the data
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

# Define the parameter distribution for RandomizedSearchCV
param_dist = {
    'n_estimators': [100, 200, 300],
    'max_depth': [10, 20, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'bootstrap': [True, False]
}

# Create Random Forest model
rf_model = RandomForestClassifier(random_state=42)

# Set up RandomizedSearchCV
random_search = RandomizedSearchCV(estimator=rf_model, param_distributions=param_dist, n_iter=100, cv=3, n_jobs=-1, verbose=2, random_state=42)

# Fit randomized search
random_search.fit(X_train, y_train)

# Get the best parameters and score
print("Best Parameters from Random Search:", random_search.best_params_)
print("Best Score from Random Search:", random_search.best_score_)

# Save the best model
import joblib
joblib.dump(random_search.best_estimator_, 'best_random_forest_randomized_model.pkl')

# Evaluate the model
y_pred = random_search.best_estimator_.predict(X_test)
from sklearn.metrics import accuracy_score, classification_report
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))
