import joblib
import pandas as pd
import matplotlib.pyplot as plt

# Load the trained Random Forest model
rf_model = joblib.load('random_forest_model.pkl')

# Load the saved features (X)
X = pd.read_csv('features.csv')

# Get feature importance from the trained model
feature_importances = rf_model.feature_importances_

# Create a DataFrame to map features to their importance values
feature_importance_df = pd.DataFrame({
    'Feature': X.columns,
    'Importance': feature_importances
})

# Sort the DataFrame by importance
feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)

# Plot the feature importance
plt.figure(figsize=(10, 6))
plt.barh(feature_importance_df['Feature'], feature_importance_df['Importance'])
plt.xlabel('Importance')
plt.ylabel('Feature')
plt.title('Feature Importance in Random Forest Model')
plt.show()
