import pandas as pd
import numpy as np

# Set a random seed for reproducibility
np.random.seed(42)

# Simulating Genomic Data
num_samples = 100
genomic_data = pd.DataFrame({
    f"GENE{i+1}": np.random.rand(num_samples) for i in range(5)  # 5 genes
})

# Simulating Clinical Data
clinical_data = pd.DataFrame({
    "Age": np.random.randint(20, 80, num_samples),  # Age between 20 and 80
    "BMI": np.random.uniform(18.5, 40, num_samples),  # BMI in a realistic range
    "Has_Hypertension": np.random.choice([0, 1], num_samples),  # Binary flags
    "Has_Diabetes": np.random.choice([0, 1], num_samples)
})

# Simulating Demographic Data
demographic_data = pd.DataFrame({
    "Gender": np.random.choice(["Male", "Female"], num_samples),
    "Ethnicity": np.random.choice(["Caucasian", "Asian", "African-American", "Hispanic"], num_samples)
})

# Integrating Data
# Concatenate all dataframes along the columns
multi_modal_data = pd.concat([genomic_data, clinical_data, demographic_data], axis=1)

# Display the first few rows of the integrated dataset
print(multi_modal_data.head())

# Save to CSV (optional)
multi_modal_data.to_csv("multi_modal_data.csv", index=False)
