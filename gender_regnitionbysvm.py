import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
import joblib  # Import joblib for saving models

# Load the dataset
voice_data = pd.read_csv(r"C:\Users\HP\Desktop\pfe2\project\dataset\M_or_F\voice.csv")
print("Dataset loaded successfully")

# Display the first few rows of the dataset
print(voice_data.head())

# Data preprocessing 
# Check for missing values
voice_data_missing_values = voice_data.isnull().sum()
missing_columns = voice_data_missing_values[voice_data_missing_values > 0]

print("Columns with missing values:")
print(missing_columns)

# Data visualization 
# Correlation matrix
voice_data_numeric_columns = voice_data.select_dtypes(include=['float64', 'int64']).columns
correlation_matrix = voice_data[voice_data_numeric_columns].corr()
plt.figure(figsize=(12, 10))
sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)

plt.title('Correlation Heatmap of Voice Features')
plt.xticks(rotation=45)
plt.yticks(rotation=0)
plt.show()

# Dropping 'centroid' and 'dfrange' which are highly correlated
voice_data = voice_data.drop(columns=["centroid", "dfrange"])
print(voice_data.head())

# Separate features and labels
X = voice_data.drop(columns=["label"]) 
Y = voice_data["label"]

print("Shape of X:", X.shape)
print("Shape of Y:", Y.shape)

print(X.head())
print(Y.head())

# Data Standardisation
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Encoding of labeled data
label_encoder = LabelEncoder()
Y_encoded = label_encoder.fit_transform(Y)

print("Original classes:", label_encoder.classes_)
print("Encoded classes:", Y_encoded)

# Data splitting
X_train, X_test, Y_train, Y_test = train_test_split(X, Y_encoded, test_size=0.3, random_state=42)

# SVM Model 
svc = SVC()
svc.fit(X_train, Y_train)
Y_pred = svc.predict(X_test)

# Print accuracy score
print('Accuracy Score:')
print(metrics.accuracy_score(Y_test, Y_pred))

# Ensure the directory exists before saving the model, scaler, and label encoder
save_dir = r'C:\\Users\\HP\\Desktop\\pfe2\\project'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

# Save the trained model, scaler, and label encoder for gender classification
joblib.dump(svc, os.path.join(save_dir, 'gender_model.joblib'))
joblib.dump(scaler, os.path.join(save_dir, 'scaler.joblib'))
joblib.dump(label_encoder, os.path.join(save_dir, 'label_encoder_gender.joblib'))

print(f"Model, scaler, and label encoder saved successfully in {save_dir}")
