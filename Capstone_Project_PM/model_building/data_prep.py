# Script for preproceesing the dataset

# For data manipulation
import pandas as pd
import sklearn

# For creating folders
import os

# For data preprocessing and pipeline creation
from sklearn.model_selection import train_test_split

# For encoding categorical values
from sklearn.preprocessing import LabelEncoder

# For connecting to HuggingFace space
from huggingface_hub import login, HfApi

# Constants for the dataset and output paths
api = HfApi(token=os.getenv("HF_TOKEN"))              #HF_TOKEN is stored in the GitHub secrets
DATASET_PATH = "hf://datasets/KavinPrasathK/Engine_Condition_Prediction/engine_data.csv"
df = pd.read_csv(DATASET_PATH)

print("Dataset loaded successfully.")

# Target column
target_col = 'Engine condition'

# Split into X (features) and y (target)
X = df.drop(columns=[target_col])
y = df[target_col]

# Perform train-test split
Xtrain, Xtest, ytrain, ytest = train_test_split(
    X, y, test_size=0.3, random_state=42
)

Xtrain.to_csv("Xtrain.csv",index=False)
Xtest.to_csv("Xtest.csv",index=False)
ytrain.to_csv("ytrain.csv",index=False)
ytest.to_csv("ytest.csv",index=False)


files = ["Xtrain.csv","Xtest.csv","ytrain.csv","ytest.csv"]

# Upload the split dataset to HuggingFace space
for file_path in files:
    api.upload_file(
        path_or_fileobj=file_path,
        path_in_repo=file_path.split("/")[-1],
        repo_id="KavinPrasathK/Engine_Condition_Prediction",
        repo_type="dataset",
    )
