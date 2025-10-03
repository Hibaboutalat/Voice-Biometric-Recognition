import os
import librosa
import numpy as np
from sklearn.neighbors import KNeighborsClassifier

# Function to extract MFCC features from audio file and ensure fixed length
def extract_features(audio_path, max_length):
    y, sr = librosa.load(audio_path, sr=None)  # Load audio file
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)  # Extract MFCC features
    # Pad or truncate features to the fixed length
    if mfccs.shape[1] < max_length:
        mfccs = np.pad(mfccs, ((0, 0), (0, max_length - mfccs.shape[1])), mode='constant')
    else:
        mfccs = mfccs[:, :max_length]
    return mfccs.flatten()  # Flatten the feature matrix

# Path to your voice dataset
dataset_path = r'C:\Users\HP\Documents\PFE\project\dataset\wiam_voice'

# Find the maximum length of features among all samples
max_length = 0
for filename in os.listdir(dataset_path):
    if filename.endswith('.wav'):
        audio_path = os.path.join(dataset_path, filename)
        y, sr = librosa.load(audio_path, sr=None)
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        max_length = max(max_length, mfccs.shape[1])

# Extract features from each audio file in the dataset
dataset_features = []
for filename in os.listdir(dataset_path):
    if filename.endswith('.wav'):
        audio_path = os.path.join(dataset_path, filename)
        features = extract_features(audio_path, max_length)
        dataset_features.append(features)

# Convert the list of features into a numpy array
X = np.array(dataset_features)

# Assuming you have labels for each audio file (0 for you, 1 for your friend)
y = np.array([0, 0, ...])  # Replace with your actual labels

# Train the KNN classifier
k = 3  # Number of neighbors to consider
knn = KNeighborsClassifier(n_neighbors=k)
knn.fit(X, y)

# Extract features from the recorded audio
recorded_audio_path = r'C:\Users\HP\Documents\PFE\project\Output.wav'
recorded_features = extract_features(recorded_audio_path, max_length)

# Predict the label for the recorded audio
prediction = knn.predict([recorded_features])

# Print the result
if prediction == 0:
    print("This is you, I know you.")
else:
    print("Sorry, I don't know.")
