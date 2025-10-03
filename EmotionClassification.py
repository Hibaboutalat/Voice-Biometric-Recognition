import os
import warnings
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import librosa
import librosa.display
from IPython.display import Audio
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout
import joblib  # Import joblib for saving models

warnings.filterwarnings('ignore')

# Set matplotlib backend to avoid backend issues
import matplotlib
matplotlib.use('Agg')

# Initialize lists for paths and labels
paths = []
labels = []

# Define the directory path to your dataset
dataset_directory = r'C:\Users\HP\Desktop\pfe2\project\dataset\emotions\tess'

# Walk through the dataset directory and collect paths and labels
for dirname, _, filenames in os.walk(dataset_directory):
    for filename in filenames:
        file_path = os.path.join(dirname, filename)
        paths.append(file_path)
        
        # Derive the label from the filename
        label = filename.split('_')[-1].split('.')[0].lower()
        labels.append(label)
        
        # Limit the number of paths to 2800
        if len(paths) == 2800:
            break
    if len(paths) == 2800:
        break

print('Dataset is Loaded')

# Create a dataframe
df = pd.DataFrame()
df['speech'] = paths
df['label'] = labels
print(df.head())  # Print the first 5 rows of the dataframe

# Print the value counts of labels
print(df['label'].value_counts())

# Plot the count of each label using seaborn
plt.figure(figsize=(10, 6))
sns.countplot(data=df, x='label')
plt.title('Count of Each Label')
plt.xlabel('Emotion Label')
plt.ylabel('Count')
plt.xticks(rotation=45)
plt.savefig('label_count.png')  # Save the figure instead of showing it
plt.close()  # Close the figure to free up memory

# Function to plot waveplot
def waveplot(data, sr, emotion):
    plt.figure(figsize=(10, 4))
    plt.title(emotion, size=20)
    librosa.display.waveshow(data, sr=sr)
    plt.savefig(f'waveplot_{emotion}.png')  # Save the figure instead of showing it
    plt.close()  # Close the figure to free up memory

# Function to plot spectrogram
def spectrogram(data, sr, emotion):
    x = librosa.stft(data)
    xdb = librosa.amplitude_to_db(abs(x))
    plt.figure(figsize=(11, 4))
    plt.title(emotion, size=20)
    librosa.display.specshow(xdb, sr=sr, x_axis='time', y_axis='hz')
    plt.colorbar()
    plt.savefig(f'spectrogram_{emotion}.png')  # Save the figure instead of showing it
    plt.close()  # Close the figure to free up memory

# Choose emotions and get paths
emotions = ['fear', 'angry', 'disgust', 'neutral', 'sad', 'ps', 'happy']
for emotion in emotions:
    path = np.array(df['speech'][df['label'] == emotion])[0]
    data, sampling_rate = librosa.load(path)
    waveplot(data, sampling_rate, emotion)
    spectrogram(data, sampling_rate, emotion)
    Audio(path)

# Function to extract MFCC features
def extract_mfcc(filename, max_length=100):
    y, sr = librosa.load(filename, duration=3, offset=0.5)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)  # Extract 40 MFCC coefficients
    # Pad or truncate the MFCC feature arrays to a fixed length
    if mfcc.shape[1] < max_length:
        mfcc = np.pad(mfcc, ((0, 0), (0, max_length - mfcc.shape[1])), mode='constant')
    elif mfcc.shape[1] > max_length:
        mfcc = mfcc[:, :max_length]
    return mfcc

# Extract MFCC features for all files
max_length = 100  # Adjust as needed
X_mfcc = df['speech'].apply(lambda x: extract_mfcc(x, max_length=max_length))
X = np.array(X_mfcc.tolist())

# Encode labels
enc = OneHotEncoder()
y = enc.fit_transform(df[['label']]).toarray()

# Build the model
model = Sequential([
    LSTM(256, return_sequences=False, input_shape=(40, max_length)),
    Dropout(0.2),
    Dense(128, activation='relu'),
    Dropout(0.2),
    Dense(64, activation='relu'),
    Dropout(0.2),
    Dense(7, activation='softmax')
])

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()

# Train the model
history = model.fit(X, y, validation_split=0.2, epochs=50, batch_size=64)

# Plot training history
epochs = list(range(1, 51))  # Adjusted to match range for plotting
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

plt.plot(epochs, acc, label='train accuracy')
plt.plot(epochs, val_acc, label='val accuracy')
plt.xlabel('epochs')
plt.ylabel('accuracy')
plt.legend()
plt.savefig('training_accuracy.png')  # Save the figure instead of showing it
plt.close()  # Close the figure to free up memory

loss = history.history['loss']
val_loss = history.history['val_loss']

plt.plot(epochs, loss, label='train loss')
plt.plot(epochs, val_loss, label='val loss')
plt.xlabel('epochs')
plt.ylabel('loss')
plt.legend()
plt.savefig('training_loss.png')  # Save the figure instead of showing it
plt.close()  # Close the figure to free up memory

# Save the trained emotion classification model using the recommended format
model.save('C:\\Users\\HP\\Desktop\\pfe2\\project\\emotion_model.keras')
