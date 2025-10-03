import os
import librosa
import tensorflow as tf
import csv
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Dense, Flatten

def check_file_exists(filepath):
    if not os.path.isfile(filepath):
        raise FileNotFoundError(f"File not found: {filepath}")

def load_wav_16k_mono(filename):
    # Ensure filename is a string
    if isinstance(filename, tf.Tensor):
        filename = filename.numpy().decode('utf-8')
    
    # Load the wav file
    wav, sample_rate = librosa.load(filename, sr=16000, mono=True)
    return wav

def load_wav_tensor(filename):
    wav = tf.py_function(load_wav_16k_mono, [filename], tf.float32)
    wav.set_shape([None])
    return wav

def preprocess(file_path, label):
    wav = load_wav_tensor(file_path)
    wav = wav[:48000]
    zero_padding = tf.zeros([48000] - tf.shape(wav), dtype=tf.float32)
    wav = tf.concat([zero_padding, wav], 0)
    spectrogram = tf.signal.stft(wav, frame_length=320, frame_step=32)
    spectrogram = tf.abs(spectrogram)
    spectrogram = tf.expand_dims(spectrogram, axis=2)
    spectrogram.set_shape([1491, 257, 1])
    return spectrogram, label

def build_model():
    model = Sequential([
        Conv2D(8, (3, 3), activation='relu', input_shape=(1491, 257, 1)),
        Conv2D(8, (3, 3), activation='relu'),
        Flatten(),
        Dense(64, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='Adam', loss='BinaryCrossentropy', metrics=[tf.keras.metrics.Recall(), tf.keras.metrics.Precision()])
    return model

def load_mp3_16k_mono(filename):
    wav, sample_rate = librosa.load(filename, sr=16000, mono=True)
    return wav

def preprocess_mp3(sample, index):
    sample = sample[0]
    zero_padding = tf.zeros([48000] - tf.shape(sample), dtype=tf.float32)
    wav = tf.concat([zero_padding, sample], 0)
    spectrogram = tf.signal.stft(wav, frame_length=320, frame_step=32)
    spectrogram = tf.abs(spectrogram)
    spectrogram = tf.expand_dims(spectrogram, axis=2)
    return spectrogram

def save_results_to_csv(results, filename='results.csv'):
    with open(filename, 'w', newline='') as f:
        writer = csv.writer(f, delimiter=',')
        writer.writerow(['recording', 'capuchin_calls'])
        for key, value in results.items():
            writer.writerow([key, value])
