import os
import tensorflow as tf
from itertools import groupby
from audio_classification_module import build_model, load_wav_16k_mono, preprocess_mp3, check_file_exists

def predict_audio(file_path, model):
    # Ensure the file_path is a string
    if isinstance(file_path, tf.Tensor):
        file_path = file_path.numpy().decode('utf-8')
    
    # Load the wav file
    wav = load_wav_16k_mono(file_path)
    
    # Create audio slices
    audio_slices = tf.keras.utils.timeseries_dataset_from_array(wav, wav, sequence_length=48000, sequence_stride=48000, batch_size=1)
    audio_slices = audio_slices.map(preprocess_mp3)
    audio_slices = audio_slices.batch(64)

    # Predict
    yhat = model.predict(audio_slices)
    yhat = [1 if prediction > 0.5 else 0 for prediction in yhat]
    yhat = [key for key, group in groupby(yhat)]
    calls = tf.math.reduce_sum(yhat).numpy()
    return calls

def main_single_prediction(audio_file, model_path):
    # Check if files exist
    check_file_exists(audio_file)
    check_file_exists(model_path)

    # Load the model
    model = build_model()
    model.load_weights(model_path)

    # Predict
    calls_detected = predict_audio(audio_file, model)
    return f"Total Capuchin calls detected in {audio_file}: {calls_detected}"

if __name__ == '__main__':
    # Example usage
    model_path = r'C:\Users\HP\Desktop\pfe2\AudioClassification\model_checkpoint.weights.h5'
    audio_file = r'C:\Users\HP\Desktop\pfe2\AudioClassification\data\Forest Recordings\recording_03.mp3'  # Ensure correct file path and extension

    result = main_single_prediction(audio_file, model_path)
    print(result)
