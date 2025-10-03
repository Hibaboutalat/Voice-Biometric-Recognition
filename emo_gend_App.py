import pyaudio
import wave
import numpy as np
import librosa
import joblib
import matplotlib.pyplot as plt
from tkinter import filedialog, messagebox  # Import filedialog and messagebox from tkinter
from keras.models import load_model
import customtkinter as ctk

# Function for visualizing the audio waveform and spectrogram
def visualize_audio(y, sr):
    plt.figure(figsize=(12, 6))

    # Waveform
    plt.subplot(2, 1, 1)
    librosa.display.waveshow(y, sr=sr)
    plt.title('Waveform')

    # Spectrogram
    plt.subplot(2, 1, 2)
    librosa.display.specshow(librosa.amplitude_to_db(np.abs(librosa.stft(y))), sr=sr, x_axis='time', y_axis='log')
    plt.title('Spectrogram')

    plt.tight_layout()

    # Save the figures as images
    waveform_path = "waveform.png"
    spectrogram_path = "spectrogram.png"
    plt.savefig(waveform_path)
    plt.savefig(spectrogram_path)
    plt.close()

    return waveform_path, spectrogram_path

# Constants for audio recording
FRAMES_PER_BUFFER = 3200
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 48000
RECORDING_SECONDS = 5

# Global variables for model paths
gender_model_path = ""
emotion_model_path = ""
scaler_path = "C:\\Users\\HP\\Desktop\\pfe2\\project_emo_gend\\scaler.joblib"
label_encoder_emotion_path = "C:\\Users\\HP\\Desktop\\pfe2\\project_emo_gend\\label_encoder_emotion.joblib"

# Function to select directory for gender or emotion model
def select(textbox, type):
    directory = filedialog.askdirectory()
    if directory:
        textbox.set(directory)
        if type == 'gender':
            global gender_model_path
            gender_model_path = directory
        elif type == 'emotion':
            global emotion_model_path
            emotion_model_path = directory

# Function to record audio and predict gender and emotion
def record_and_predict(textbox_results, progress_bar, progress_label, image_label, gender_model_path, emotion_model_path):
    if not gender_model_path or not emotion_model_path:
        messagebox.showerror("Error", "Please select both gender and emotion model directories.")
        return
    
    # Record audio sample (replace this with your audio recording logic)
    p = pyaudio.PyAudio()
    stream = p.open(
        format=FORMAT,
        channels=CHANNELS,
        rate=RATE,
        input=True,
        frames_per_buffer=FRAMES_PER_BUFFER
    )

    print("Start recording")
    frames = []
    for _ in range(0, int(RATE / FRAMES_PER_BUFFER * RECORDING_SECONDS)):
        data = stream.read(FRAMES_PER_BUFFER)
        frames.append(data)

    stream.stop_stream()
    stream.close()
    p.terminate()
    print('Stop recording')

    # Save audio sample to a WAV file
    output_wav_file = "Output.wav"
    with wave.open(output_wav_file, "wb") as obj:
        obj.setnchannels(CHANNELS)
        obj.setsampwidth(p.get_sample_size(FORMAT))
        obj.setframerate(RATE)
        obj.writeframes(b"".join(frames))

    # Load the recorded audio sample
    y, sr = librosa.load(output_wav_file, sr=None)  # Set sr=None to use original sample rate

    # Visualize the recorded audio and get paths to images
    waveform_path, spectrogram_path = visualize_audio(y, sr)

    # Update image label with waveform and spectrogram images
    waveform_image = ctk.CTkImage(waveform_path)
    spectrogram_image = ctk.CTkImage(spectrogram_path)
    image_label.config(image=waveform_image)
    image_label.image = waveform_image  # Keep a reference

    # Feature extraction with 18 MFCC coefficients
    mfcc = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=18).T, axis=0)
    features = mfcc.reshape(1, -1)  # Reshape to be compatible with scikit-learn

    # Load the trained models, scaler, and label encoders
    gender_model = joblib.load(gender_model_path)
    emotion_model = load_model(emotion_model_path)
    scaler = joblib.load(scaler_path)
    label_encoder_emotion = joblib.load(label_encoder_emotion_path)

    # Standardize the features
    features = scaler.transform(features)

    # Predict the gender
    gender_prediction = gender_model.predict(features)
    predicted_gender = 'Male' if gender_prediction[0] == 1 else 'Female'

    # Predict the emotion
    emotion_prediction = emotion_model.predict(features)
    predicted_emotion_index = np.argmax(emotion_prediction, axis=1).reshape(-1, 1)  # Reshape to 2D array
    predicted_emotion = label_encoder_emotion.inverse_transform(predicted_emotion_index)[0]

    # Update the results textbox
    textbox_results.delete(1.0, ctk.END)
    textbox_results.insert(ctk.END, f"Predicted gender: {predicted_gender}\nPredicted emotion: {predicted_emotion}")

    # Update the progress bar and label
    progress_bar.set(100)
    progress_label.config(text='Accuracy: 100%')

# Setting appearance and color theme
ctk.set_appearance_mode('light')  # Set light mode
ctk.set_default_color_theme('blue')  # Set a default color theme

# Define custom colors
custom_colors = {
    "frame_bg_color": "#FFFFFF",    # White
    "widget_bg_color": "#F5F5F5",   # Light Grey
    "button_bg_color": "#1DA1F2",   # Blue
    "button_text_color": "#FFFFFF", # White
    "text_color": "#000000",        # Black
    "progress_color": "#0E6387"    # Blue
}

# Creating the main application window
app = ctk.CTk()
app.geometry('800x600')
app.title('Speaker Recognition')

# Frame for file selection of gender and emotion models
frame_file_select = ctk.CTkFrame(app, width=760, height=150, corner_radius=15, fg_color=custom_colors["frame_bg_color"])
frame_file_select.place(x=20, y=10)

# Text fields and buttons for selecting model directories
label_model_gender = ctk.CTkLabel(frame_file_select, text='Gender Model Directory:', font=ctk.CTkFont(family='calibre', size=17, weight='bold'), text_color=custom_colors["text_color"])
label_model_gender.place(x=30, y=10)
textbox_model_gender = ctk.CTkTextbox(frame_file_select, width=500, height=30, font=ctk.CTkFont(family='calibre', size=14), fg_color=custom_colors["widget_bg_color"], text_color=custom_colors["text_color"])
textbox_model_gender.place(x=30, y=50)
button_model_gender = ctk.CTkButton(frame_file_select, text='Select Gender Model Directory', width=250, height=40, corner_radius=15, font=ctk.CTkFont(family='calibre', size=15, weight='bold'), fg_color=custom_colors["button_bg_color"], text_color=custom_colors["button_text_color"], command=lambda: select(textbox_model_gender, 'gender'))
button_model_gender.place(x=550, y=50)

label_model_emotion = ctk.CTkLabel(frame_file_select, text='Emotion Model Directory:', font=ctk.CTkFont(family='calibre', size=17, weight='bold'), text_color=custom_colors["text_color"])
label_model_emotion.place(x=30, y=100)
textbox_model_emotion = ctk.CTkTextbox(frame_file_select, width=500, height=30, font=ctk.CTkFont(family='calibre', size=14), fg_color=custom_colors["widget_bg_color"], text_color=custom_colors["text_color"])
textbox_model_emotion.place(x=30, y=140)
button_model_emotion = ctk.CTkButton(frame_file_select, text='Select Emotion Model Directory', width=250, height=40, corner_radius=15, font=ctk.CTkFont(family='calibre', size=15, weight='bold'), fg_color=custom_colors["button_bg_color"], text_color=custom_colors["button_text_color"], command=lambda: select(textbox_model_emotion, 'emotion'))
button_model_emotion.place(x=550, y=140)

# Frame for displaying prediction results
frame_results = ctk.CTkFrame(app, width=760, height=400, corner_radius=15, fg_color=custom_colors["frame_bg_color"])
frame_results.place(x=20, y=180)

label_results = ctk.CTkLabel(frame_results, text='Results:', font=ctk.CTkFont(family='cal
ibre', size=17, weight='bold'), text_color=custom_colors["text_color"])
label_results.place(x=30, y=10)

textbox_results = ctk.CTkTextbox(frame_results, width=700, height=200, font=ctk.CTkFont(family='calibre', size=17), fg_color=custom_colors["widget_bg_color"], text_color=custom_colors["text_color"])
textbox_results.place(x=30, y=40)

progress_bar = ctk.CTkProgressBar(frame_results, width=600, height=20, fg_color=custom_colors["widget_bg_color"], progress_color=custom_colors["progress_color"])
progress_bar.place(x=30, y=260)
progress_bar.set(0)  # Initialize progress bar to 0

progress_label = ctk.CTkLabel(frame_results, text='Accuracy: 0%', font=ctk.CTkFont(family='calibre', size=17, weight='bold'), text_color=custom_colors["text_color"])
progress_label.place(x=640, y=260)

image_label = ctk.CTkLabel(frame_results, text='', fg_color=custom_colors["frame_bg_color"])
image_label.place(x=30, y=290)

button_predict = ctk.CTkButton(frame_results, text='Start Prediction', width=200, height=50, corner_radius=15, font=ctk.CTkFont(family='calibre', size=17, weight='bold'), fg_color=custom_colors["button_bg_color"], text_color=custom_colors["button_text_color"], command=lambda: record_and_predict(textbox_results, progress_bar, progress_label, image_label, textbox_model_gender.get(), textbox_model_emotion.get()))
button_predict.place(x=280, y=350)

# Frame for displaying waveform and spectrogram
frame_visualization = ctk.CTkFrame(app, width=760, height=150, corner_radius=15, fg_color=custom_colors["frame_bg_color"])
frame_visualization.place(x=20, y=600)

label_visualization = ctk.CTkLabel(frame_visualization, text='Audio Visualization', font=ctk.CTkFont(family='calibre', size=17, weight='bold'), text_color=custom_colors["text_color"])
label_visualization.place(x=30, y=10)

waveform_label = ctk.CTkLabel(frame_visualization, text='Waveform:', font=ctk.CTkFont(family='calibre', size=15, weight='bold'), text_color=custom_colors["text_color"])
waveform_label.place(x=30, y=50)

spectrogram_label = ctk.CTkLabel(frame_visualization, text='Spectrogram:', font=ctk.CTkFont(family='calibre', size=15, weight='bold'), text_color=custom_colors["text_color"])
spectrogram_label.place(x=30, y=100)

waveform_image_label = ctk.CTkLabel(frame_visualization, text='', fg_color=custom_colors["frame_bg_color"])
waveform_image_label.place(x=150, y=50)

spectrogram_image_label = ctk.CTkLabel(frame_visualization, text='', fg_color=custom_colors["frame_bg_color"])
spectrogram_image_label.place(x=150, y=100)

# Run the application
app.mainloop()
