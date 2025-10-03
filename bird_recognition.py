# Import necessary libraries
import os  # Provides functions to interact with the operating system
import pygame  # Library for creating multimedia applications
import tensorflow as tf  # Deep learning library
import customtkinter  # CustomTkinter for creating modern GUI applications
from itertools import groupby  # Provides functions to group data
from tkinter import filedialog  # Module for file dialog boxes
from PIL import Image, ImageTk  # Python Imaging Library for image processing
from audio_classification_module import build_model, load_wav_16k_mono, preprocess_mp3, check_file_exists, save_results_to_csv  # Custom audio classification functions

# Function to predict audio calls
def predict_audio(file_path, model):
    if isinstance(file_path, tf.Tensor):
        file_path = file_path.numpy().decode('utf-8')
    
    # Load and preprocess the audio file
    wav = load_wav_16k_mono(file_path)
    audio_slices = tf.keras.utils.timeseries_dataset_from_array(wav, wav, sequence_length=48000, sequence_stride=48000, batch_size=1)
    audio_slices = audio_slices.map(preprocess_mp3)
    audio_slices = audio_slices.batch(64)

    # Make predictions using the model
    yhat = model.predict(audio_slices)
    yhat = [1 if prediction > 0.5 else 0 for prediction in yhat]
    yhat = [key for key, group in groupby(yhat)]
    calls = tf.math.reduce_sum(yhat).numpy()
    
    return calls > 0, calls

# Function to handle a single prediction
def main_single_prediction(audio_file, model):
    check_file_exists(audio_file)

    calls_detected, num_calls = predict_audio(audio_file, model)
    accuracy = 0.95  # Example accuracy value as a decimal
    return calls_detected, num_calls, accuracy

# Function to select a file using file dialog
def select_file(textbox, type='audio'):
    filetypes = (("wav files", "*.wav"), ("mp3 files", "*.mp3")) if type == 'audio' else (("h5 files", "*.h5"),)
    file_path = filedialog.askopenfilename(title="Select File", filetypes=filetypes)
    if file_path:
        textbox.delete(0, customtkinter.END)
        textbox.insert(0, file_path)
    return file_path

# Function to start the prediction process
def start_prediction(audio_path, model_path, result_textbox, progress_bar, progress_label, image_label):
    if not audio_path or not model_path:
        result_textbox.delete(0.0, customtkinter.END)
        result_textbox.insert(customtkinter.END, "Please select both an audio file and a model file")
        return
    
    model = build_model()
    model.load_weights(model_path)

    calls_detected, num_calls, accuracy = main_single_prediction(audio_path, model)
    result_textbox.delete(0.0, customtkinter.END)
    result_textbox.insert(customtkinter.END, f"There are Capuchin bird calls: {'Yes' if calls_detected else 'No'}\nNumber of calls detected: {num_calls}")
    progress_bar.set(accuracy)
    progress_label.configure(text=f"Accuracy: {accuracy*100:.2f}%")

    # Display the corresponding image based on the prediction
    image_path = "C:\\Users\\HP\\Desktop\\pfe2\\AudioClassification\\capuchinbird_exists.jpg" if calls_detected else "C:\\Users\\HP\\Desktop\\pfe2\\AudioClassification\\capuchinbird_not_exists.jpg"
    image = Image.open(image_path)
    image = image.resize((150, 150), Image.LANCZOS)
    capuchin_image = ImageTk.PhotoImage(image)

    image_label.configure(image=capuchin_image)
    image_label.image = capuchin_image

# Function to play audio
def play_audio(audio_path):
    if audio_path:
        pygame.mixer.init()
        pygame.mixer.music.load(audio_path)
        pygame.mixer.music.play()

# Function to stop audio playback
def stop_audio():
    pygame.mixer.music.stop()

# Set appearance mode and color theme for the GUI
customtkinter.set_appearance_mode('light')  
customtkinter.set_default_color_theme('blue')

# Initialize the main application window
app = customtkinter.CTk()
app.geometry('780x700')
app.title('Capuchin Call Detection')

# Define custom colors
custom_colors = {
    "frame_bg_color": "#FFFFFF",
    "widget_bg_color": "#F5F5F5",
    "button_bg_color": "#1DA1F2",
    "button_text_color": "#FFFFFF",
    "text_color": "#000000",
    "progress_color": "#1DA1F2"
}

# Create a frame for file selection
frame_file_select = customtkinter.CTkFrame(app, width=760, height=200, corner_radius=15, fg_color=custom_colors["frame_bg_color"])
frame_file_select.place(x=10, y=20)

# Add label, textbox, and button for audio file selection
label_audio = customtkinter.CTkLabel(frame_file_select, text='Audio file:', font=customtkinter.CTkFont(family='calibre', size=17, weight='bold'), text_color=custom_colors["text_color"])
label_audio.place(x=30, y=5)
textbox_audio = customtkinter.CTkEntry(frame_file_select, width=500, font=customtkinter.CTkFont(family='calibre', size=17), fg_color=custom_colors["widget_bg_color"], text_color=custom_colors["text_color"])
textbox_audio.place(x=30, y=40)
button_audio = customtkinter.CTkButton(frame_file_select, text='Select Audio File', width=200, height=40, corner_radius=15, font=customtkinter.CTkFont(family='calibre', size=15, weight='bold'), fg_color=custom_colors["button_bg_color"], text_color=custom_colors["button_text_color"], command=lambda: select_file(textbox_audio, 'audio'))
button_audio.place(x=550, y=40)

# Add label, textbox, and button for model file selection
label_model = customtkinter.CTkLabel(frame_file_select, text='Model file:', font=customtkinter.CTkFont(family='calibre', size=17, weight='bold'), text_color=custom_colors["text_color"])
label_model.place(x=30, y=85)
textbox_model = customtkinter.CTkEntry(frame_file_select, width=500, font=customtkinter.CTkFont(family='calibre', size=17), fg_color=custom_colors["widget_bg_color"], text_color=custom_colors["text_color"])
textbox_model.place(x=30, y=120)
button_model = customtkinter.CTkButton(frame_file_select, text='Select Model File', width=200, height=40, corner_radius=15, font=customtkinter.CTkFont(family='calibre', size=15, weight='bold'), fg_color=custom_colors["button_bg_color"], text_color=custom_colors["button_text_color"], command=lambda: select_file(textbox_model, 'model'))
button_model.place(x=550, y=120)

# Create a frame for audio controls
frame_audio_controls = customtkinter.CTkFrame(app, width=760, height=80, corner_radius=15, fg_color=custom_colors["frame_bg_color"])
frame_audio_controls.place(x=10, y=230)

# Load play and stop icons
play_image = customtkinter.CTkImage(light_image=Image.open("C:\\Users\\HP\\Desktop\\pfe2\\pynnote_vr\\play_183665.png"), size=(20, 20))
stop_image = customtkinter.CTkImage(light_image=Image.open("C:\\Users\\HP\\Desktop\\pfe2\\pynnote_vr\\stop_167111.png"), size=(20, 20))

# Add buttons for audio controls with icons
button_play = customtkinter.CTkButton(frame_audio_controls, text='Play Audio', width=200, height=40, corner_radius=15, font=customtkinter.CTkFont(family='calibre', size=15, weight='bold'), fg_color=custom_colors["button_bg_color"], text_color=custom_colors["button_text_color"], command=lambda: play_audio(textbox_audio.get()), image=play_image)
button_play.place(x=180, y=20)

button_stop = customtkinter.CTkButton(frame_audio_controls, text='Stop Audio', width=200, height=40, corner_radius=15, font=customtkinter.CTkFont(family='calibre', size=15, weight='bold'), fg_color=custom_colors["button_bg_color"], text_color=custom_colors["button_text_color"], command=stop_audio , image=stop_image)
button_stop.place(x=380, y=20)

# Create a frame for displaying results
frame_results = customtkinter.CTkFrame(app, width=760, height=300, corner_radius=15, fg_color=custom_colors["frame_bg_color"])
frame_results.place(x=10, y=330)

# Add label, textbox, and progress bar for displaying results
label_results = customtkinter.CTkLabel(frame_results, text='Results:', font=customtkinter.CTkFont(family='calibre', size=17, weight='bold'), text_color=custom_colors["text_color"])
label_results.place(x=30, y=5)
textbox_results = customtkinter.CTkTextbox(frame_results, width=700, height=100, font=customtkinter.CTkFont(family='calibre', size=17), fg_color=custom_colors["widget_bg_color"], text_color=custom_colors["text_color"])
textbox_results.place(x=30, y=40)

progress_bar = customtkinter.CTkProgressBar(frame_results, width=600, height=20, fg_color=custom_colors["widget_bg_color"], progress_color=custom_colors["progress_color"])
progress_bar.place(x=30, y=160)
progress_bar.set(0)  # Initialize progress bar to 0

progress_label = customtkinter.CTkLabel(frame_results, text='Accuracy: 0%', font=customtkinter.CTkFont(family='calibre', size=17, weight='bold'), text_color=custom_colors["text_color"])
progress_label.place(x=640, y=160)

# Add an image label to display prediction result images
image_label = customtkinter.CTkLabel(frame_results, text='', fg_color=custom_colors["frame_bg_color"])
image_label.place(x=30, y=190)

# Add a button to start the prediction
button_predict = customtkinter.CTkButton(frame_results, text='Start Prediction', width=200, height=50, corner_radius=15, font=customtkinter.CTkFont(family='calibre', size=17, weight='bold'), fg_color=custom_colors["button_bg_color"], text_color=custom_colors["button_text_color"], command=lambda: start_prediction(textbox_audio.get(), textbox_model.get(), textbox_results, progress_bar, progress_label, image_label))
button_predict.place(x=280, y=250)

# Start the Tkinter event loop to run the GUI application
app.mainloop()
