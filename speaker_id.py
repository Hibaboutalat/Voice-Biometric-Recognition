# Import necessary libraries
import customtkinter
from tkinter import filedialog
from pyannote.audio import Model
from pyannote.audio import Inference
from scipy.spatial.distance import pdist
from PIL import Image
import pygame
import os
import sounddevice as sd
import soundfile as sf
import numpy as np
import tempfile

# Load the pre-trained Pyannote embedding model and create an inference object
model = Model.from_pretrained("pyannote/embedding", use_auth_token="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")
vec = Inference(model, window="whole")

# Define a dictionary mapping speaker IDs to their names
dataset = {"1": "elon musk", "2": "sam altman", "3": "kanye west", "4": "stephen wolfram",
           "5": "liv boeree", "6": "anna frebel", "7": "simone giertz", "8": "shannon curry",
           "9": "ginni rometty", "10": "manolis kellis", "11": "paul rosolie", "12": "robert playter",
           "13": "Guido van Rossum", "14": "wiam makboul", "15": "Hiba Boutalat"}

# Initialize the Pygame mixer for audio playback
pygame.mixer.init()

# Function to select an audio file
def select():
    global test, distance, file_path
    distance = 0.7
    name = "unknown"
    textbox2.delete(0.0, 'end')
    textbox1.delete(0.0, 'end')
    label3.configure(text="0%")
    progress_Bar.set(0)
    progress_bar2.set(0)
    progress_label2.configure(text="0%")

    # Open file dialog to select audio file
    file_path = filedialog.askopenfilename(initialdir=os.path.expanduser("C:\\Users\\HP\\Desktop\\pfe2\\pynnote_vr\\audios"), 
                                           title="Select File", filetypes=(("wav files", "*.wav"), ("mp3 files", "*.mp3")))
    if file_path:
        # Display the selected file path
        textbox1.insert(0.0, file_path)
        pygame.mixer.music.load(file_path)
        # Extract embedding from the selected file
        test = vec(file_path)

# Functions to control audio playback
def play_audio():
    global file_path
    if file_path:
        pygame.mixer.music.play()

def pause_audio():
    pygame.mixer.music.pause()

def stop_audio():
    pygame.mixer.music.stop()

# Function to record audio
def record_audio():
    global test, file_path
    duration = 5
    fs = 44100
    # Create a temporary file to save the recorded audio
    file_path = tempfile.mktemp(prefix="recorded_", suffix=".wav", dir="")

    print("Recording...")
    # Record audio
    audio = sd.rec(int(duration * fs), samplerate=fs, channels=2, dtype='float64')
    sd.wait()
    print("Recording complete.")

    # Save the recorded audio to the temporary file
    sf.write(file_path, audio, fs)
    # Extract embedding from the recorded audio
    test = vec(file_path)
    # Display the file path of the recorded audio
    textbox1.delete(0.0, 'end')
    textbox1.insert(0.0, file_path)

# Function to search for the speaker in the dataset
def search():
    global distance, name
    if test is None:
        textbox2.insert('end', "\nPlease select or record an audio file first")
        return
    
    distance = 0.7
    name = "I don't know the speaker"
    num = len(dataset)
    # Compare the test embedding with embeddings of known speakers
    for i in range(num):
        percent = (i + 1) / num
        a = vec(f"actors/{i + 1}.wav")
        x = [test, a]
        y = pdist(x, metric='cosine')
        # Update the closest matching speaker
        if round(y[0], 4) <= distance:
            distance = round(y[0], 4)
            name = dataset[f"{i + 1}"]
        
        # Update progress bar
        progress_Bar.set(percent)
        label3.configure(text=f"{int(percent * 100)}%")
        label3.update()
    
    # Update accuracy progress bar and display the result
    progress_bar2.set(1 - distance)
    progress_label2.configure(text=f"{int((1 - distance) * 100)}%")
    textbox2.insert('end', "\n" + name)

# Set appearance mode and color theme for the GUI
customtkinter.set_appearance_mode('light')
customtkinter.set_default_color_theme('blue')

# Define custom colors
custom_colors = {
    "frame_bg_color": "#FFFFFF",
    "widget_bg_color": "#F5F5F5",
    "button_bg_color": "#1DA1F2",
    "button_text_color": "#FFFFFF",
    "text_color": "#000000",
    "progress_color": "#0E6387"
}

# Initialize the main application window
app = customtkinter.CTk()
app.geometry('780x780')
app.title('Speaker Recognition')

# Create a frame for selecting or recording an audio file
frame4 = customtkinter.CTkFrame(app, width=760, height=200, corner_radius=15, fg_color=custom_colors["frame_bg_color"])
frame4.place(x=10, y=20)

# Add label, textbox, and buttons for file selection and recording
label1 = customtkinter.CTkLabel(frame4, text='File name or Record:', font=customtkinter.CTkFont(family='calibre', size=17, weight='bold'), text_color=custom_colors["text_color"])
label1.place(x=30, y=5)
textbox1 = customtkinter.CTkTextbox(frame4, width=730, height=40, font=customtkinter.CTkFont(family='calibre', size=17), fg_color=custom_colors["widget_bg_color"], text_color=custom_colors["text_color"])
textbox1.place(x=15, y=40)

# Button to select an audio file
button_select = customtkinter.CTkButton(frame4, command=select, text='Select File', width=120, height=40, corner_radius=15, font=customtkinter.CTkFont(family='calibre', size=15, weight='bold'), fg_color=custom_colors["button_bg_color"], text_color=custom_colors["button_text_color"])
button_select.place(x=200, y=100)
# Button to record audio
button_record = customtkinter.CTkButton(frame4, command=record_audio, text='Record Audio', width=120, height=40, corner_radius=15, font=customtkinter.CTkFont(family='calibre', size=15, weight='bold'), fg_color=custom_colors["button_bg_color"], text_color=custom_colors["button_text_color"])
button_record.place(x=440, y=100)

# Create a frame for audio controls
frame3 = customtkinter.CTkFrame(app, width=760, height=100, corner_radius=15, fg_color=custom_colors["frame_bg_color"])
frame3.place(x=10, y=240)

# Load play, pause, and stop icons
play_image = customtkinter.CTkImage(light_image=Image.open("C:\\Users\\HP\\Desktop\\pfe2\\pynnote_vr\\play_183665.png"), size=(20, 20))
pause_image = customtkinter.CTkImage(light_image=Image.open("C:\\Users\\HP\\Desktop\\pfe2\\pynnote_vr\\pause_259451.png"), size=(20, 20))
stop_image = customtkinter.CTkImage(light_image=Image.open("C:\\Users\\HP\\Desktop\\pfe2\\pynnote_vr\\stop_167111.png"), size=(20, 20))

# Add buttons for audio controls with icons
button_play = customtkinter.CTkButton(frame3, command=play_audio, text='Play', width=80, height=40, corner_radius=15, font=customtkinter.CTkFont(family='calibre', size=15, weight='bold'), fg_color=custom_colors["button_bg_color"], text_color=custom_colors["button_text_color"], image=play_image)
button_play.place(x=150, y=30)
button_pause = customtkinter.CTkButton(frame3, command=pause_audio, text='Pause', width=80, height=40, corner_radius=15, font=customtkinter.CTkFont(family='calibre', size=15, weight='bold'), fg_color=custom_colors["button_bg_color"], text_color=custom_colors["button_text_color"], image=pause_image)
button_pause.place(x=350, y=30)
button_stop = customtkinter.CTkButton(frame3, command=stop_audio, text='Stop', width=80, height=40, corner_radius=15, font=customtkinter.CTkFont(family='calibre', size=15, weight='bold'), fg_color=custom_colors["button_bg_color"], text_color=custom_colors["button_text_color"], image=stop_image)
button_stop.place(x=550, y=30)

# Create a frame for progress bars and results
frame2 = customtkinter.CTkFrame(app, width=760, height=240, corner_radius=15, fg_color=custom_colors["frame_bg_color"])
frame2.place(x=10, y=360)

# Add labels and progress bars to the frame
label2 = customtkinter.CTkLabel(frame2, text='Progress Bar:', font=customtkinter.CTkFont(family='calibre', size=15, weight='bold'), text_color=custom_colors["text_color"])
label2.place(x=30, y=5)
label3 = customtkinter.CTkLabel(frame2, width=40, text='0%', font=customtkinter.CTkFont(family='calibre', size=20, weight='bold'), text_color=custom_colors["text_color"])
label3.place(x=690, y=5)
progress_Bar = customtkinter.CTkProgressBar(frame2, width=740, height=10, corner_radius=15, fg_color=custom_colors["progress_color"])
progress_Bar.set(0)
progress_Bar.place(x=10, y=40)

# Add additional labels and progress bar for accuracy
progress_label1 = customtkinter.CTkLabel(frame2, text='Accuracy:', font=customtkinter.CTkFont(family='calibre', size=15, weight='bold'), text_color=custom_colors["text_color"])
progress_label1.place(x=30, y=60)
progress_label2 = customtkinter.CTkLabel(frame2, width=40, text='0%', font=customtkinter.CTkFont(family='calibre', size=20, weight='bold'), text_color=custom_colors["text_color"])
progress_label2.place(x=690, y=55)
progress_bar2 = customtkinter.CTkProgressBar(frame2, width=740, height=10, corner_radius=15, fg_color=custom_colors["progress_color"])
progress_bar2.set(0)
progress_bar2.place(x=10, y=90)

# Add a label and textbox to display the identified speaker
label4 = customtkinter.CTkLabel(frame2, width=40, text='The Speaker is:', font=customtkinter.CTkFont(family='calibre', size=15, weight='bold'), text_color=custom_colors["text_color"])
label4.place(x=30, y=110)
textbox2 = customtkinter.CTkTextbox(frame2, width=730, height=90, font=customtkinter.CTkFont(family='calibre', size=17, weight='bold'), fg_color=custom_colors["widget_bg_color"], text_color=custom_colors["text_color"])
textbox2.place(x=15, y=140)

# Add a button to start the speaker search
button2 = customtkinter.CTkButton(frame2, command=search, text='Search', width=200, height=90, corner_radius=5, font=customtkinter.CTkFont(family='calibre', size=15, weight='bold'), fg_color=custom_colors["button_bg_color"], text_color=custom_colors["button_text_color"])
button2.place(x=550, y=140)

# Start the Tkinter event loop to run the GUI
app.mainloop()
