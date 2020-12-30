# Realtime emotion detection program 
# Final Project
# Joheun Kang (N11691404)

import librosa
import librosa.display
import matplotlib.pyplot as plt
import tensorflow as tf
from matplotlib.pyplot import specgram
import keras
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Embedding
from keras.layers import LSTM
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras.layers import Input, Flatten, Dropout, Activation
from keras.layers import Conv1D, MaxPooling1D, AveragePooling1D
from keras.models import Model
from keras.callbacks import ModelCheckpoint
from sklearn.metrics import confusion_matrix
import pandas as pd
import pickle as pkl
import math
from keras.models import model_from_json
from sklearn.preprocessing import LabelEncoder
import pyaudio
import wave
import struct
import numpy as np
import matplotlib.pyplot as plt
import time
import librosa
import librosa.display

# for playing sound in Ipython notebook
from scipy.io.wavfile import read as wavread
from IPython.display import Audio
import time 
from pyfiglet import Figlet



f = Figlet(font='banner')
print (f.renderText('FALL 2020 DSP Final'))

# Load model and its weights 
print("........Loading emotions recognizer.......")

#json_file = open('model_new.json', 'r')
json_file = open('model_new.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)

# load weights into new model
loaded_model.load_weights("new.h5")

print("######## Loaded model from disk ########")

y_labels_encoded = {0: 'angry',1: 'calm',2: 'disgust',3: 'fear',4: 'happy',5: 'neutral',6: 'sad',7: 'surprised'}

print("Your model is loaded!")


# Voice recording start
print(".......Recording your voice to detect emotion......")

WAVE_OUTPUT_FILENAME = "final_project_voice.wav"

WIDTH = 2           # bytes per sample
CHANNELS = 1        # mono
RATE = 44100         # frames per second
CHUNK = 1024     # block length in samples
DURATION = 4       # Duration in seconds

K = int( DURATION * RATE / CHUNK )   # Number of blocks

# Real-time plotting 
plt.ion()           # Turn on interactive mode
plt.figure(1)
[g1] = plt.plot([], [], 'blue')  # Create empty line

n = range(0, CHUNK)
plt.xlim(0, CHUNK)         # set x-axis limits
plt.xlabel('Time (n)')
g1.set_xdata(n)  

plt.ylim(-5000, 5000) 

p = pyaudio.PyAudio()
FORMAT = p.get_format_from_width(WIDTH)

stream = p.open(format=FORMAT,
                channels=CHANNELS,
                rate=RATE,
                input=True,
                output = False,
                frames_per_buffer=CHUNK) #buffer

frames = []

#for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
for i in range(K):
    data = stream.read(CHUNK,exception_on_overflow = False)
    frames.append(data)
    signal_block = struct.unpack('h' * CHUNK, data)
    g1.set_ydata(signal_block)  
    plt.pause(0.0001) # to make real-time

print("Done recording!")
print('Please exit the real-time input voice plot!') 
stream.stop_stream()
stream.close()
p.terminate()

plt.ioff()           # Turn off interactive mode
plt.show()           # Keep plot showing at end of program
plt.close()

wf = wave.open(WAVE_OUTPUT_FILENAME, 'wb')
wf.setnchannels(CHANNELS)
wf.setsampwidth(p.get_sample_size(FORMAT))
wf.setframerate(RATE)
wf.writeframes(b''.join(frames))
wf.close()

print('\n')
print('\n')

print("program is analyzing your voice......")

# feature extraction for pre-trained emotion analyzer 
X, sample_rate = librosa.load('final_project_voice.wav', res_type='kaiser_fast',duration=2.5,sr=22050*2,offset=0.5)
sample_rate = np.array(sample_rate)
mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=13),axis=0)
featurelive = mfccs
livedf2 = featurelive
livedf2= pd.DataFrame(data=livedf2)
livedf2 = livedf2.stack().to_frame().T
twodim= np.expand_dims(livedf2, axis=2)
livepreds = loaded_model.predict(twodim, 
                         batch_size=32, 
                         verbose=1)
livepreds1=livepreds.argmax(axis=1)
liveabc = livepreds1.astype(int).flatten()

print("DONE analyzing")

print('\n')
print('\n')

print("You are expressing =>:", y_labels_encoded[liveabc[0]])
print('\n')


