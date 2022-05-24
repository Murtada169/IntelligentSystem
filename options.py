#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 21 21:02:18 2022

@author: iman
"""

from attr import field
import pygame
import time
import speech_recognition as sr 
# ----- Voon Tao ----- #
import datetime
import shutil
import os
from PDF_Generator import PDFGenerator
from MinutesManager import MinutesManager
#MURTADA #
import sounddevice as sd
import numpy as np
from scipy.io.wavfile import write
import soundfile
import pathlib
from turtle import st
import wave
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import models
from IPython import display

stop_it = False

def callback(recognizer, audio):  # this is called from the background thread
    global stop_it
    stop = "stop recording"
    
    try:
        recognized = recognizer.recognize_google(audio)
        print("You said: ", recognized)
        message_display(recognized)
        if stop in recognized:
            print("stopping now...")
            stop_it = True
        manager.recordMinutes(recognized)
    except sr.RequestError as exc:
        print(exc)
    except sr.UnknownValueError:
        print("Unable to recognize")

def listen():
    recog = sr.Recognizer()
    mic = sr.Microphone()

    with mic:
        recog.adjust_for_ambient_noise(mic)
    print("Pls say something..")
    return recog.listen_in_background(mic, callback)

# ----- Voon Tao ----- #
#---------- MODELS INITIATION  START     -------------------#
class_labels = [
    'backward','bed' ,'bird' ,'cat', 'dog', 'down' ,'eight' ,'five' ,'follow', 'forward', 'four',
 'go' ,'happy', 'house' ,'learn' ,'left', 'marvin', 'nine', 'no', 'off' ,'on' ,'one',
 'right', 'seven', 'sheila', 'six', 'stop', 'three', 'tree', 'two', 'up', 'visual',
 'wow', 'yes', 'zero'
]

DATASET_PATH = 'Data/speech_commands'
data_dir = pathlib.Path(DATASET_PATH)
commands = np.array(tf.io.gfile.listdir(str(data_dir)))
commands = commands[commands != 'README.md']
new_model_cnn = tf.keras.models.load_model('models/cnn/')
new_model_rnn = tf.keras.models.load_model('models/rnn/')

#-------------MODEL INITIATION END  ------------------------#
pygame.init()

display_width = 820
display_height = 600

black =(0,0,0)
alpha = (0,88,255)
white =(255,255,255)
red =(200,0,0)
green =(0,200,0)
bright_red =(255,0,0)
bright_green =(0,255,0)

## NEW
blue = (28,71,231,255)
teal = (91,198,173,255)
yellow = (244,189,65,255)
orange = (235,86,48,255)

bright_blue = (33,81,246,255)
bright_teal = (104,224,196,255)
bright_yellow = (248,213,72,255)
bright_orange = (236,95,51,255)

# Initialing Color
color = (211,211,211)

gameDisplay = pygame.display.set_mode((display_width,display_height))
pygame.display.set_caption("Smart Meeting Minutes")

gameDisplay.fill(white)

# Drawing Rectangle
pygame.draw.rect(gameDisplay, color, pygame.Rect(0, 450, 820, 150))
pygame.display.flip()


# need to change the path on your PC 
def close():
    pdf = PDFGenerator()
    pdf.print_chapter(filepath)
    pdfFilepath = os.path.join(r"C:\xampp\htdocs\IntelligentSystem", 'meeting_minutes.pdf')
    pdf.output(pdfFilepath, 'F')

    pygame.quit()
    quit()
    
def text_objects(text,font):
    textSurface = font.render(text,True,black)
    return textSurface,textSurface.get_rect()

def message_display(text):
    gameDisplay.fill(white) # update the text

    largeText = pygame.font.Font('freesansbold.ttf',30)
    TextSurf, TextRect = text_objects(text,largeText)
    TextRect.center =((display_width/2),(display_height/2))
    gameDisplay.blit(TextSurf,TextRect)
    pygame.display.update()


def button(msg,x,y,w,h,ic,ac,action=None):
    mouse=pygame.mouse.get_pos()
    click=pygame.mouse.get_pressed()
    if x+w>mouse[0]>x and y+h>mouse[1]>y:
        pygame.draw.rect(gameDisplay,ac,(x,y,w,h))

        if click[0]==1 and action!=None:
            action()
    else:
        pygame.draw.rect(gameDisplay,ic,(x,y,w,h))
    smallText = pygame.font.SysFont("arial",20, bold=True)
    textSurf, textRect=text_objects(msg,smallText)
    textRect.center =((x+(w/2)),(y+(h/2)))
    gameDisplay.blit(textSurf,textRect)
######################################## START PRE PROCESSING MODEL ####################################################
def decode_audio(audio_binary):
  # Decode WAV-encoded audio files to `float32` tensors, normalized
  # to the [-1.0, 1.0] range. Return `float32` audio and a sample rate.
  audio, _ = tf.audio.decode_wav(contents=audio_binary)
  # Since all the data is single channel (mono), drop the `channels`
  # axis from the array.
  return tf.squeeze(audio, axis=-1)

def get_label(file_path):
  parts = tf.strings.split(
      input=file_path,
      sep=os.path.sep)
  # Note: You'll use indexing here instead of tuple unpacking to enable this
  # to work in a TensorFlow graph.
  return parts[-2]

def get_waveform_and_predict(file_path):
  label = "Prediction"
  audio_binary = tf.io.read_file(file_path)
  waveform = decode_audio(audio_binary)
  return waveform, label

AUTOTUNE = tf.data.AUTOTUNE
def get_spectrogram(waveform):
  # Zero-padding for an audio waveform with less than 16,000 samples.
  input_len = 16000
  waveform = waveform[:input_len]
  zero_padding = tf.zeros(
      [16000] - tf.shape(waveform),
      dtype=tf.float32)
  # Cast the waveform tensors' dtype to float32.
  waveform = tf.cast(waveform, dtype=tf.float32)
  # Concatenate the waveform with `zero_padding`, which ensures all audio
  # clips are of the same length.
  equal_length = tf.concat([waveform, zero_padding], 0)
  # Convert the waveform to a spectrogram via a STFT.
  spectrogram = tf.signal.stft(
      equal_length, frame_length=255, frame_step=128)
  # Obtain the magnitude of the STFT.
  spectrogram = tf.abs(spectrogram)
  # Add a `channels` dimension, so that the spectrogram can be used
  # as image-like input data with convolution layers (which expect
  # shape (`batch_size`, `height`, `width`, `channels`).
  spectrogram = spectrogram[..., tf.newaxis]
  return spectrogram


def get_spectrogram_and_label_id(audio, label):
  spectrogram = get_spectrogram(audio)
  label_id = tf.math.argmax(label == commands)
  return spectrogram, label_id

def preprocess_dataset(files):
  files_ds = tf.data.Dataset.from_tensor_slices(files)
  output_ds = files_ds.map(
      map_func=get_waveform_and_predict,
      #map_func=get_waveform_and_label,
      num_parallel_calls=AUTOTUNE)
  output_ds = output_ds.map(
      map_func=get_spectrogram_and_label_id,
      num_parallel_calls=AUTOTUNE)
  return output_ds
######################################## END PRE PROCESSING MODEL  ######################################################
######################################## START RECORD AUDIO  ###########################################################
def recordaudio():
    fs = 44100  # Sample rate
    seconds = 2  # Duration of recording
    print ("talk")
    time.sleep(1)
    myrecording = sd.rec(int(seconds * fs), samplerate=fs, channels=1)
    sd.wait()  # Wait until recording is finished
    write('record.wav', fs, myrecording)  # Save as WAV file 
    data, samplerate = soundfile.read('record.wav')
    soundfile.write('record.wav', data, samplerate, subtype='PCM_16')

######################################## END RECORD AUDIO ###############################################################

def s2t_google():
    #takes recognition 
    #gameDisplay.blit(carImg,(0,0))
    r=sr.Recognizer()

    #start using microphone
    with sr.Microphone() as source:
        print('Say Something!')
        audio =r.listen(source)
        print("Done!")

    text = r.recognize_google(audio)
    
    print(text)
    message_display(text)

def s2t_cnn():
    recordaudio()
    predict_cnn()

def s2t_rnn():
    recordaudio()
    predict_rnn()

######################################## END SPEECH TO TEXT MODULES ######################################################

def predict_cnn():
    sample_file ='record.wav'
    sample_ds_cnn = preprocess_dataset([str(sample_file)])
    for spectrogram, label in sample_ds_cnn.batch(1):
        #print('Spectrogram shape:', spectrogram.shape)
        prediction = new_model_cnn(spectrogram)
        #print('prediction')
        #print(prediction) 
        single_result=prediction[0]
        #print('single_result: ')
        #print(single_result)
        most_likely_class_index = int(np.argmax(single_result))
        #print('most_likely:')
        #print(most_likely_class_index)
        class_likelihood = single_result[most_likely_class_index]
        #print("class_likelihood")
        #print(class_likelihood) 
        class_label = class_labels[most_likely_class_index]
        message_display(class_label)
        manager.recordMinutes(class_label)
        print("CNN MODEL PREDICTION: "+ class_label)

def predict_rnn():
    sample_file ='record.wav'
    sample_ds_rnn = preprocess_dataset([str(sample_file)])
    for spectrogram, label in sample_ds_rnn.batch(1):
        #print('Spectrogram shape:', spectrogram.shape)
        prediction = new_model_cnn(spectrogram)
        #print('prediction')
        #print(prediction) 
        single_result=prediction[0]
        #print('single_result: ')
        #print(single_result)
        most_likely_class_index = int(np.argmax(single_result))
        #print('most_likely:')
        #print(most_likely_class_index)
        class_likelihood = single_result[most_likely_class_index]
        #print("class_likelihood")
        #print(class_likelihood) 
        class_label = class_labels[most_likely_class_index]
        message_display(class_label)
        manager.recordMinutes(class_label)
        print("RNN MODEL PREDICTION: "+ class_label)
######################################## END PREDICTION MODEL ######################################################

def googleSpeechRecognition():
    global google_start
    google_start = True
    
    
def main():
    global stop_it
    global filepath
    global manager
    manager = MinutesManager()
    global google_start
    google_start = False
    filepath = manager.getFilePath()
    
    

    while True:
        if(google_start==True):
            audio_stopper = listen()
            google_start = False
        
        if stop_it:
            audio_stopper(wait_for_stop=True)
    
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()
        button("CNN",50,500,120,50,blue,bright_blue,s2t_cnn)
        button("RNN",250,500,120,50,teal,bright_teal,s2t_rnn)
        
        button("Google",450,500,120,50,yellow,bright_yellow,googleSpeechRecognition)
        button("Download",650,500,120,50,orange,bright_orange,close)
        pygame.display.update()

main()