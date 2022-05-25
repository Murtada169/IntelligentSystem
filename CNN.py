import os
import pathlib

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import tensorflow as tf

from tensorflow.keras import layers
from tensorflow.keras import models
from IPython import display

# Set the seed value for experiment reproducibility.
seed = 42
tf.random.set_seed(seed)
np.random.seed(seed)

# download dataset

DATASET_PATH = 'Data/speech_commands'
data_dir = pathlib.Path(DATASET_PATH)

# print folder names

commands = np.array(tf.io.gfile.listdir(str(data_dir)))
commands = commands[commands != 'README.md']
print('\nCommands:', commands)
print('\n')

# shuffle names

filenames = tf.io.gfile.glob(str(data_dir) + '/*/*') # get all files into a list
filenames = tf.random.shuffle(filenames) # shuffle 
num_samples = len(filenames) # number of files
print('Number of total examples:', num_samples)
print('Number of examples per label:',
      len(tf.io.gfile.listdir(str(data_dir/commands[0])))) # number of files in each label
print('Example file tensor:', filenames[0])
print('\n')

# split into test, training and validation
train_files = filenames[:int(abs(num_samples*0.8))]
val_files = filenames[int(abs(num_samples*0.8)): int(abs(num_samples*0.8) + abs(num_samples*0.1))]
test_files = filenames[-int((abs(num_samples*0.1))):]

print('Training set size', len(train_files))
print('Validation set size', len(val_files))
print('Test set size', len(test_files))
print('Total examples used', len(train_files) + len(val_files) + len(test_files))
print('\n')

# ---Preprocessing---#

# function to decode audio files
def decode_audio(audio_binary):
  audio, _ = tf.audio.decode_wav(contents=audio_binary) #decode into normalised float32 tensors
  return tf.squeeze(audio, axis=-1) #drop channels axis due to it being mono

# function to associate labels to audio file
def get_label(file_path):
  parts = tf.strings.split(
      input=file_path,
      sep=os.path.sep)
  return parts[-2]

# function to combine audio and labels
def get_waveform_and_label(file_path):
  label = get_label(file_path)
  audio_binary = tf.io.read_file(file_path)
  waveform = decode_audio(audio_binary)
  return waveform, label # returns as (audio, label)

AUTOTUNE = tf.data.AUTOTUNE # optimisation

files_ds = tf.data.Dataset.from_tensor_slices(train_files) # creates array of tensors

waveform_ds = files_ds.map( # edits all elements in the array to be in (audio, label)
    map_func=get_waveform_and_label, # function to get (audio, label) pairs
    num_parallel_calls=AUTOTUNE)

# ---Conversion into spectograms---

def get_spectrogram(waveform):
  input_len = 16000 # ensure consistent length size
  waveform = waveform[:input_len]
  zero_padding = tf.zeros(
      [16000] - tf.shape(waveform), # pads waveforms with less than 16000 bits
      dtype=tf.float32) 
  waveform = tf.cast(waveform, dtype=tf.float32) # changes data type to float32

  equal_length = tf.concat([waveform, zero_padding], 0) # concatenates to make sure each waveform is 16000

  spectrogram = tf.signal.stft( # creates spectogram via fourier transformation while keeping time information
      equal_length, frame_length=255, frame_step=128)

  spectrogram = tf.abs(spectrogram) # obtains magnitude
  spectrogram = spectrogram[..., tf.newaxis] # adds channel dimesions to make sure input shape matches CNN
  return spectrogram

# obtains spectogram, label pairs
def get_spectrogram_and_label_id(audio, label):
  spectrogram = get_spectrogram(audio)
  label_id = tf.argmax(label == commands)
  return spectrogram, label_id

# maps previous function through out dataset
spectrogram_ds = waveform_ds.map(
  map_func=get_spectrogram_and_label_id,
  num_parallel_calls=AUTOTUNE)

# ---Model building---

# prerocessing function
def preprocess_dataset(files):
  files_ds = tf.data.Dataset.from_tensor_slices(files) # creates array of audio files
  output_ds = files_ds.map( #maps it into waveforem, label
      map_func=get_waveform_and_label,
      num_parallel_calls=AUTOTUNE)
  output_ds = output_ds.map( #maps previous into spectogram, label
      map_func=get_spectrogram_and_label_id,
      num_parallel_calls=AUTOTUNE)
  return output_ds

# repeats for validation and test set
train_ds = spectrogram_ds
val_ds = preprocess_dataset(val_files)
test_ds = preprocess_dataset(test_files)

# sets batch size
batch_size = 64
train_ds = train_ds.batch(batch_size)
val_ds = val_ds.batch(batch_size)

# reducing latency
train_ds = train_ds.cache().prefetch(AUTOTUNE)
val_ds = val_ds.cache().prefetch(AUTOTUNE)

print('\n')
for spectrogram, _ in spectrogram_ds.take(1):
  input_shape = spectrogram.shape # prints out input shape of a sample
print('Input shape:', input_shape)
num_labels = len(commands) # sets amount of labels (which is 35)

# create a normalisation later (normalises into distribution around 0)
norm_layer =  tf.keras.layers.experimental.preprocessing.Normalization() 

# normalises around input data (spectogram.ds)
norm_layer.adapt(data=spectrogram_ds.map(map_func=lambda spec, label: spec))

# creates resizing layer to downsample to train faster
resize_layer =  tf.keras.layers.experimental.preprocessing.Resizing(32, 32)

model = models.Sequential([ # provides training
    layers.Input(shape=input_shape), # declares input shape
    resize_layer, # resizes for efficiency
    norm_layer, # normalises
    layers.Conv2D(32, 3, activation='relu'), # 2D convolutional layer, 32 output filters, kernel size 3, rectified linear unit function activation
    layers.Conv2D(64, 3, activation='relu'), # 2D convolutional layer, 64 output filters, kernel size 3, rectified linear unit function activation
    layers.MaxPooling2D(), # pools by going over the image in 'windows'
    layers.Dropout(0.25), # prevents overfitting by dropping a quarter of input units
    layers.Flatten(), # flattens input
    layers.Dense(128, activation='relu'), # dense layer maps every input to every output, 128 output space, relu activation
    layers.Dropout(0.5), # prevents overfitting by dropping half of input units
    layers.Dense(num_labels) # final dense layers with as many outputs as labels (35)
])

model.summary() # prints model summary
print('\n')

model.compile(
    optimizer=tf.keras.optimizers.Adam(), # Adam optimiser
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), # computes loss
    metrics=['accuracy'],
)
# saving the model strucutre and weights 
model.save('models/cnn')

EPOCHS = 10
history = model.fit(
    train_ds, # input data
    validation_data=val_ds, # validation data
    epochs=EPOCHS, # number of epochs
    callbacks=tf.keras.callbacks.EarlyStopping(verbose=1, patience=2), # early stopping to prevent long/unnecessary training times
)

# ---Metrics---

metrics = history.history
plt.plot(history.epoch, metrics['loss'], metrics['val_loss'])
plt.legend(['loss', 'val_loss'])
plt.show()

# ---Testing---

# empty sets
test_audio = []
test_labels = []

# split audio, label 
for audio, label in test_ds:
  test_audio.append(audio.numpy())
  test_labels.append(label.numpy())

# create arrays to hold audio and labels
test_audio = np.array(test_audio)
test_labels = np.array(test_labels)

# use model to predict
y_pred = np.argmax(model.predict(test_audio), axis=1)
y_true = test_labels

print('\n')
# calculates accuracy
test_acc = sum(y_pred == y_true) / len(y_true)
print(f'Test set accuracy: {test_acc:.0%}')
print('\n')

# display confusion matrix
confusion_mtx = tf.math.confusion_matrix(y_true, y_pred)
plt.figure(figsize=(10, 8))
sns.heatmap(confusion_mtx,
            xticklabels=commands,
            yticklabels=commands,
            annot=True, fmt='g')
plt.xlabel('Prediction')
plt.ylabel('Label')
plt.show()

# individual audio file testing
sample_file = data_dir/'no/01bb6a2a_nohash_0.wav'

sample_ds = preprocess_dataset([str(sample_file)])

for spectrogram, label in sample_ds.batch(1):
  prediction = model(spectrogram)
  plt.bar(commands, tf.nn.softmax(prediction[0]))
  plt.title(f'Predictions for "{commands[label[0]]}"')
  plt.show()