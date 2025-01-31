import os
import glob
import numpy as np
from tqdm import tqdm
import itertools
import matplotlib.pyplot as plt
import pandas as pd

# Audio
import librosa
import librosa.display

# Scikit learn
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import shuffle
from sklearn.utils import class_weight

# Keras
import tensorflow as tf 
from tensorflow import keras
from keras._tf_keras.keras.models import Sequential
from keras._tf_keras.keras.layers import Input
from keras._tf_keras.keras.layers import Dense, Dropout, Activation, Flatten
from keras._tf_keras.keras.layers import Convolution2D, Conv2D, MaxPooling2D, GlobalAveragePooling2D
from keras._tf_keras.keras.utils import to_categorical
from keras._tf_keras.keras.applications import ResNet50

import warnings
warnings.filterwarnings('ignore', category=FutureWarning)

dataset = []
for folder in ["//content/drive/MyDrive/Stethoscope project/Deekshitha M/new dataset like set_a and set_b/set_a/**","/content/drive/MyDrive/Stethoscope project/Deekshitha M/new dataset like set_a and set_b/set_b/**"]:
    for filename in glob.iglob(folder, recursive=True):
        if os.path.isfile(filename):
            label = os.path.basename(filename).split("_")[0]
            duration = librosa.get_duration(filename=filename)
            # skip audio smaller than 3 secs
            if duration>=3:
                slice_size = 3
                iterations = int((duration-slice_size)/(slice_size-1))
                iterations += 1
                #initial_offset = (duration % slice_size)/2
                initial_offset = (duration - ((iterations*(slice_size-1))+1))/2
                if label not in ["Aunlabelledtest", "Bunlabelledtest", "artifact"]:
                    for i in range(iterations):
                        offset = initial_offset + i*(slice_size-1)
                        if (label == "normal"):
                            dataset.append({
                                "filename": filename,
                                "label": "normal",
                                "offset": offset
                            })
                        else:
                            dataset.append({
                                "filename": filename,
                                "label": "abnormal",
                                "offset": offset
                            })

dataset = pd.DataFrame(dataset)
dataset = shuffle(dataset, random_state=42)
dataset.info()

plt.figure(figsize=(4,6))
dataset.label.value_counts().plot(kind='bar', title="Dataset distribution")
plt.show()

train, test = train_test_split(dataset, test_size=0.2, random_state=42)

print("Train: %i" % len(train))
print("Test: %i" % len(test))

plt.figure(figsize=(20,10))
idx = 0
for label in dataset.label.unique():
    y, sr = librosa.load(dataset[dataset.label==label].filename.iloc[33], duration=3)
    print(dataset[dataset.label==label].filename.iloc[33])

    # Wave plot
    idx+=1
    plt.subplot(2, 3, idx)
    plt.title("%s waveshow" % label)
    librosa.display.waveshow(y, sr=sr)

    # Mel Spectrogram
    idx+=1
    plt.subplot(2, 3, idx)
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=2048, hop_length=512, n_mels=128)
    S_DB = librosa.power_to_db(S, ref=np.max)
    librosa.display.specshow(S_DB, sr=sr, hop_length=512, x_axis='time', y_axis='mel')
    plt.title("%s mel spectogram" % label)

    # MFCC (Mel spectrogram)
    idx+=1
    mfccs = librosa.feature.mfcc(S=librosa.power_to_db(S), n_mfcc=40)
    plt.subplot(2, 3, idx)
    librosa.display.specshow(mfccs, x_axis='time')
    plt.title("%s mfcc(Mel Spectrogram)" % label)
plt.show()

def extract_features(audio_path,offset):
#     y, sr = librosa.load(audio_path, duration=3)
    y, sr = librosa.load(audio_path, offset=offset, duration=3)
#     y = librosa.util.normalize(y)

    S = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=2048,
                                   hop_length=512,
                                   n_mels=128)
    mfccs = librosa.feature.mfcc(S=librosa.power_to_db(S), n_mfcc=40)

#     mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
    return mfccs

x_train = []
x_test = []

for idx in tqdm(range(len(train))):
    x_train.append(extract_features(train.filename.iloc[idx],train.offset.iloc[idx]))

for idx in tqdm(range(len(test))):
    x_test.append(extract_features(test.filename.iloc[idx],test.offset.iloc[idx]))

x_test = np.asarray(x_test)
x_train = np.asarray(x_train)

print("X train:", x_train.shape)
print("X test:", x_test.shape)

# Encode Labels
encoder = LabelEncoder()
encoder.fit(train.label)

y_train = encoder.transform(train.label)
y_test = encoder.transform(test.label)
from sklearn.utils import class_weight
class_weights = class_weight.compute_class_weight(class_weight='balanced', classes=np.unique(y_train), y=y_train)

x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], x_train.shape[2], 1)
x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], x_test.shape[2], 1)
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

print("X train:", x_train.shape)
print("Y train:", y_train.shape)
print("X test:", x_test.shape)
print("Y test:", y_test.shape)

inputs = Input(shape=(x_train.shape[1], x_train.shape[2], x_train.shape[3]))

# Define the Sequential model
model = Sequential()
model.add(Conv2D(filters=16, kernel_size=2, input_shape=(x_train.shape[1],x_train.shape[2],x_train.shape[3]), activation='relu'))
model.add(MaxPooling2D(pool_size=2))
model.add(Dropout(0.2))

model.add(Conv2D(filters=32, kernel_size=2, activation='relu'))
model.add(MaxPooling2D(pool_size=2))
model.add(Dropout(0.2))

model.add(Conv2D(filters=64, kernel_size=2, activation='relu'))
model.add(MaxPooling2D(pool_size=2))
model.add(Dropout(0.2))

model.add(Conv2D(filters=128, kernel_size=2, activation='relu'))
model.add(MaxPooling2D(pool_size=2))
model.add(Dropout(0.5))

# Add GlobalAveragePooling2D layer
model.add(GlobalAveragePooling2D())

# Add the Dense layer for classification
model.add(Dense(len(encoder.classes_), activation='softmax'))

# Print model summary
model.summary()

# Assuming x_train initially has shape (1024, 40, 130, 1)
x_train = tf.image.grayscale_to_rgb(tf.convert_to_tensor(x_train))  # Convert to 3 channels
# Now x_train has shape (1024, 40, 130, 3)

# Repeat the same for x_test
x_test = tf.image.grayscale_to_rgb(tf.convert_to_tensor(x_test))  # Convert to 3 channels

adam = keras.optimizers.Adam(learning_rate=0.001)
model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer=adam)

#check if x_train has 3 channels, if so, convert it to 1 channel(grayscale)
if x_train.shape[-1] == 3:
    x_train = tf.image.rgb_to_grayscale(x_train).numpy()
if x_test.shape[-1] == 3:
    x_test = tf.image.rgb_to_grayscale(x_test).numpy()

#Assuming class_weights is a NumPy array obtained from compute_class_weight
#and encoder.classes_ contains the unique class labels

#Convert class_weights array to dictionary
class_weight_dict = dict(enumerate(class_weights))
#OR
#Create a dictionary mapping class indices to weights
# class_weight_dict = {i: weight for i, weight in enumerate(class_weights)}

#Now use class_weight_dict in model.fit
history = model.fit(x_train,
                    y_train,
                    batch_size=128,
                    epochs=310,
                    validation_data=(x_test, y_test),
                    class_weight=class_weight_dict, # Use the dictionary here
                    shuffle=True)
                    #callbacks=[tf.keras.callbacks.TerminateOnNaN()]) # Add this line to stop training if NaN loss occurs
                    
# Loss Curves
plt.figure(figsize=[14,10])
plt.subplot(211)
plt.plot(history.history['loss'],'#d62728',linewidth=3.0)
plt.plot(history.history['val_loss'],'#1f77b4',linewidth=3.0)
plt.legend(['Training loss', 'Validation Loss'],fontsize=18)
plt.xlabel('Epochs ',fontsize=16)
plt.ylabel('Loss',fontsize=16)
plt.title('Loss Curves',fontsize=16)

# Accuracy Curves
plt.figure(figsize=[14,10])
plt.subplot(212)
plt.plot(history.history['accuracy'],'#d62728',linewidth=3.0)
plt.plot(history.history['val_accuracy'],'#1f77b4',linewidth=3.0)
plt.legend(['Training Accuracy', 'Validation Accuracy'],fontsize=18)
plt.xlabel('Epochs ',fontsize=16)
plt.ylabel('Accuracy',fontsize=16)
plt.title('Accuracy Curves',fontsize=16)

scores = model.evaluate(x_test, y_test, verbose=1)
print('Test loss:', scores[0])
print('Test accuracy:', scores[1])

predictions = model.predict(x_test, verbose=1)

y_true, y_pred = [],[]
classes = encoder.classes_
for idx, prediction in enumerate(predictions):
    y_true.append(classes[np.argmax(y_test[idx])])
    y_pred.append(classes[np.argmax(prediction)])

print(classification_report(y_pred, y_true))

model_name = "heartbeat_classifier (normalised).h5"
model.save(model_name)

# # load and evaluate a saved model
from keras._tf_keras.keras.models import load_model

# # load model
model = load_model("trained_heartbeat_classifier.h5")

# # File to be classified
classify_file = "my_heartbeat.wav"
x_test = []
x_test.append(extract_features(classify_file,0.5))
x_test = np.asarray(x_test)
x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], x_test.shape[2], 1)
pred = model.predict(x_test,verbose=1)

print(pred)

pred_class = model.predict_classes(x_test)
if pred_class[0]:
     print("Normal heartbeat")
     print("confidence:",pred[0][1])
else:
     print("Abnormal heartbeat")
     print("confidence:",pred[0][0])