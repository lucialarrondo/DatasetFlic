#Modules
import matplotlib.pyplot as plt
import os
import numpy as np
import cv2
from sklearn.model_selection import train_test_split
from keras import utils as np_utils
from keras import models as m
from keras import layers as l

width=212
height=212
data=[]
labels=[]
classes=['2-fast-2-furious-','12-oclock-high-special-edition-','along-came-polly-','american-wedding-unrated6x9-',
        'basic-instint-','batman-returns-','battle-cry-','bend-of-the-river-','bourne-supremacy-',
        'casino-royale-','collateral-disc1-','daredevil-disc1-','diehard2-dts-','flt0nnw1-',
        'funny-girl-dvd-video-','giant-side-a-','goldeneye-','hitch-','irma-la-douce','italian-job-',
        'mi2-','million-dollar-baby-disc-','monster-in-law-d1-','mr-mrs-smith-ws-','national-treasure-',
        'oceans-eleven-2001-','princess-diaries-2-','schindlers-list-','ten-commandments-disc1-','the-departed-']

#First we separate the images in diferent folders
#for i in range(len(classes)):
#    path='./'+classes[i]
#    os.mkdir(path)


#Retrieving the images and their labels
for i in range(len(classes)):
    path = os.getcwd()+'\\'+classes[i]
    images = os.listdir(path)

    for a in images:
        try:
            img = cv2.imread(path + '\\'+ a)
            img = cv2.resize(img, (width, height))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            data.append(img)
            labels.append(i)
        except:
            print("Error loading image")

#Converting lists into numpy arrays
data = np.array(data)
labels = np.array(labels)

#Splitting training and testing dataset
X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42)

#Images of a class as an example
for i in np.where(y_train == 0)[0][:3]:
    plt.imshow(X_train[i],cmap='gray')
    plt.show()

#Data visualization
#For faster learning scale values to [0;1] boundaries
X_train_norm=X_train/255
X_test_norm=X_test/255

#We use to_categorical
y_train=np_utils.to_categorical(y_train,len(classes))
y_test=np_utils.to_categorical(y_test,len(classes))

#Building the model
model=m.Sequential([
    l.InputLayer(shape=(width,height,1)),
    l.Conv2D(filters=32,kernel_size=5,padding='valid',activation='relu'),
    l.BatchNormalization(),
    l.MaxPool2D(pool_size=2,strides=2),
    l.Dropout(rate=0.25),
    l.Flatten(),
    l.Dense(64, activation='relu'),
    l.BatchNormalization(),
    l.Dropout(rate=0.5),
    l.Dense(len(classes),activation='softmax'),

])
print(model.summary())

#Compile the model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

#Training the model
epochs = 6
history = model.fit(X_train_norm, y_train, batch_size=32, epochs=epochs, validation_data=(X_test_norm, y_test))

#Two graphs that show what happen in the training
plt.figure(0)
plt.plot(history.history['accuracy'],label='training accuracy')
plt.plot(history.history['val_accuracy'],label='val_accuracy')
plt.title('Accuracy')
plt.xlabel('epochs')
plt.ylabel('accuracy')
plt.legend()
plt.show()

plt.figure(1)
plt.plot(history.history['loss'],label='training loss')
plt.plot(history.history['val_loss'],label='val_loss')
plt.title('Loss')
plt.xlabel('epochs')
plt.ylabel('loss')
plt.legend()
plt.show()

#Saving the model
model.save('datasetflic.h5')
model.save('datasetflic.keras')