import pandas as pd
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

import seaborn as sns
import random
import os
import gc

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import models, Sequential
from tensorflow.keras import optimizers

from keras.layers.core import Dense, Dropout, Flatten
from keras.layers.convolutional import Conv2D, MaxPooling2D, SeparableConv2D

from tensorflow.keras.applications.vgg16 import VGG16

from keras.preprocessing.image import ImageDataGenerator

from tensorflow.keras.preprocessing.image import img_to_array, load_img

from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import precision_recall_curve

from keras.utils import get_file



                                 # partie1: importation des données

train_dir = "C:/Users/A.Khaterchi/Desktop/mini_projet/COM/AutismDataset/train"
test_dir = "C:/Users/A.Khaterchi/Desktop/mini_projet/COM/AutismDataset/test"

if os.path.exists(train_dir) and os.path.exists(train_dir):
    print("Les dossiers existent")
else:
    print("Les dossiers n'existent pas")
    

# ajout des images d'entrainement aux listes

#listes
train_non_autistic = []   
train_autistic = []


# pour chaque fichier si son nom contient 'non_autistic' on l'ajoute à la première liste sinon à la deuxième
for i in os.listdir(train_dir):
    if 'Non_Autistic' in ("C:/Users/A.Khaterchi/Desktop/mini_projet/COM/AutismDataset/train/{}".format(i)):
        train_non_autistic.append(("C:/Users/A.Khaterchi/Desktop/mini_projet/COM/AutismDataset/train/{}".format(i)))
    else:
        train_autistic.append(("C:/Users/A.Khaterchi/Desktop/mini_projet/COM/AutismDataset/train/{}".format(i)))
        
# création de la liste de test et ajout des images
test_imgs = ["C:/Users/A.Khaterchi/Desktop/mini_projet/COM/AutismDataset/test/{}".format(i) for i in os.listdir(test_dir)]


# concaténation des listes et mélange de façon aléatoire
train_imgs = train_autistic + train_non_autistic
random.shuffle(train_imgs)

# suppression des listes qui ne sont plus utiles
del train_autistic
del train_non_autistic


                                 # partie2: prétraitement des données





# définir la dimension et les couleurs (RGB)
nrows = 150
ncolumns  = 150
channels = 3

# création d'une fonction pour redimensionner une image

def dim(list_images):
    
    X = []  #pour stocker les images
    y = []  #pour stocker les labels (0:non autiste, 1:autiste)
    
    for image in list_images:
        X.append(cv2.resize(cv2.imread(image, cv2.IMREAD_COLOR), (nrows, ncolumns), interpolation = cv2.INTER_CUBIC))
        if 'Non_Autistic' in image:
            y.append(0)
        else:
            y.append(1)
            
    return X,y




# obtenir les images redimensionnées et leurs labels
X_train, y_train = dim(train_imgs)

del train_imgs



# convertir les listes en tableaux

plt.figure(figsize=(12, 8))  #créer une figure 12x8 pouces

X_train = np.array(X_train)  #convertir en tableau
y_train = np.array(y_train)

sns.countplot(y_train, saturation=1)  #créer un graphique

plt.title("Train image labels")

# vérifier les dimensions
print("Shape of train images:", X_train.shape)
print("Shape of train labels:", y_train.shape)


#images de validation

val_autistic = "C:/Users/A.Khaterchi/Desktop/mini_projet/COM/AutismDataset/valid/Autistic"
val_non_autistic = "C:/Users/A.Khaterchi/Desktop/mini_projet/COM/AutismDataset/valid/Non_Autistic"

val_autistic_imgs = ["C:/Users/A.Khaterchi/Desktop/mini_projet/COM/AutismDataset/valid/Autistic/{}".format(i) for i in os.listdir(val_autistic)]
val_non_autistic_imgs = ["C:/Users/A.Khaterchi/Desktop/mini_projet/COM/AutismDataset/valid/Non_Autistic/{}".format(i) for i in os.listdir(val_non_autistic)]

val_imgs = val_autistic_imgs + val_non_autistic_imgs
random.shuffle(val_imgs)

del val_autistic_imgs
del val_non_autistic_imgs

X_val, y_val = dim(val_imgs)

del val_imgs

plt.figure(figsize=(12, 8))
X_val = np.array(X_val)
y_val = np.array(y_val)
sns.countplot(y_val, saturation=1)
plt.title("Validation image labels")

print("Shape of validation images:", X_val.shape)
print("Shape of validation labels:", y_val.shape)

ntrain = len(X_train)
nval = len(X_val)
batch_size = 32


                                 # partie3: Construction du modèle



base_model = VGG16(include_top=False, weights=None, input_shape=(150,150,3))   #modèle pré-entrainé

weights_path = 'C:/Users/A.Khaterchi/Desktop/mini_projet/COM/vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5'
base_model.load_weights(weights_path)



for layer in base_model.layers:
   layer.trainable = False


model = keras.models.Sequential()

model.add(base_model)

model.add(layers.Flatten())

model.add(layers.Dense(512, activation = 'relu'))

model.add(layers.Dropout(0.5))

model.add(layers.Dense(1, activation = 'sigmoid'))


model.summary()







model.compile(loss = 'binary_crossentropy', optimizer = keras.optimizers.Adam(), metrics = ['acc']) # Compiler le modèle

train_datagen = ImageDataGenerator(rescale = 1./255,
                                  rotation_range = 40,
                                  width_shift_range = 0.2,
                                  height_shift_range = 0.2,
                                  shear_range = 0.2,
                                  zoom_range = 0.2,
                                  horizontal_flip = True)

val_datagen = ImageDataGenerator(rescale = 1./255)

# image generator
train_generator = train_datagen.flow(X_train, y_train, batch_size = batch_size)
val_generator = val_datagen.flow(X_val, y_val, batch_size = batch_size)






                                 # partie4: Entrainement

history = model.fit(train_generator,
                              steps_per_epoch=ntrain // batch_size,
                              epochs=1,
                              validation_data=val_generator,
                              validation_steps=nval // batch_size
                             )
                            

history_df = pd.DataFrame(history.history)
history_df


plt.figure(figsize=(12, 8))
sns.lineplot(data=history_df.loc[:, ["acc", "val_acc"]], palette=['b', 'r'], dashes=False)
sns.set_style("whitegrid")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.title("Training and Validation Accuracy")

plt.figure(figsize=(12, 8))
sns.lineplot(data=history_df.loc[:, ["loss", "val_loss"]], palette=['b', 'r'], dashes=False)
sns.set_style("whitegrid")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Training and Validation Loss")




                                 # partie4: Test


random.shuffle(test_imgs)
X_test, y_test = dim(test_imgs)
X = np.array(X_test)



                                 # partie5: Prediction


pred = model.predict(X)
threshold = 0.5
predictions = np.where(pred > threshold, 1,0)

test = pd.DataFrame(data = predictions, columns = ["predictions"])
test
test["filename"] = [os.path.basename(i) for i in test_imgs]
test["test_labels"] = y_test
test = test[["filename", "test_labels", "predictions"]]
test

plt.figure(figsize=(12, 8))
sns.countplot(test["predictions"], saturation=1)

model_accuracy = accuracy_score(y_test, predictions)
print("Model Accuracy: {:.2f}%".format(model_accuracy * 100))


plt.figure(figsize=(4,4))
for val, i in enumerate(test_imgs[:10]):
    img = mpimg.imread(i)
    imgplot = plt.imshow(img)
    plt.title(os.path.basename(i) + ' - Prediction: ' +  f"{'Autistic' if predictions[val] == 1 else 'Non-Autistic'}")
    plt.show()



