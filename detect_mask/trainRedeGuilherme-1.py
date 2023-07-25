# pip install opencv-python
#pip install imutils


import os
import zipfile
import random
import shutil
import tensorflow as tf
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping,ReduceLROnPlateau
from shutil import copyfile
from os import getcwd
from os import listdir
import cv2
from tensorflow.keras.layers import Conv2D, Input, ZeroPadding2D, BatchNormalization, Activation, MaxPooling2D, Flatten, Dense
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from sklearn.utils import shuffle
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import imutils
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image  as mpimg
import argparse
import csv
import pandas as pd



# initialize the initial learning rate, number of epochs to train for,
# and batch size
INIT_LR = 1e-4 # mais um parâmetro para teste
#EPOCHS = 100
EPOCHS_DEFAULT = 30
BS_DEFAULT = 32

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
#ap.add_argument("-d", "--dataset", required=True,
#	help="path to input dataset")
ap.add_argument("-p", "--plot", type=str, default="plot.png",
	help="path to output loss/accuracy plot")
ap.add_argument("-m", "--model", type=str,
	default="mask_detector.model",
	help="path to output face mask detector model")
ap.add_argument("-e", "--epochs", type=int,
	default="30",
	help="quantity of epochs to learning")
ap.add_argument("-bs", "--batch_size", type=int,
	default="32",
	help="batch size")
ap.add_argument("-lr", "--learning_rate", type=int,
	default="-1",
	help="Learning rate [from 0.0 to 1.0]")
args = vars(ap.parse_args())

# argumentos
EPOCHS_PARAM = args['epochs']
BS = args['batch_size']

if(args['learning_rate'] > 0.0):
    INIT_LR = args['learning_rate']

#print("The number of images with facemask labelled 'yes':",len(os.listdir('data/with_mask')))
#print("The number of images with facemask labelled 'no':",len(os.listdir('data/without_mask')))

def data_summary(main_path):
    
    yes_path = main_path+'with_mask'
    no_path = main_path+'without_mask'
        
    # number of files (images) that are in the the folder named 'yes' that represent tumorous (positive) examples
    m_pos = len(listdir(yes_path))
    # number of files (images) that are in the the folder named 'no' that represent non-tumorous (negative) examples
    m_neg = len(listdir(no_path))
    # number of all examples
    m = (m_pos+m_neg)
    
    pos_prec = (m_pos* 100.0)/ m
    neg_prec = (m_neg* 100.0)/ m
    
    print("-------------------------------------------------")
    print("-------------------------------------------------")
    print("-------------------------------------------------")
    print("[INFO] Number of examples: {",m,"}")
    print("[INFO] Percentage of positive examples: {",pos_prec,"}, number of pos examples: {",m_pos,"}") 
    print("[INFO] Percentage of negative examples: {",neg_prec,"}, number of neg examples: {",m_neg,"}") 
    print("-------------------------------------------------")
    print("-------------------------------------------------")
    print("-------------------------------------------------")
    
#augmented_data_path = 'facemask-dataset/trial1/augmented data1/' 
#augmented_data_path = 'data/'  
#augmented_data_path = 'facemask-rodolfo/FaceMaskDataSet/' 
augmented_data_path = 'data/' 
# observations-master/experiements/test/
data_summary(augmented_data_path)

def split_data(SOURCE, TRAINING, TESTING, SPLIT_SIZE):
    dataset = []
    
    for unitData in os.listdir(SOURCE):
        data = SOURCE + unitData
        if(os.path.getsize(data) > 0):
            dataset.append(unitData)
        else:
            print('Skipped ' + unitData)
            print('Invalid file i.e zero size')
    
    train_set_length = int(len(dataset) * SPLIT_SIZE)
    test_set_length = int(len(dataset) - train_set_length)
    shuffled_set = random.sample(dataset, len(dataset))
    train_set = dataset[0:train_set_length]
    test_set = dataset[-test_set_length:]
       
    for unitData in train_set:
        temp_train_set = SOURCE + unitData
        final_train_set = TRAINING + unitData
        copyfile(temp_train_set, final_train_set)
    
    for unitData in test_set:
        temp_test_set = SOURCE + unitData
        final_test_set = TESTING + unitData
        copyfile(temp_test_set, final_test_set)
 
YES_SOURCE_DIR   = augmented_data_path+"with_mask/"          #\data\with_mask
TRAINING_YES_DIR = augmented_data_path+"train/with_mask/"    #\dest_folder\train\with_mask
TESTING_YES_DIR  = augmented_data_path+"test/with_mask/"     #\dest_folder\test\with_mask
NO_SOURCE_DIR    = augmented_data_path+"without_mask/"       #\data\without_mask
TRAINING_NO_DIR  = augmented_data_path+"train/without_mask/" #\dest_folder\train\without_mask
TESTING_NO_DIR   = augmented_data_path+"test/without_mask/"  #\dest_folder\test\without_mask
WRONG_SOURCE_DIR = augmented_data_path+"wrong_use/"          #\data\without_mask

print("[INFO] Spliting data to trainnig and testing...")
split_size = .8
split_data(YES_SOURCE_DIR, TRAINING_YES_DIR, TESTING_YES_DIR, split_size)
split_data(NO_SOURCE_DIR, TRAINING_NO_DIR, TESTING_NO_DIR, split_size)

print("-------------------------------------------------")
print("-------------------------------------------------")
print("-------------------------------------------------")
print("[INFO] The number of images with facemask in the training set labelled 'yes':", len(os.listdir(TRAINING_YES_DIR)))
print("[INFO] The number of images with facemask in the test set labelled 'yes':", len(os.listdir(TESTING_YES_DIR)))
print("[INFO] The number of images without facemask in the training set labelled 'no':", len(os.listdir(TRAINING_NO_DIR)))
print("[INFO] The number of images without facemask in the test set labelled 'no':", len(os.listdir(TESTING_NO_DIR)))
print("-------------------------------------------------")
print("-------------------------------------------------")
print("-------------------------------------------------")

model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(100, (3,3), activation='relu', input_shape=(224, 224, 3)),
    tf.keras.layers.MaxPooling2D(2,2),
    
    tf.keras.layers.Conv2D(100, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(50, activation='relu'),
    tf.keras.layers.Dense(2, activation='softmax')
])

# compile our model
print("[INFO] compiling model...")
#opt = Adam(lr=INIT_LR, decay=INIT_LR / EPOCHS)
#model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc']) # usado com duas classes
model.compile(loss="categorical_crossentropy", optimizer='adam',metrics=["acc"]) # usado com duas ou mais classes

print("[INFO] summary of the compiled model...")
model.summary()
print("")

print("[INFO] prepering images...")

# dataset para treino
#TRAINING_DIR = "facemask-dataset/trial1/augmented data1/training"
#TRAINING_DIR = "dest_folder/train"
TRAINING_DIR = augmented_data_path+"train"

train_datagen = ImageDataGenerator(rescale=1.0/255,
                                   rotation_range=40,
                                   width_shift_range=0.2,
                                   height_shift_range=0.2,
                                   shear_range=0.2,
                                   zoom_range=0.2,
                                   horizontal_flip=True,
                                   fill_mode='nearest')

train_generator = train_datagen.flow_from_directory(TRAINING_DIR, 
                                                    batch_size=BS, 
                                                    target_size=(224, 224))
                            
# dataset para validação                            
#VALIDATION_DIR = "facemask-dataset/trial1/augmented data1/testing"
#VALIDATION_DIR = "dest_folder/test"
VALIDATION_DIR = augmented_data_path+"test"

validation_datagen = ImageDataGenerator(rescale=1.0/255)

validation_generator = validation_datagen.flow_from_directory(VALIDATION_DIR, 
                                                         batch_size=BS, 
                                                         target_size=(224, 224),
                                                         shuffle=False)

# classes encontradas
print("\n[INFO] summary of the processed classes:")
nome_classes = []
valoresReais = []
labels = (validation_generator.class_indices)
for k,v in labels.items():
    nome_classes.append(k)
    valoresReais.append(v)
#print(labels,' ',nome_classes)
labels = dict((v,k) for k,v in labels.items())
print(labels)
print("\nNome classes: ",nome_classes)
print("Índice classes: ",valoresReais)

# preparar os callbacks
print("\n[INFO] Preparing callbacks ...")
checkpoint = ModelCheckpoint('model-{epoch:03d}.model',monitor='val_loss',verbose=0,save_best_only=True,mode='auto')
#early = EarlyStopping(monitor='val_loss', min_delta=0, patience=5, verbose=1, mode='auto')
early = EarlyStopping(monitor='val_loss', min_delta=0, patience=0, verbose=0, mode='auto',baseline=None,restore_best_weights=False)
reduce = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, verbose=1, mode='auto', min_lr=0.00001)

# gerar o modelo de aprendizagem
print("[INFO] Genereting model ...")

history = model.fit(train_generator,
                      epochs=EPOCHS_PARAM,
                      validation_data=validation_generator,
                      callbacks=[]) #callbacks=[checkpoint,early])
                              
# serialize the model to disk
print("[INFO] saving mask detector model in ",args["model"],"...")
#model.save(args["model"]); #funciona só no linux
model.save(args["model"], save_format="h5") #funciona só no windows

print("[INFO] evaluating the network...")
#avaliador de gerador
STEP_SIZE_VALID=validation_generator.n//validation_generator.batch_size
model.evaluate(validation_generator,steps=STEP_SIZE_VALID)

# make predictions on the testing set
classificacoesPreditas = model.predict(validation_generator, batch_size=BS)
#print (classificacoesPreditas)
# for each image in the testing set we need to find the index of the
# label with corresponding largest predicted probability
classificacoesPreditas = np.argmax(classificacoesPreditas, axis=1)
classificacoesVerdadeiras = validation_generator.classes[validation_generator.index_array]
#print("Classificações verdadeiras:\n",classificacoesVerdadeiras)
#print("--")
#print ("Classificações preditivas:\n",classificacoesPreditas)


print("\n[INFO] Saving classification Report...")
# >>> print(classification_report(y_true, y_pred, target_names=target_names))
# show a nicely formatted classification report
relatorio = classification_report(classificacoesVerdadeiras, classificacoesPreditas,
                                  target_names=nome_classes, output_dict=True)
df = pd.DataFrame(relatorio).transpose()
df.to_csv(args["model"]+'_classification_report.csv', index = False, sep=';', encoding='utf-8')
# imprime relatório
print(df)

print("\n[INFO] Saving confusion matrix...")
matriz_confusao = confusion_matrix(classificacoesVerdadeiras,classificacoesPreditas)
df = pd.DataFrame(matriz_confusao).transpose()
df.to_csv(args["model"]+'_matriz_confusao.csv', index = False, sep=';', encoding='utf-8')
# imprime relatório
print(df)
#print("\n",matriz_confusao)

# plot the training loss and accuracy
print("\n[INFO] plotting stats...")
#N = EPOCHS_PARAM
N = len(history.history["loss"])
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, N), history.history["loss"], label="train_loss")
plt.plot(np.arange(0, N), history.history["val_loss"], label="val_loss")
#plt.plot(np.arange(0, N), history.history["accuracy"], label="train_acc")
plt.plot(np.arange(0, N), history.history["acc"], label="train_acc")
#plt.plot(np.arange(0, N), history.history["val_accuracy"], label="val_acc")
plt.plot(np.arange(0, N), history.history["val_acc"], label="val_acc")
plt.title("Training Loss and Accuracy")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="lower left")
plt.savefig(args["plot"])

# salva histórico usando pandas
#df = pd.DataFrame(history, columns= ['loss', 'val_loss', 'acc','val_acc'])
df = pd.DataFrame(history.history)
# salva em CSV
df.to_csv (args["model"]+'_history.csv', index =True, header=True, sep=';', encoding='utf-8')

print ("\n",df)