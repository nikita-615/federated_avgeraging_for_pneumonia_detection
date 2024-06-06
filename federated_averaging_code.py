import numpy as np # forlinear algebra
import matplotlib.pyplot as plt #for plotting things
import os
from PIL import Image


# Keras Libraries
import keras
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.preprocessing.image import ImageDataGenerator, load_img
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import confusion_matrix, recall_score, precision_score, f1_score
from sklearn.utils.class_weight import compute_class_weight

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D , MaxPooling2D , Flatten , Activation , Dense , Dropout , BatchNormalization
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam, Adamax
from tensorflow.keras import regularizers

import warnings
warnings.filterwarnings('ignore')

train_folder= 'chest_xray/train'
val_folder = 'chest_xray/val'
test_folder = 'chest_xray/test'

# train
os.listdir(train_folder)
train_n = train_folder+'/NORMAL/'
train_p = train_folder+'/PNEUMONIA/'

# test
os.listdir(test_folder)
test_n = test_folder+'/NORMAL/'
test_p = test_folder+'/PNEUMONIA/'

# validate
os.listdir(val_folder)
val_n = val_folder+'/NORMAL/'
val_p = val_folder+'/PNEUMONIA/'

print("Normal training images are: ",len(os.listdir(train_n)))
print("Pnemonia training images are: ",len(os.listdir(train_p)))
print("Normal testing images are: ",len(os.listdir(test_n)))
print("Pnemonia testing images are: ",len(os.listdir(test_n)))
print("Normal validation images are: ",len(os.listdir(val_p)))
print("Pnemonia validation images are: ",len(os.listdir(val_p)))

#Normal pic
rand_norm= np.random.randint(0,len(os.listdir(train_n)))
norm_pic = os.listdir(train_n)[rand_norm]
norm_pic_address = train_n+norm_pic
print('normal picture title: ',norm_pic)

#Pneumonia
rand_p = np.random.randint(0,len(os.listdir(train_p)))
sic_pic =  os.listdir(train_p)[rand_norm]
sic_address = train_p+sic_pic
print('pneumonia picture title:', sic_pic)

# Load the images
norm_load = Image.open(norm_pic_address)
sic_load = Image.open(sic_address)

#Let's plt these images
f = plt.figure(figsize= (10,6))

a1 = f.add_subplot(1,2,1)
img_plot = plt.imshow(norm_load)
a1.set_title('Normal')

a2 = f.add_subplot(1, 2, 2)
img_plot = plt.imshow(sic_load)
a2.set_title('Pneumonia')

# # Fitting the CNN to the images
# # The function ImageDataGenerator augments your image by iterating through image as your CNN is getting ready to process that image

# train_datagen = ImageDataGenerator(rescale = 1./255,shear_range = 0.2,zoom_range = 0.2,horizontal_flip = True)
# validation_datagen = ImageDataGenerator(rescale = 1./255,shear_range = 0.2,zoom_range = 0.2,horizontal_flip = True)
# test_datagen = ImageDataGenerator(rescale = 1./255,shear_range = 0.2,zoom_range = 0.2,horizontal_flip = True)

# training_set = train_datagen.flow_from_directory('/content/drive/MyDrive/lung_cnn/chest_xray/train/',target_size = (64, 64),batch_size = 32,class_mode = 'binary')
# train_data, train_labels = next(training_set)

# validation_set = validation_datagen.flow_from_directory('/content/drive/MyDrive/lung_cnn/chest_xray/val',target_size = (64, 64),batch_size = 32,class_mode = 'binary')
# validation_data, validation_labels = next(validation_set)

# test_set = test_datagen.flow_from_directory('/content/drive/MyDrive/lung_cnn/chest_xray/test',target_size = (64, 64),batch_size = 32,class_mode = 'binary')
# test_data, test_labels = next(test_set)

from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Create ImageDataGenerator instances
train_datagen = ImageDataGenerator(rescale=1./255, shear_range=0.2, zoom_range=0.2, horizontal_flip=True)
validation_datagen = ImageDataGenerator(rescale=1./255, shear_range=0.2, zoom_range=0.2, horizontal_flip=True)
test_datagen = ImageDataGenerator(rescale=1./255, shear_range=0.2, zoom_range=0.2, horizontal_flip=True)

# Create generators using flow_from_directory
training_set = train_datagen.flow_from_directory('chest_xray/train/', target_size=(64, 64), batch_size=64, class_mode='binary')
validation_set = validation_datagen.flow_from_directory('chest_xray/val/', target_size=(64, 64), batch_size=64, class_mode='binary')
test_set = test_datagen.flow_from_directory('chest_xray/test/', target_size=(64, 64), batch_size=64, class_mode='binary')

# Convert the generators to tf.data.Dataset and repeat
train_dataset = tf.data.Dataset.from_generator(lambda: training_set, output_signature=(tf.TensorSpec(shape=(None, 64, 64, 3), dtype=tf.float32), tf.TensorSpec(shape=(None,), dtype=tf.float32)))
train_dataset = train_dataset.repeat()

validation_dataset = tf.data.Dataset.from_generator(lambda: validation_set, output_signature=(tf.TensorSpec(shape=(None, 64, 64, 3), dtype=tf.float32), tf.TensorSpec(shape=(None,), dtype=tf.float32)))
validation_dataset = validation_dataset.repeat()

test_dataset = tf.data.Dataset.from_generator(lambda: test_set, output_signature=(tf.TensorSpec(shape=(None, 64, 64, 3), dtype=tf.float32), tf.TensorSpec(shape=(None,), dtype=tf.float32)))
test_dataset = test_dataset.repeat()

# Extract one batch for each dataset
train_data, train_labels = next(iter(train_dataset))
validation_data, validation_labels = next(iter(validation_dataset))
test_data, test_labels = next(iter(test_dataset))

global client_models
client_models = []
training_accuracy = []
training_loss = []

#hyperparameters
num_clients = 3
learning_rate = 0.00001
epochs = 10
rounds = 3
batch_size = 64
global_learning_rate = 0.005

model = create_model()

cnn_model = model.fit(training_set,
                         steps_per_epoch = 100,
                         epochs = 6,
                         validation_data = validation_set,
                         validation_steps = 5)

test_accu = model.evaluate(test_set,steps=9)

print('The testing accuracy is :',test_accu[1]*100, '%')

"""Implementing Federated Learning"""

#hyperparameters
num_clients = 3
learning_rate = 0.001
epochs = 20
rounds = 5
batch_size = 18
global_learning_rate = 0.1

#function to create a client dataset for federated learning
def create_client_dataset(train_data, train_labels, num_clients, batch_size):
    client_datasets = []
    samples_per_client = len(train_data) // num_clients
    for i in range(num_clients):
        start_index = i * samples_per_client
        end_index = start_index + samples_per_client
        client_data = train_data[start_index:end_index]
        client_labels = train_labels[start_index:end_index]

        client_datasets.append(tf.data.Dataset.from_tensor_slices((client_data, client_labels)).batch(batch_size))
    return client_datasets

#function to train the model on a client dataset
def train_on_client(model, client_dataset, learning_rate, epochs):
    history = model.fit(client_dataset, validation_data=(train_data, train_labels), epochs=epochs, verbose=1)
    training_accuracy.append(history.history['accuracy'])
    training_loss.append(history.history['loss'])

    # Plot the training accuracy and validation accuracy over the number of epochs
    graph_epochs = range(1, len(history.history['accuracy']) + 1)
    plt.plot(graph_epochs, history.history['accuracy'], label='Training Accuracy')
    plt.plot(graph_epochs, history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()

    # Plot the training loss and validation loss over the number of epochs
    plt.plot(graph_epochs, history.history['loss'], label='Training Loss')
    plt.plot(graph_epochs, history.history['val_loss'], label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()
    return model, history

def federated_averaging(global_model, client_models, global_learning_rate):
    # average the weights of each layer across all client models
    new_weights = []
    for i, layer in enumerate(global_model.get_weights()):
        new_layer_weights = np.array([client_models[j][i] for j in range(len(client_models))]).mean(axis=0)
        new_weights.append(new_layer_weights)

    # apply the averaged weights to the global model
    for i, layer in enumerate(global_model.get_weights()):
        global_model_weights = np.array(layer)
        delta_weights = global_learning_rate * (new_weights[i] - global_model_weights)
        updated_weights = global_model_weights + delta_weights
        new_weights[i] = updated_weights
    global_model.set_weights(new_weights)
    return global_model

def federated_learning(train_data, train_labels, num_clients, learning_rate, epochs, rounds, batch_size):
    global_model = create_model()
    client_datasets = create_client_dataset(train_data, train_labels, num_clients, batch_size)
    for r in range(rounds):
        client_models = []
        for i in range(num_clients):
            client_model = create_model(global_model.get_weights())
            client_model, history = train_on_client(client_model, client_datasets[i], learning_rate, epochs)
            client_models.append(client_model.get_weights())



        # training accuracy and training loss for each client
        graph_epochs = range(1, len(history.history['accuracy']) + 1)
        for i in range(len(training_accuracy)):
          plt.plot(graph_epochs, training_accuracy[i], label=f'Client {i+1}')
        plt.title('Training Accuracy for Each Client')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.show()
        for i in range(len(training_loss)):
          plt.plot(graph_epochs, training_loss[i], label=f'Client {i+1}')
        plt.title('Training Loss for Each Client')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.show()


        global_model = federated_averaging(global_model, client_models, learning_rate)


        loss, accuracy = global_model.evaluate(train_data, train_labels, verbose=1)
        print("Round: {}, Loss: {}, Accuracy: {}".format(r + 1, loss, accuracy))

        predicted_labels = global_model.predict(test_data)
        predicted_labels = np.round(predicted_labels).flatten()
        cm = confusion_matrix(test_labels, predicted_labels)
        print("Confusion Matrix:\n", cm)

        precision = precision_score(test_labels, predicted_labels)
        print("Precision:", precision)

        recall = recall_score(test_labels, predicted_labels)
        print("Recall:", recall)

        f1 = f1_score(test_labels, predicted_labels)
        print("F1 score:", f1)

    return global_model

federated_learning(train_data, train_labels, num_clients, learning_rate, epochs, rounds, batch_size)