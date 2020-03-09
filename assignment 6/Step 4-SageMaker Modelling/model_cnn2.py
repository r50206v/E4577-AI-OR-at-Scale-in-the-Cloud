"""
Model definition for CNN sentiment training


"""

import os
import numpy as np
from sentiment_dataset import train_input_fn, eval_input_fn, validation_input_fn
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding
from tensorflow.keras import layers
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Convolution1D, GlobalMaxPooling1D
from tensorflow.keras.optimizers import Adam



def keras_model_fn(_, config):
    """
    Creating a CNN model for sentiment modeling

    """
    model=tf.keras.Sequential()
    input_length = config["padding_size"]
    input_dim = config["embeddings_dictionary_size"]
    output_dim = config["embeddings_vector_size"]
  

    i = 0
   # embedding_matrix = np.zeros((input_dim, output_dim))
   # with open(config['embeddings_path'], 'r',encoding="utf-8") as f:
   #     for line in f:
   #         values = line.split()
   #         if values[0].isalnum():
   #             embedding_matrix[i] = values[1:]
   #             i += 1

   # print(embedding_matrix.shape)


    model.add(tf.keras.layers.Embedding(
        input_dim, 
        output_dim, 
        #weights=[embedding_matrix], 
        input_length=input_length, 
        trainable=True
    ))
    #model.add(tf.keras.layers.Conv1D(100, 2, strides=1, padding='valid', activation='relu'))
    #model.add(tf.keras.layers.MaxPool1D(2))
    #model.add(tf.keras.layers.Conv1D(100, 2, strides=1, padding='same', activation='relu'))
    #model.add(tf.keras.layers.MaxPool1D(2))
    #model.add(tf.keras.layers.GlobalMaxPool1D())
    #model.add(tf.keras.layers.Dense(100))
    #model.add(tf.keras.layers.Activation('relu'))
    model.add(tf.keras.layers.LSTM(100))
    model.add(tf.keras.layers.Dense(1))
    model.add(tf.keras.layers.Activation('sigmoid'))
    adam1=keras.optimizers.Adam(learning_rate=0.01, beta_1=0.9, beta_2=0.999, amsgrad=False)
    model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])


    return model

def save_model(model, output):
    """
    Method to save models in SaveModel format with signature to allow for serving


    """
    model.save(output)

    print("Model successfully saved at: {}".format(output))
