"""
Model definition for CNN sentiment training


"""

import os
import numpy as np
from sentiment_dataset import train_input_fn, eval_input_fn, validation_input_fn
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding
from tensorflow.keras import layers
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Convolution1D, GlobalMaxPooling1D
from tensorflow.keras.optimizers import Adam
import boto3



def keras_model_fn(_, config):
    """
    Creating a CNN model for sentiment modeling

    """
    model=tf.keras.Sequential()
    input_length = config["padding_size"]
    input_dim = config["embeddings_dictionary_size"]
    output_dim = config["embeddings_vector_size"]
  

    embedding_matrix = load_embeddings(
        config['embeddings_path'],
        config["embeddings_dictionary_size"],
        config["embeddings_vector_size"]
    )

    model.add(tf.keras.layers.Embedding(
        input_dim, 
        output_dim, 
        weights=[embedding_matrix], 
        input_length=input_length, 
        trainable=True
    ))
    model.add(tf.keras.layers.Conv1D(100, 2, strides=1, padding='valid', activation='relu'))
    model.add(tf.keras.layers.GlobalMaxPool1D())
    model.add(tf.keras.layers.Dense(100))
    model.add(tf.keras.layers.Activation('relu'))
    model.add(tf.keras.layers.Dense(1))
    model.add(tf.keras.layers.Activation('sigmoid'))
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])


    return model


def load_embeddings(embeddings_path, embeddings_dictionary_size, embeddings_vector_size):
    print("Load embeddings at path: %s" % embeddings_path)

    embeddings_matrix = np.zeros((embeddings_dictionary_size, embeddings_vector_size))

    s3_file = False
    if "s3://" in embeddings_path:
        s3_file = True
        s3_client = boto3.client("s3")

        path_split = embeddings_path.replace("s3://", "").split("/")
        bucket = path_split.pop(0)
        key = "/".join(path_split)

        data = s3_client.get_object(Bucket=bucket, Key=key)
        embeddings_file = data['Body'].iter_lines()

    else:
        embeddings_file = open(embeddings_path, 'r')

    for index, line in enumerate(embeddings_file):
        if index == embeddings_dictionary_size:
            break
        
        if s3_file:
            line = line.decode("utf-8")

        split = line.split(" ")
        vector = np.asarray(split[1:], dtype="float32")
        embeddings_matrix[index] = vector
    
    return embeddings_matrix


def save_model(model, output):
    """
    Method to save models in SaveModel format with signature to allow for serving


    """
    from datetime import datetime
    print('='*20, "\nSaving Model..")
    tf.saved_model.save(model, os.path.join(output, "1"))

    print("Model successfully saved at: {}".format(output))

