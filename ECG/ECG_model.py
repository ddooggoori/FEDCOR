import pandas as pd
import numpy as np
import os
import random
import tensorflow as tf
from keras.layers import Dense, Convolution2D, MaxPool2D, Flatten, Dropout, Input, Concatenate, Convolution1D, MaxPool1D, LSTM
from tensorflow.keras.layers import *
from tensorflow.keras.models import Sequential
from sklearn.utils.class_weight import compute_class_weight
from keras.models import Model
from tensorflow.keras.utils import plot_model
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Activation, BatchNormalization, Dropout, add, GlobalAveragePooling2D
from tensorflow.keras.layers import LeakyReLU, ELU, PReLU, ReLU
from tensorflow.keras.regularizers import l2,l1
from tensorflow.keras.metrics import * 
from sklearn.preprocessing import *
from tensorflow.keras.losses import BinaryFocalCrossentropy
from keras.regularizers import l1_l2
from keras.constraints import MaxNorm
from imblearn.over_sampling import *
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adam, RMSprop
from tensorflow.keras.callbacks import EarlyStopping
from Preprocessing import *
import warnings
warnings.filterwarnings('ignore')

seed = 42
os.environ["PYTHONHASHSEED"] = str(seed)
os.environ['TF_DETERMINISTIC_OPS'] = '1'
os.environ['TF_CUDNN_DETERMINISTIC'] = '1'
os.environ['TF_CUDNN_USE_FRONTEND '] = '1'
os.environ["CUDA_VISIBLE_DEVICES"] = '1' 
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
random.seed(seed)
np.random.seed(seed)
tf.random.set_seed(seed)  


def conv_block(input_tensor, num_channels):
    x = Convolution2D(num_channels, (3, 3), padding='same')(input_tensor)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    return x

def identity_block(input_tensor, num_channels):
    x = conv_block(input_tensor, num_channels)
    x = Convolution2D(num_channels, (3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = add([x, input_tensor])
    x = Activation('relu')(x)
    return x



def ECG_model(ecg, model):
    
    if model == '2d':
        ecg_input = Input(shape=(ecg.shape[1], ecg.shape[2], 1), name="ECG input") 

        ecg_model = Convolution2D(filters=32, kernel_size=(3, 3), padding='same', activation='relu')(ecg_input)
        ecg_model = BatchNormalization()(ecg_model)
        ecg_model = Convolution2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu')(ecg_model)
        ecg_model = BatchNormalization()(ecg_model)
        ecg_model = MaxPool2D(pool_size=(2, 2), strides=2, padding='same')(ecg_model)
        ecg_model = Dropout(0.2)(ecg_model)

        ecg_model = Convolution2D(filters=32, kernel_size=(3, 3), padding='same', activation='relu')(ecg_model)
        ecg_model = BatchNormalization()(ecg_model)
        ecg_model = Convolution2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu')(ecg_model)
        ecg_model = BatchNormalization()(ecg_model)
        ecg_model = MaxPool2D(pool_size=(2, 2), strides=2, padding='same')(ecg_model)
        ecg_model = Dropout(0.2)(ecg_model)

        ecg_model = Convolution2D(filters=32, kernel_size=(3, 3), padding='same', activation='relu')(ecg_model)
        ecg_model = BatchNormalization()(ecg_model)
        ecg_model = Convolution2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu')(ecg_model)
        ecg_model = BatchNormalization()(ecg_model)
        ecg_model = MaxPool2D(pool_size=(2, 2), strides=2, padding='same')(ecg_model)
        ecg_model = Dropout(0.2)(ecg_model)

        ecg_model = Flatten()(ecg_model)
        ecg_model = Dense(units = 512, activation='relu')(ecg_model)
        ecg_output = Dense(units = 64, activation='relu')(ecg_model)
    
    elif model == '1d':
        ecg_input = Input(shape=(ecg.shape[1], ecg.shape[2]))

        ecg_model = Convolution1D(filters=64, kernel_size=3, activation='relu', kernel_initializer='he_normal')(ecg_input)
        ecg_model = BatchNormalization()(ecg_model)
        ecg_model = MaxPool1D(pool_size=2)(ecg_model)
        ecg_model = Dropout(0.2)(ecg_model)

        ecg_model = Convolution1D(filters=128, kernel_size=3, activation='relu', kernel_initializer='he_normal')(ecg_model)
        ecg_model = BatchNormalization()(ecg_model)
        ecg_model = MaxPool1D(pool_size=2)(ecg_model)
        ecg_model = Dropout(0.2)(ecg_model)

        ecg_model = Flatten()(ecg_model)
        ecg_model = Dense(128, activation='relu')(ecg_model)
        ecg_output = Dense(64)(ecg_model)

    elif model == 'resnet':
        ecg_input = Input(shape=(ecg.shape[1], ecg.shape[2], 1))

        ecg_model = Convolution2D.Conv2D(64, (7, 7), strides=2, padding='same')(ecg_input)
        ecg_model = BatchNormalization()(ecg_model)
        ecg_model = Activation('relu')(ecg_model)
        ecg_model = MaxPool2D((3, 3), strides=2, padding='same')(ecg_model)

        ecg_model = conv_block(ecg_model, 64)
        ecg_model = identity_block(ecg_model, 64)
        ecg_model = MaxPool2D((3, 3), strides=2, padding='same')(ecg_model)

        ecg_model = conv_block(ecg_model, 128)
        ecg_model = identity_block(ecg_model, 128)
        ecg_model = MaxPool2D((3, 3), strides=2, padding='same')(ecg_model)

        ecg_model = conv_block(ecg_model, 256)
        ecg_model = identity_block(ecg_model, 256)
        ecg_model = MaxPool2D((3, 3), strides=2, padding='same')(ecg_model)
        
        ecg_model = GlobalAveragePooling2D()(ecg_model)
        ecg_output = Dense(64)(ecg_model)

    elif model == 'mlp':
        ecg_input = Input(shape=(ecg.shape[1], ecg.shape[2]), name="ECG input") 
        ecg_model = Flatten()(ecg_input)
        ecg_model = Dense(units = 1024)(ecg_model)
        ecg_model = BatchNormalization()(ecg_model)
        ecg_model = ReLU()(ecg_model)
        ecg_model = Dropout(0.2)(ecg_model)
        
        ecg_model = Dense(units = 128)(ecg_model)
        ecg_model = BatchNormalization()(ecg_model)
        ecg_model = ReLU()(ecg_model)
        ecg_model = Dropout(0.2)(ecg_model)
        
        ecg_model = Dense(units = 512)(ecg_model)
        ecg_model = BatchNormalization()(ecg_model)
        ecg_model = ReLU()(ecg_model)
        ecg_model = Dropout(0.2)(ecg_model)
        
        ecg_model = Dense(units = 64)(ecg_model)
        ecg_model = BatchNormalization()(ecg_model)
        ecg_model = ReLU()(ecg_model)
        ecg_output = Dense(units = 64)(ecg_model)
        
    elif model == 'lstm':
        ecg_input = Input(shape=(ecg.shape[1:]), name="ECG input") 
        ecg_model = LSTM(32, return_sequences=True)(ecg_input)
        ecg_model = Dropout(0.2)(ecg_model)
        ecg_model = LSTM(64, return_sequences=True)(ecg_model)
        ecg_model = Dropout(0.2)(ecg_model)
        ecg_model = LSTM(64, return_sequences=False)(ecg_model)
        ecg_model = Dropout(0.2)(ecg_model)
        ecg_model = Flatten()(ecg_model)
        ecg_model = Dense(units = 512)(ecg_model)
        ecg_output = Dense(units = 64)(ecg_model)
    
    ecg_MODEL = Model(inputs=ecg_input, outputs=ecg_output)
    
    return ecg_MODEL


def EHR_model(X_train):
    ehr_input = Input(shape=(X_train.shape[1], ), name="EHR input")
    ehr_model = Dense(units = 128, activation='relu', kernel_initializer='he_normal')(ehr_input)
    ehr_output = Dense(units = 64, activation='relu', kernel_initializer='he_normal')(ehr_model)
    
    ehr_MODEL = Model(inputs=ehr_input, outputs=ehr_output)
    
    return ehr_MODEL


def Combined_model(ecg_model, ehr_model):
               
    combined = Concatenate()([ecg_model.output, ehr_model.output])

    final_output = Dense(128, activation='relu')(combined)
    final_output = Dense(1, activation='sigmoid')(final_output)

    model = Model(inputs=[ecg_model.input, ehr_model.input], outputs=final_output)

    optimizer = Adam(learning_rate=0.001)

    model.compile(optimizer=optimizer, loss=BinaryFocalCrossentropy(), metrics=['accuracy'])

    # plot_model(model, show_shapes=True, show_layer_names=True)
    
    return model
    