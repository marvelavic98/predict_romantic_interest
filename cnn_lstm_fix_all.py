#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 12 04:58:01 2021

@author: marvela
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 12 04:34:43 2021

@author: marvela
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 12 04:12:14 2021

@author: marvela
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 22 01:18:28 2020

@author: marvela
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Dec  8 14:11:09 2020

@author: User
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Oct 13 14:51:34 2020

@author: marvelavic
"""

import cv2
import os
import numpy as np 
import pickle
from sklearn.utils import shuffle 
from tensorflow import keras
from tensorflow.keras.callbacks import ModelCheckpoint
from sklearn.metrics import accuracy_score
from sklearn.metrics import balanced_accuracy_score
from sklearn.metrics import confusion_matrix
from imblearn.metrics import sensitivity_specificity_support
from tensorflow.keras.layers import Input, Dense, Flatten, TimeDistributed, Conv2D, Activation, MaxPooling2D, Dropout
from tensorflow.keras.layers import LSTM
from tensorflow.keras import optimizers
from tensorflow.keras import Model as ModelKeras
from tensorflow.keras.callbacks import EarlyStopping

import tensorflow as tf
tf.__version__
# tf.compat.v1.disable_v2_behavior()
# print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))


class PickleDumpLoad(object): 
    def __init__(self):
        self.address = '/media/marvela/fix_research/datas_try/no_flip/'
        
    def save_config(self, obj, filename): 
        with open('{}{}' . format(self.address, filename), 'wb') as config_f:
            pickle.dump(obj, config_f, protocol=4)       
        print('{} saved!' . format(filename))
        
    def load_config(self, filename):  
        with open('{}{}' . format(self.address, filename), 'rb') as f_in:
             obj = pickle.load(f_in)
        return obj

class EncodeLabelsCategorically(object):  
    def manual_categorical_labeling(self, label_to_encode=[], num_classes=2):
        uniqueness = sorted(list(set(label_to_encode)))
        labs = np.zeros((len(label_to_encode), num_classes))
        for ind, val in enumerate(label_to_encode): 
            labs[ind][uniqueness.index(val)] = 1 
        return labs  
    
class ExtractImages(object):
    def extract(self, status='train', min_pos_rank=5, img_dim=32, seq=20):
        
        if status == 'train':
            status = 'training'
        if status == 'test':
            status = 'testing'
        
        #convert videos into images/frames
        parent_path = '/media/marvela/fix_research/datas_try/no_flip/3rd try/{}' . format(status)
        listing = os.listdir(parent_path) 
        x_train = []
        y_train = []
        for label in listing:
            add = '{}/{}' . format(parent_path, label)
            
            # DETERMINE WHETHER PAIR OR UNPAIR
            sp = label.split(' ')
            first_rank = int(sp[0])
            second_rank = int(sp[2])
            label = 'PAIR' if first_rank >= min_pos_rank and second_rank >= min_pos_rank else 'UNPAIR'
            
            lst = os.listdir(add) 
            for s_l in lst:
                s_add = '{}/{}' . format(add, s_l)
                cap = cv2.VideoCapture(s_add)
                success, image = cap.read()
                print('{} --> Original Dimensions: {} --> Label: {}' . format(s_add, image.shape, label))
                split_data = list()
                
                while success:
                    success, image = cap.read() 
                    if image is not None:
                        width = img_dim
                        height = img_dim
                        dim = (width, height)
                        resized = cv2.resize(image, dim, interpolation = cv2.INTER_AREA)   
                        split_data.append(resized) 
                        
                        if len(split_data) == seq:
                            x_train.append(np.array(split_data))
                            y_train.append(label)
                            split_data = list() 
                
                print('Done! Current total sets {}.' . format(np.array(x_train).shape))
                
        # LABEL ENCODER
        y_train = EncodeLabelsCategorically().manual_categorical_labeling(label_to_encode=y_train, num_classes=2) 
        
        print('TRAIN SETS', np.array(x_train).shape)
        print('LABEL SETS', np.array(y_train).shape)
        return np.array(x_train), np.array(y_train) 

class Assesment(object):
    def get_performance_evaluation(self, y_pred, y_test): 
        accuracy = accuracy_score(y_pred, y_test)
        balanced_accuracy = balanced_accuracy_score(y_pred, y_test)
        sensitivity, specitivity, support = sensitivity_specificity_support(y_pred, y_test, average="micro")
        conf_matrix = confusion_matrix(y_pred, y_test)
        return accuracy, balanced_accuracy, sensitivity, specitivity, conf_matrix
    
class CNN_Model(object):   
    def vgg16_lstm(self, dimension=32, seq=20, classes=2):   
        input_shape = Input(shape=(seq, dimension, dimension, 3))  
        
        # 64        
        x1 = TimeDistributed(Conv2D(64, (3,3), padding='same', data_format='channels_last'))(input_shape)
        x1 = TimeDistributed(Activation('relu'))(x1)
        x1 = TimeDistributed(Conv2D(64, (3,3), padding='same', data_format='channels_last'))(x1)
        x1 = TimeDistributed(Activation('relu'))(x1)
        x1 = TimeDistributed(Conv2D(64, (3,3), padding='same', data_format='channels_last'))(x1)
        x1 = TimeDistributed(Activation('relu'))(x1)
        x1 = TimeDistributed(MaxPooling2D(pool_size=(2,2), strides=(2, 2)))(x1)
        x1 = TimeDistributed(Dropout(0.4))(x1)
        
        #96
        x1 = TimeDistributed(Conv2D(64, (3,3), padding='same', data_format='channels_last'))(x1)
        x1 = TimeDistributed(Activation('relu'))(x1)
        x1 = TimeDistributed(Conv2D(64, (3,3), padding='same', data_format='channels_last'))(x1)
        x1 = TimeDistributed(Activation('relu'))(x1)
        x1 = TimeDistributed(Conv2D(64, (3,3), padding='same', data_format='channels_last'))(x1)
        x1 = TimeDistributed(Activation('relu'))(x1)
        x1 = TimeDistributed(Dropout(0.4))(x1)
        
        x1 = TimeDistributed(Conv2D(64, (3,3), padding='same', data_format='channels_last'))(x1)
        x1 = TimeDistributed(Activation('relu'))(x1)
        x1 = TimeDistributed(Conv2D(64, (3,3), padding='same', data_format='channels_last'))(x1)
        x1 = TimeDistributed(Activation('relu'))(x1)
        x1 = TimeDistributed(Conv2D(64, (3,3), padding='same', data_format='channels_last'))(x1)
        x1 = TimeDistributed(Activation('relu'))(x1)
        x1 = TimeDistributed(Dropout(0.4))(x1)
        
        x1 = TimeDistributed(Conv2D(96, (3,3), padding='same', data_format='channels_last'))(x1)
        x1 = TimeDistributed(Activation('relu'))(x1)
        x1 = TimeDistributed(Conv2D(96, (3,3), padding='same', data_format='channels_last'))(x1)
        x1 = TimeDistributed(Activation('relu'))(x1)
        x1 = TimeDistributed(Conv2D(96, (3,3), padding='same', data_format='channels_last'))(x1)
        x1 = TimeDistributed(Activation('relu'))(x1)
        x1 = TimeDistributed(Dropout(0.4))(x1)
#        
#        
        x1 = TimeDistributed(Conv2D(96, (3,3), padding='same', data_format='channels_last'))(x1)
        x1 = TimeDistributed(Activation('relu'))(x1)
        x1 = TimeDistributed(Conv2D(96, (3,3), padding='same', data_format='channels_last'))(x1)
        x1 = TimeDistributed(Activation('relu'))(x1)
        x1 = TimeDistributed(Conv2D(96, (3,3), padding='same', data_format='channels_last'))(x1)
        x1 = TimeDistributed(Activation('relu'))(x1)
        x1 = TimeDistributed(Dropout(0.4))(x1)
        
        x1 = TimeDistributed(Conv2D(96, (3,3), padding='same', data_format='channels_last'))(x1)
        x1 = TimeDistributed(Activation('relu'))(x1)
        x1 = TimeDistributed(Conv2D(96, (3,3), padding='same', data_format='channels_last'))(x1)
        x1 = TimeDistributed(Activation('relu'))(x1)
        x1 = TimeDistributed(Conv2D(96, (3,3), padding='same', data_format='channels_last'))(x1)
        x1 = TimeDistributed(Activation('relu'))(x1)
        x1 = TimeDistributed(Dropout(0.4))(x1)
        
#        
#        
        #96
        x1 = TimeDistributed(Flatten())(x1) 
        
        
        x = Dense(512, activation='relu')(x1) 
        x = LSTM(512, return_sequences=False)(x)
        x = Dense(512, activation='relu')(x)  
        x = Dense(classes, activation='softmax', name='predictions')(x) 
        model = ModelKeras(inputs=input_shape, outputs=x)
        op = optimizers.Adam(lr=0.0001) 
        model.compile(loss='categorical_crossentropy', optimizer=op, metrics=['accuracy', 'mse']) 
        model.summary() 
        return model  

class DataGenerator(keras.utils.Sequence):
    def __init__(self, x_train, y_train, batch_size=4, shuffle=True):
        self.batch_size = batch_size
        self.x_train = x_train
        self.y_train = y_train
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        return int(np.floor(len(self.x_train) / self.batch_size))
    
    def __getitem__(self, index):
        #GENERATE INDEX OF BATCH DATA
        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]
        
        #GET DATA BASED ON THE GENERATED INDEXES
        x_train = np.array(list(map(lambda k: np.array(self.x_train[k]).astype(np.float32), indexes)))
        y_train = np.array(list(map(lambda k: np.array(self.y_train[k]).astype(np.float32), indexes)))
        
        # print(x_train_acc_.shape, x_train_fr_.shape, y_train_fr_.shape)
        
        return x_train, y_train
    
    def on_epoch_end(self):
        self.indexes = np.arange(len(self.x_train))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

class Main(object):
    def __init__(self):    
        # PARAMETER
        self.min_pos_rank = 5
        self.img_dim = 32
        self.seq = 20
        self.num_classes = 2
    
    def dataset_to_pickle(self):  
        # EXTRACT DATSETS
        x_train, y_train = ExtractImages().extract(status='train', min_pos_rank=self.min_pos_rank, img_dim=self.img_dim, seq=self.seq)
        x_test, y_test = ExtractImages().extract(status='test', min_pos_rank=self.min_pos_rank, img_dim=self.img_dim, seq=self.seq)

        # SAVE DATASETS
        PickleDumpLoad().save_config((x_train, y_train), 'frame_train_nocsv_noflip_32_3try.pickle')
        PickleDumpLoad().save_config((x_test, y_test), 'frame_test_nocsv_noflip_32_3try.pickle')
    
    def load_datasets(self):
        # LOAD THE SAVED DATASETS
        x_train, y_train = PickleDumpLoad().load_config('frame_train_nocsv_noflip_32_3try.pickle')
        x_test, y_test = PickleDumpLoad().load_config('frame_test_nocsv_noflip_32_3try.pickle')
        
        return x_train, y_train, x_test, y_test
    
    def run(self):    
        epochs = 100
        batch_size = 64
        # LOAD DATASETS
        x_train, y_train, x_test, y_test = self.load_datasets()
        
        idx = np.arange(len(y_train))
        np.random.shuffle(idx)  
        x_train = list(map(lambda x: x_train[x], idx))
        y_train = list(map(lambda x: y_train[x], idx))
        
        x_train = np.array(x_train[:])
        y_train = np.array(y_train[:])  
        
        x_test = np.array(list(map(lambda x: np.array(x_test[x]).astype(np.float32),  np.arange(len(x_test)))))
        y_test = np.array(list(map(lambda x: np.array(y_test[x]).astype(np.float32),  np.arange(len(y_test)))))
        
        
        # SHUFFLE THE DATA
        x_train, y_train = shuffle(x_train, y_train)
        x_test, y_test = shuffle(x_test, y_test)
        
        x_train, x_test = x_train.astype(np.float32), x_test.astype(np.float32)
#        y_train, y_test = y_train.astype(np.int32), y_test.astype(np.int32)
#        
#        print(x_train.shape, y_train.shape)
#        print(x_test.shape, y_test.shape)
        
        # TRAINING SECTION
         
        #filepath ="/media/marvela/fix_research/no_csv/weights-improvement-{epoch:02d}-{loss:.4f}-bigger.hdf5"
        #checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=True, mode='min')
        #early_stopping = EarlyStopping(monitor='val_loss', patience=0)
       # callbacks_list = [checkpoint]   
        
        
        
        model = CNN_Model().vgg16_lstm(dimension=self.img_dim, seq=self.seq, classes=self.num_classes)
        train_datagen = DataGenerator(x_train, y_train, batch_size=batch_size)   
        
        for ep in range(epochs):
            print('epoch', ep, 'from', epochs)
            #csv_logger = CSVLogger("/home/marvela/Documents/model_history_log.csv", append=True)
            #model.fit(x_train, y_train, batch_size=128, verbose=1, epochs=1, callbacks=callbacks_list)
            model.fit(train_datagen)
            pred = model.predict(x_test)
            #y_pred = model.predict(x_train)
            predicted = np.argmax(pred, axis=1)
            #y_predicted = np.argmax(y_pred, axis=1)
            #y_train_1d = np.argmax(y_train, axis=1)
            accuracy, balanced_accuracy, sensitivity, specitivity, conf_matrix = Assesment().get_performance_evaluation(predicted, np.argmax(y_test, axis=1))
            print('Test: {}' . format(accuracy))
        
            #training_error = mean_squared_error(y_train_1d, y_predicted)
            #print('Training_error :{} '.format(training_error))
            #print('prediction data: ', y_train_1d)
        
        num_1 = np.count_nonzero(predicted==1)
        #print(num_1)
    
        num_0 = np.count_nonzero(predicted==0)
    #print(num_0)
    
        if num_1>num_0:
            print('NOT INTERESTED')
        else:
            print('INTERESTED')
        
        
        #model.fit_generator(...,callbacks=[csv_logger])
    
run = Main()

# TO EXTRACT DATASETS
run.dataset_to_pickle()

# TO TRAIN AND TEST    
run.run() 
