#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun  1 08:51:20 2021

@author: marvela
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 24 11:54:14 2021

@author: marvela
"""

# -*- coding: utf-8 -*-
"""
Created on Mon Dec 21 00:53:15 2020

@author: User
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
from tensorflow import keras
from tensorflow.keras.callbacks import ModelCheckpoint
from sklearn.metrics import accuracy_score
from sklearn.metrics import balanced_accuracy_score
from sklearn.metrics import confusion_matrix
from imblearn.metrics import sensitivity_specificity_support
from tensorflow.keras.layers import Reshape, Input, Dense, Flatten, TimeDistributed, Conv2D, Conv1D, Activation, MaxPooling1D, MaxPooling2D, Concatenate, Dropout
from tensorflow.keras.layers import LSTM
from tensorflow.keras import optimizers
from tensorflow.keras import Model as ModelKeras 

import tensorflow as tf
tf.__version__
# tf.compat.v1.disable_v2_behavior()
# print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))


class PickleDumpLoad(object): 
    def __init__(self):
        self.address = '/media/marvela/fix_research/'
        
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
    def extract(self, status='train', min_pos_rank=5, img_dim=16, seq=20):
        
        if status == 'train':
            status = 'training'
        if status == 'test':
            status = 'testing'
        
        #convert videos into images/frames
        parent_path = '/media/marvela/fix_research/{}' . format(status)
        listing = os.listdir(parent_path) 
        x_train = []
        y_train = []
        user_lbl = []
        for label in listing:
            user_lbl.append(label)
            add = '{}/{}' . format(parent_path, label)
            
            # DETERMINE WHETHER PAIR OR UNPAIR
            sp = label.split(' ') #label based on number
            first_rank = int(sp[0])
            second_rank = int(sp[2])
            label = 'PAIR' if first_rank >= min_pos_rank and second_rank >= min_pos_rank else 'UNPAIR'
            
            lst = os.listdir(add)
            
            #pre_x_train = []
            #pre_y_train = []
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
            #x_train.append(pre_x_train)
            #y_train.append(pre_y_train)
            
        # LABEL ENCODER
        y_train = EncodeLabelsCategorically().manual_categorical_labeling(label_to_encode=y_train, num_classes=2) 
        
        print('TRAIN SETS ', np.array(x_train).shape)
        print('LABEL SETS ', np.array(y_train).shape)
        return np.array(x_train), np.array(y_train), np.array(user_lbl)

class Assesment(object):
    def get_performance_evaluation(self, y_pred, y_test): 
        accuracy = accuracy_score(y_pred, y_test)
        balanced_accuracy = balanced_accuracy_score(y_pred, y_test)
        sensitivity, specitivity, support = sensitivity_specificity_support(y_pred, y_test, average="micro")
        conf_matrix = confusion_matrix(y_pred, y_test)
        return accuracy, balanced_accuracy, sensitivity, specitivity, conf_matrix

class DataCsv(object):
    def training_file(self, min_pos_rank=5, seq=20, data_dim=120):
        basepath_training = '/media/marvela/fix_research/acceleration_data/fix/training'
        # CALL LIST OF CSV IN THE FOLDER
        #csv_training = glob.glob(os.path.join(basepath_training, '*.csv'))
        csv_training= os.listdir(basepath_training)
        dataframes_trn = []
        user_lbl = []     
        
        y_train= []
        for file_training in csv_training:            
            add_t = '{}/{}' . format(basepath_training, file_training)
            
            # LABEL BASED ON NUMBER
            sp_t = file_training.split(' ') 
            first_rank_t = int(sp_t[0])
            second_rank_t = int(sp_t[2])
            file_training= 'PAIR' if first_rank_t >= min_pos_rank and second_rank_t >= min_pos_rank else 'UNPAIR'
            
            lst_ = os.listdir(add_t)
            
            #split_data = list()
            for s_l in lst_: 
                s_add_t = '{}/{}' . format(add_t, s_l)
                a_ = open(s_add_t, 'r').read()
                a_split = a_.split('\n') # SPLIT DATA EVERY AFTER \n
                a_ = []
                for x in a_split:
                    x_s = x.split(',')
                    if len(x_s) > 0:
                        a_.append(x_s)
                a = []
                b = []
                for x in a_:
                    if len(x) > 4:
                        a.append(x)
                        #if len(a) == seq:
                         #   b.append(a)
                
                for x in range(data_dim):
                     b.append(a[x::data_dim])
                     dataframes_trn.append(b)
                     user_lbl.append(file_training.split('/')[-1].split('.')[0])
                print('Done! Current total sets {}.' . format(np.array(dataframes_trn).shape))
                
                #user_lbl.append(file_testing.split('/')[-1].split('.')[0])
            y_train= EncodeLabelsCategorically().manual_categorical_labeling(label_to_encode=user_lbl, num_classes=2) 
        return dataframes_trn, user_lbl, np.array(y_train)
    
    def testing_file(self, min_pos_rank=5, seq=20, data_dim=120):
        #basepath_testing = '/media/marvela/fix_research/acceleration_data/fix/testing/'
        basepath_ = '/media/marvela/fix_research/acceleration_data/fix/testing'
        #csv_testing = glob.glob(os.path.join(basepath_, '*.csv'))
        csv_testing = os.listdir(basepath_)
        
        dataframes_tst = []
        user_lbl = []
        y_test = []
        for file_testing in csv_testing:            
            add_t = '{}/{}' . format(basepath_, file_testing)
            
            # LABEL BASED ON NUMBER
            sp_t = file_testing.split(' ') 
            first_rank_t = int(sp_t[0])
            second_rank_t = int(sp_t[2])
            file_testing = 'PAIR' if first_rank_t >= min_pos_rank and second_rank_t >= min_pos_rank else 'UNPAIR'
            
            lst_ = os.listdir(add_t)
            
            #split_data = list()
            for s_l in lst_: 
                s_add_t = '{}/{}' . format(add_t, s_l)
                a_ = open(s_add_t, 'r').read()
                a_split = a_.split('\n') # SPLIT DATA EVERY AFTER \n
                a_ = []
                for x in a_split:
                    x_s = x.split(',')
                    if len(x_s) > 0:
                        a_.append(x_s)
                a = []
                b = []
                for x in a_:
                    if len(x) > 4:
                        a.append(x)
                
                for x in range(data_dim):
                     b.append(a[x::data_dim])
                     dataframes_tst.append(b)
                     user_lbl.append(file_testing.split('/')[-1].split('.')[0])
                
                print('Done! Current total sets {}.' . format(np.array(dataframes_tst).shape))
    
            y_test = EncodeLabelsCategorically().manual_categorical_labeling(label_to_encode=user_lbl, num_classes=2)    
        return dataframes_tst, np.array(y_test), user_lbl
     
    
class CNN_Model(object):   
    def vgg19_lstm(self, dimension=16, seq=20, classes=2, datas_a=120):   
        #SHAPE
        input_shape_frame = Input(shape=(seq, dimension, dimension, 3))  
        input_shape_acc = Input(shape=(datas_a, seq, 6))
        
        #print(input_shape_frame)
        #print(input_shape_acc)
        
        # COMPUTE THE NUMBER OF PARAMETERS
        # ((shape of width of the filter * shape of height of the filter * number of filters in the previous layer+1)*current number of filters)
        
        #FRAME
        # 64        
        frame = TimeDistributed(Conv2D(32, (3,3), padding='same', data_format='channels_last'))(input_shape_frame)
        frame = TimeDistributed(Activation('relu'))(frame)
        frame = TimeDistributed(Conv2D(32, (3,3), padding='same', data_format='channels_last'))(frame)
        frame = TimeDistributed(Activation('relu'))(frame)
        frame = TimeDistributed(Conv2D(32, (3,3), padding='valid', data_format='channels_last'))(frame)
        frame = TimeDistributed(Activation('relu'))(frame)  
        
        frame = TimeDistributed(Conv2D(32, (3,3), padding='same', data_format='channels_last'))(frame)
        frame = TimeDistributed(Activation('relu'))(frame)
        frame = TimeDistributed(Conv2D(32, (3,3), padding='same', data_format='channels_last'))(frame)
        frame = TimeDistributed(Activation('relu'))(frame)
        frame = TimeDistributed(Conv2D(32, (3,3), padding='valid', data_format='channels_last'))(frame)
        frame = TimeDistributed(Activation('relu'))(frame)  
        
        #128
        frame = TimeDistributed(Conv2D(64, (3,3), padding='same', data_format='channels_last'))(frame)
        frame = TimeDistributed(Activation('relu'))(frame)
        frame = TimeDistributed(Conv2D(64, (3,3), padding='same', data_format='channels_last'))(frame)
        frame = TimeDistributed(Activation('relu'))(frame)
        frame = TimeDistributed(Conv2D(64, (3,3), padding='valid', data_format='channels_last'))(frame)
        frame = TimeDistributed(Activation('relu'))(frame)
        
        #128
        frame = TimeDistributed(Conv2D(64, (3,3), padding='same', data_format='channels_last'))(frame)
        frame = TimeDistributed(Activation('relu'))(frame)
        frame = TimeDistributed(Conv2D(64, (3,3), padding='same', data_format='channels_last'))(frame)
        frame = TimeDistributed(Activation('relu'))(frame)
        frame = TimeDistributed(Conv2D(64, (3,3), padding='valid', data_format='channels_last'))(frame)
        frame = TimeDistributed(Activation('relu'))(frame)
        frame = TimeDistributed(MaxPooling2D(pool_size=(2,2), strides=(2, 2)))(frame)
        
        frame = TimeDistributed(Conv2D(64, (3,3), padding='same', data_format='channels_last'))(frame)
        frame = TimeDistributed(Activation('relu'))(frame)
        frame = TimeDistributed(Conv2D(64, (3,3), padding='same', data_format='channels_last'))(frame)
        frame = TimeDistributed(Activation('relu'))(frame)
        frame = TimeDistributed(Conv2D(64, (3,3), padding='valid', data_format='channels_last'))(frame)
        frame = TimeDistributed(Activation('relu'))(frame)
        frame = TimeDistributed(MaxPooling2D(pool_size=(2,2), strides=(2, 2)))(frame)
        #frame = Dropout(0.4)(frame)
        
        # ACC
        # 64        
        acc = TimeDistributed(Conv1D(32, 3, padding='same', data_format='channels_last'))(input_shape_acc)
        acc = TimeDistributed(Activation('relu'))(acc)
        acc = TimeDistributed(Conv1D(32, 3, padding='same', data_format='channels_last'))(acc)
        acc = TimeDistributed(Activation('relu'))(acc)
        acc = TimeDistributed(Conv1D(32, 3, padding='valid', data_format='channels_last'))(acc)
        acc = TimeDistributed(Activation('relu'))(acc)
        
        acc = TimeDistributed(Conv1D(32, 3, padding='same', data_format='channels_last'))(acc)
        acc = TimeDistributed(Activation('relu'))(acc)
        acc = TimeDistributed(Conv1D(32, 3, padding='same', data_format='channels_last'))(acc)
        acc = TimeDistributed(Activation('relu'))(acc)
        acc = TimeDistributed(Conv1D(32, 3, padding='valid', data_format='channels_last'))(acc)
        acc = TimeDistributed(Activation('relu'))(acc)
        
        acc = TimeDistributed(Conv1D(32, 3, padding='same', data_format='channels_last'))(acc)
        acc = TimeDistributed(Activation('relu'))(acc)
        acc = TimeDistributed(Conv1D(32, 3, padding='same', data_format='channels_last'))(acc)
        acc = TimeDistributed(Activation('relu'))(acc)
        acc = TimeDistributed(Conv1D(32, 3, padding='valid', data_format='channels_last'))(acc)
        acc = TimeDistributed(Activation('relu'))(acc)  
        acc = TimeDistributed(MaxPooling1D(pool_size=2, strides=2))(acc)
        
        acc = TimeDistributed(Conv1D(64, 3, padding='same', data_format='channels_last'))(acc)
        acc = TimeDistributed(Activation('relu'))(acc)
        acc = TimeDistributed(Conv1D(64, 3, padding='same', data_format='channels_last'))(acc)
        acc = TimeDistributed(Activation('relu'))(acc)
        acc = TimeDistributed(Conv1D(64, 3, padding='valid', data_format='channels_last'))(acc)
        acc = TimeDistributed(Activation('relu'))(acc)
        acc = TimeDistributed(MaxPooling1D(pool_size=2, strides=2))(acc)
        
        #acc = Dropout(0.4)(acc)
        
        acc_f = Flatten()(acc)
        frame_f = Flatten()(frame)
        
        # CONCAT AND RESHAPE
        concat = Concatenate()([acc_f, frame_f])  
        x_reshaped = Reshape([concat.get_shape()[1], 1])(concat)  
        
        cc = Conv1D(32, 3, padding='same', data_format='channels_last')(x_reshaped)
        cc = Conv1D(32, 3, padding='same', data_format='channels_last')(cc)
        cc = Conv1D(32, 3, padding='valid', data_format='channels_last')(cc)
        cc = MaxPooling1D(pool_size=2, strides=2)(cc)
        
        cc = Conv1D(32, 3, padding='same', data_format='channels_last')(cc)
        cc = Conv1D(32, 3, padding='same', data_format='channels_last')(cc)
        cc = Conv1D(32, 3, padding='valid', data_format='channels_last')(cc)
        cc = MaxPooling1D(pool_size=2, strides=2)(cc)
 
        cc = Conv1D(64, 3, padding='same', data_format='channels_last')(cc)
        cc = Conv1D(64, 3, padding='same', data_format='channels_last')(cc)
        cc = Conv1D(64, 3, padding='valid', data_format='channels_last')(cc)
        cc = MaxPooling1D(pool_size=2, strides=2)(cc)
 
        cc = Conv1D(64, 3, padding='same', data_format='channels_last')(cc)
        cc = Conv1D(64, 3, padding='same', data_format='channels_last')(cc)
        cc = Conv1D(64, 3, padding='valid', data_format='channels_last')(cc)
        cc = MaxPooling1D(pool_size=2, strides=2)(cc)
        
        cc = Conv1D(64, 3, padding='same', data_format='channels_last')(cc)
        cc = Conv1D(64, 3, padding='same', data_format='channels_last')(cc)
        cc = Conv1D(64, 3, padding='valid', data_format='channels_last')(cc)
        cc = MaxPooling1D(pool_size=2, strides=2)(cc)
    
        cc = Conv1D(64, 3, padding='same', data_format='channels_last')(cc)
        cc = Conv1D(64, 3, padding='same', data_format='channels_last')(cc)
        cc = Conv1D(64, 3, padding='valid', data_format='channels_last')(cc)
        cc = MaxPooling1D(pool_size=2, strides=2)(cc)
        #cc = Dropout(0.3)(cc)
        
        x_ = Dense(512, activation='relu')(cc) 
        
        x_ = LSTM(512, return_sequences=False)(x_)
        x_ = Dense(512, activation='relu')(x_)   
        x_ = Dense(classes, activation='softmax', name='prediction')(x_)
        model = ModelKeras(inputs=[input_shape_acc, input_shape_frame], outputs = x_)
        
        op = optimizers.Adam(lr=0.0001) 
        model.compile(loss='categorical_crossentropy', optimizer=op, metrics=['accuracy']) 
        model.summary() 
        return model

class DataGenerator(keras.utils.Sequence):
    def __init__(self, x_train_acc_, x_train_fr_, y_train_fr_, batch_size=4, shuffle=True):
        self.batch_size = batch_size
        self.x_train_acc_ = x_train_acc_
        self.x_train_fr_ = x_train_fr_
        self.y_train_fr_ = y_train_fr_
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        return int(np.floor(len(self.x_train_acc_) / self.batch_size))
    
    def __getitem__(self, index):
        #GENERATE INDEX OF BATCH DATA
        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]
        
        #GET DATA BASED ON THE GENERATED INDEXES
        x_train_acc_ = np.array(list(map(lambda k: np.array(self.x_train_acc_[k]).astype(np.float32), indexes)))
        x_train_fr_ = np.array(list(map(lambda k: np.array(self.x_train_fr_[k]).astype(np.float32), indexes)))
        y_train_fr_ = np.array(list(map(lambda k: np.array(self.y_train_fr_[k]).astype(np.float32), indexes)))
        
        # print(x_train_acc_.shape, x_train_fr_.shape, y_train_fr_.shape)
        
        return [x_train_acc_, x_train_fr_], y_train_fr_
    
    def on_epoch_end(self):
        self.indexes = np.arange(len(self.x_train_acc_))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)
        
class Main(object):
    def __init__(self):    
        # PARAMETER
        self.min_pos_rank = 5
        self.img_dim = 16
        self.seq = 20
        self.num_classes = 2
        self.datas_a_ = 120
        
    #LOAD PICKLE OF FRAME
    def dataset_to_pickle_frame(self):  
        # EXTRACT DATSETS
        x_train, y_train, user_lbl_train = ExtractImages().extract(status='train', min_pos_rank=self.min_pos_rank, img_dim=self.img_dim, seq=self.seq)
        x_test, y_test, user_lbl_test = ExtractImages().extract(status='test', min_pos_rank=self.min_pos_rank, img_dim=self.img_dim, seq=self.seq)
        
        # SAVE DATASETS
        PickleDumpLoad().save_config((x_train, y_train, user_lbl_train), 'frame_train_csv_noflip_16_.pickle')
        PickleDumpLoad().save_config((x_test, y_test, user_lbl_test), 'frame_test_csv_noflip_16_.pickle')
    
    #LOAD PICKLE OF ACCELERATION DATA
    def dataset_to_pickle_acc(self):  
        x_train, y_train, user_lbl_train = DataCsv().training_file()
        x_test, y_test, user_lbl_test = DataCsv().testing_file() 
        
        # SAVE DATASETS
        PickleDumpLoad().save_config((x_train, y_train, user_lbl_train), 'acc_train_csv_noflip_16_.pickle')
        PickleDumpLoad().save_config((x_test, y_test, user_lbl_test), 'acc_test_csv_noflip_16_.pickle')
    
    # FRAME
    def load_datasets(self):
        # LOAD THE SAVED DATASETS
        x_train, y_train, user_lbl_train = PickleDumpLoad().load_config('frame_train_csv_noflip_16_.pickle')
        x_test, y_test, user_lbl_test = PickleDumpLoad().load_config('frame_test_csv_noflip_16_.pickle')
         
        return x_train, y_train, user_lbl_train, x_test, y_test, user_lbl_test
    
    # ACCEL
    def load_csv(self):
        # LOAD THE SAVED DATASETS
        x_train, y_train, user_lbl_train = PickleDumpLoad().load_config('acc_train_csv_noflip_16_.pickle')
        x_test, y_test, user_lbl_test = PickleDumpLoad().load_config('acc_test_csv_noflip_16_.pickle') 
        
        return x_train, user_lbl_train, x_test, user_lbl_test, y_test, y_train
    
    def reshape(self):
        # LOAD DATASET
        # FRAME
        x_train_fr, y_train_fr, user_lbl_train_fr = PickleDumpLoad().load_config('frame_train_csv_noflip_16_.pickle')
        x_test_fr, y_test_fr, user_lbl_test_fr = PickleDumpLoad().load_config('frame_test_csv_noflip_16_.pickle')
        
        # ACC
        x_train_acc, y_train_acc, user_lbl_train_acc = PickleDumpLoad().load_config('acc_train_csv_noflip_16_.pickle')
        x_test_acc, y_test_acc, user_lbl_test_acc = PickleDumpLoad().load_config('acc_test_csv_noflip_16_.pickle') 
        #print(x_train_acc.shape)
        #TURN THE DATA INTO ARRAY
        #x_train_acc_ = np.array(x_train_acc)
        #x_test_acc_ = np.array(x_test_acc)
        #y_test_acc_ = np.array(y_test_acc)
     
        #x_train_fr, x_test_fr = x_train_fr.astype(np.float32), x_test_fr.astype(np.float32)
        #x_train_acc, x_test_acc = x_train_acc.astype(np.float32), x_test_acc.astype(np.float32)
        
        # SHUFLLE THE DATA
        idx = np.arange(len(y_train_fr))
        np.random.shuffle(idx)  
        x_train_acc_ = list(map(lambda x: x_train_acc[x], idx))
        x_train_fr_ = list(map(lambda x: x_train_fr[x], idx))
        y_train_fr_ = list(map(lambda x: y_train_fr[x], idx))
        
        x_train_acc_ = np.array(x_train_acc_[:])
        x_train_fr_ = np.array(x_train_fr_[:])
        y_train_fr_ = np.array(y_train_fr_[:])  
        
        x_test_fr = np.array(list(map(lambda x: np.array(x_test_fr[x]).astype(np.float32), np.arange(len(x_test_fr)))))
        x_test_acc = np.array(list(map(lambda x: np.array(x_test_acc[x]).astype(np.float32),  np.arange(len(x_test_acc)))))
        y_test_fr = np.array(list(map(lambda x: np.array(y_test_fr[x]).astype(np.float32),  np.arange(len(y_test_fr)))))
         
        return x_train_fr_, x_train_acc_, y_train_fr_, x_test_fr, x_test_acc, y_test_fr

    def fit_generator(self, x_train_acc_, x_train_fr_, y_train_fr_, x_test_acc, y_test_fr):
        epochs = 100
        batch_size = 64
        
        #filepath = "/media/marvela/fix_research/acceleration_data/short/frame-weights-improvement-{epoch:02d}-{loss:.4f}-bigger.hdf5"
        #checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=True, mode='min')
        #callbacks_list = [checkpoint]   
        
        model = CNN_Model().vgg19_lstm(dimension=self.img_dim, seq=self.seq, classes=self.num_classes, datas_a = self.datas_a_)
        train_datagen = DataGenerator(x_train_acc_, x_train_fr_, y_train_fr_, batch_size=batch_size)     
         
        for epoch in range(epochs):
            print('epochs', epoch, 'from', epochs)
            model.fit(train_datagen)
            # model.fit([np.array(x_train_acc_), np.array(x_train_fr_)], np.array(y_train_fr_), batch_size=batch_size, verbose=1, epochs=1, callbacks=callbacks_list)
            
            # print(x_test_acc.shape, x_test_fr.shape)
            pred = model.predict([x_test_acc, x_test_fr])
            predicted = np.argmax(pred, axis=1)
            accuracy, balanced_accuracy, sensitivity, specitivity, conf_matrix = Assesment().get_performance_evaluation(predicted, np.argmax(y_test_fr, axis=1))
            print('Test: {}' . format(accuracy)) 
# TO EXTRACT DATASETS FRAME
run = Main()

#TO EXTRACT FRAME
#run.dataset_to_pickle_frame()

#TO EXTRACT ACC
#run.dataset_to_pickle_acc()

# TO TRAIN AND TEST

#x_train = [np.array(x_train_acc_), np.array(x_train_fr)]    
#y_train = np.array(y_train_fr)    
#ACC
x_train_fr_, x_train_acc_, y_train_fr_, x_test_fr, x_test_acc, y_test_fr = run.reshape() 

#print(x_test_fr.shape)
#print(x_test_acc.shape)
#x_train = [np.array(x_train_fr), np.array(x_train_acc)]
#y_train = [np.array(y_train_fr)]
 
# print(np.array(x_train_acc_).shape, np.array(x_train_fr_).shape, np.array(y_train_fr_).shape)

run.fit_generator(x_train_acc_, x_train_fr_, y_train_fr_, x_test_acc, y_test_fr)