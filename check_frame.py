#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 17 05:32:43 2021

@author: marvela
"""


import cv2
import os
import numpy as np 
import pickle
import glob
import pandas as pd
from sklearn.utils import shuffle 
from sklearn.metrics import accuracy_score
from sklearn.metrics import balanced_accuracy_score
from sklearn.metrics import confusion_matrix
from imblearn.metrics import sensitivity_specificity_support
from tensorflow.keras.layers import Input, Dense, Flatten, TimeDistributed, Conv2D, Activation, MaxPooling2D
from tensorflow.keras.layers import LSTM
from tensorflow.keras import optimizers
from tensorflow.keras import Model as ModelKeras 
from tensorflow.keras.callbacks import ModelCheckpoint

import tensorflow as tf
tf.__version__
# tf.compat.v1.disable_v2_behavior()
# print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))


class PickleDumpLoad(object): 
    def __init__(self):
        self.address = '/media/marvela/fix_research'
        
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

class DataCsv(object):
    def training_file(self):
        basepath_training = '/media/marvela/fix_research/acceleration_data/fix_data_training/'
        csv_training = glob.glob(os.path.join(basepath_training, '*.csv'))
         
        dataframes_trn = []
        user_lbl = [] 
        for file_training in csv_training: 
            a = [z for z in [[y for y in x.split(',') if y != ''] for x in open(file_training, 'r').read().split('\n')] if len(z) > 4]
            # df_trn = pd.read_csv(csv_training[-1])
            # d_new = df_trn.dropna(how='all')
            dataframes_trn.append(a)
            user_lbl.append(file_training.split('/')[-1].split('.')[0])
            # print("Training : ")
            # print(a) 
            
        return dataframes_trn, user_lbl
    
    def testing_file(self):
        basepath_testing = '/media/marvela/fix_research/acceleration_data/fix_data_testing/'
        csv_testing = glob.glob(os.path.join(basepath_testing, '*.csv'))
        
        dataframes_tst = []
        user_lbl = []
        for file_testing in csv_testing:
            a = [z for z in [[y for y in x.split(',') if y != ''] for x in open(file_testing, 'r').read().split('\n')] if len(z) > 4]
            dataframes_tst.append(a)
            user_lbl.append(file_testing.split('/')[-1].split('.')[0])
#            df_tst = pd.read_csv(file_testing)
#            dataframes_tst.append(df_tst)
#            print("Testing : ")
#            print(df_tst)
        return dataframes_tst, user_lbl
    
class ExtractImages(object):
    def extract(self, status='train', min_pos_rank=5, img_dim=32, seq=20):
        
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
            sp = label.split(' ')
            first_rank = int(sp[0])
            second_rank = int(sp[2])
            label = 'PAIR' if first_rank >= min_pos_rank and second_rank >= min_pos_rank else 'UNPAIR'
            
            lst = os.listdir(add)
            
            pre_x_train = []
            pre_y_train = []
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
                            pre_x_train.append(np.array(split_data))
                            pre_y_train.append(label) 
                            split_data = list()
                        elif len(split_data) > 20:
                            break

                   
                print('Done! Current total sets {}.' . format(np.array(pre_x_train).shape))
                
            x_train.append(pre_x_train)
            y_train.append(pre_y_train)
            
        # LABEL ENCODER
        # y_train = EncodeLabelsCategorically().manual_categorical_labeling(label_to_encode=y_train, num_classes=2) 
        
        print('TRAIN SETS', np.array(x_train).shape)
        print('LABEL SETS', np.array(y_train).shape)
        return np.array(x_train), np.array(y_train), np.array(user_lbl)
 
class Assesment(object):
    def get_performance_evaluation(self, y_pred, y_test): 
        accuracy = accuracy_score(y_pred, y_test)
        balanced_accuracy = balanced_accuracy_score(y_pred, y_test)
        # PARAMETER
        '''self.min_pos_rank = 5
        self.img_dim = 32
        self.seq = 20
        self.num_classes = 2
        self.epochs = 30
    
    def dataset_to_pickle(self):  
        # EXTRACT DATSETS 
        x_train, y_train = ExtractImages().extract(status='train', min_pos_rank=self.min_pos_rank, img_dim=self.img_dim, seq=self.seq)
        x_test, y_test = ExtractImages().extract(status='test', min_pos_rank=self.min_pos_rank, img_dim=self.img_dim, seq=self.seq)

        # SAVE DATASETS
        PickleDumpLoad().save_config((x_train, y_train), 'train_full.pickle')
        PickleDumpLoad().save_config((x_test, y_test), 'test_full.pickle')
    
    # FRAME
    def load_datasets(self):
        # LOAD THE SAVED DATASETSuracy_score(y_pred, y_test) '''
        sensitivity, specitivity, support = sensitivity_specificity_support(y_pred, y_test, average="micro")
        conf_matrix = confusion_matrix(y_pred, y_test)
        return accuracy, balanced_accuracy, sensitivity, specitivity, conf_matrix
    
class CNN_Model(object):   
    def vgg19_lstm(self, dimension=32, seq=20, classes=2):   
        input_shape = Input(shape=(seq, dimension, dimension, 3))  
        
        # 64        
        x1 = TimeDistributed(Conv2D(64, (3,3), padding='same', data_format='channels_last'))(input_shape)
        x1 = TimeDistributed(Activation('relu'))(x1)
        x1 = TimeDistributed(Conv2D(64, (3,3), padding='same', data_format='channels_last'))(x1)
        x1 = TimeDistributed(Activation('relu'))(x1)
        x1 = TimeDistributed(MaxPooling2D(pool_size=(2,2), strides=(2, 2)))(x1)
        
        #128
        x1 = TimeDistributed(Conv2D(128, (3,3), padding='same', data_format='channels_last'))(x1)
        x1 = TimeDistributed(Activation('relu'))(x1)
        x1 = TimeDistributed(Conv2D(128, (3,3), padding='same', data_format='channels_last'))(x1)
        x1 = TimeDistributed(Activation('relu'))(x1)
        x1 = TimeDistributed(MaxPooling2D(pool_size=(2,2), strides=(2, 2)))(x1)
        
        #256
        x1 = TimeDistributed(Conv2D(256, (3,3), padding='same', data_format='channels_last'))(x1)
        x1 = TimeDistributed(Activation('relu'))(x1)
        x1 = TimeDistributed(Conv2D(256, (3,3), padding='same', data_format='channels_last'))(x1)
        x1 = TimeDistributed(Activation('relu'))(x1)
        x1 = TimeDistributed(Conv2D(256, (3,3), padding='same', data_format='channels_last'))(x1)
        x1 = TimeDistributed(Activation('relu'))(x1)
        x1 = TimeDistributed(Conv2D(256, (3,3), padding='same', data_format='channels_last'))(x1)
        x1 = TimeDistributed(Activation('relu'))(x1)
        x1 = TimeDistributed(MaxPooling2D(pool_size=(2,2), strides=(2, 2)))(x1)
        
        #512
        x1 = TimeDistributed(Conv2D(512, (3,3), padding='same', data_format='channels_last'))(x1)
        x1 = TimeDistributed(Activation('relu'))(x1)
        x1 = TimeDistributed(Conv2D(512, (3,3), padding='same', data_format='channels_last'))(x1)
        x1 = TimeDistributed(Activation('relu'))(x1)
        x1 = TimeDistributed(Conv2D(512, (3,3), padding='same', data_format='channels_last'))(x1)
        x1 = TimeDistributed(Activation('relu'))(x1)
        x1 = TimeDistributed(Conv2D(512, (3,3), padding='same', data_format='channels_last'))(x1)
        x1 = TimeDistributed(Activation('relu'))(x1)
        x1 = TimeDistributed(MaxPooling2D(pool_size=(2,2), strides=(2, 2)))(x1)
        
        x1 = TimeDistributed(Conv2D(512, (3,3), padding='same', data_format='channels_last'))(x1)
        x1 = TimeDistributed(Activation('relu'))(x1)
        x1 = TimeDistributed(Conv2D(512, (3,3), padding='same', data_format='channels_last'))(x1)
        x1 = TimeDistributed(Activation('relu'))(x1)
        x1 = TimeDistributed(Conv2D(512, (3,3), padding='same', data_format='channels_last'))(x1)
        x1 = TimeDistributed(Activation('relu'))(x1)
        x1 = TimeDistributed(Conv2D(512, (3,3), padding='same', data_format='channels_last'))(x1)
        x1 = TimeDistributed(Activation('relu'))(x1)
        x1 = TimeDistributed(MaxPooling2D(pool_size=(2,2), strides=(2, 2)))(x1)
        
        x1 = TimeDistributed(Flatten())(x1) 
        
        x = Dense(512, activation='relu')(x1) 
        x = LSTM(512, return_sequences=False)(x)
        x = Dense(512, activation='relu')(x)  
        x = Dense(classes, activation='softmax', name='predictions')(x) 
        model = ModelKeras(inputs=input_shape, outputs=x)
        op = optimizers.Adam(lr=0.0001) 
        model.compile(loss='categorical_crossentropy', optimizer=op, metrics=['accuracy']) 
        model.summary() 
        return model  
   

class Main(object):
    def __init__(self):    
        # PARAMETER
        self.min_pos_rank = 5
        self.img_dim = 32
        self.seq = 20
        self.num_classes = 2
        self.epochs = 30
    
    def dataset_to_pickle_frame(self):  
        # EXTRACT DATSETS
        x_train, y_train, user_lbl_train = ExtractImages().extract(status='train', min_pos_rank=self.min_pos_rank, img_dim=self.img_dim, seq=self.seq)
        x_test, y_test, user_lbl_test = ExtractImages().extract(status='test', min_pos_rank=self.min_pos_rank, img_dim=self.img_dim, seq=self.seq)

        # SAVE DATASETS
        PickleDumpLoad().save_config((x_train, y_train, user_lbl_train), 'frame_train_full.pickle')
        PickleDumpLoad().save_config((x_test, y_test, user_lbl_test), 'frame_test_full.pickle')
    
    def dataset_to_pickle_acc(self):  
        x_train, user_lbl_train = DataCsv().training_file()
        x_test, user_lbl_test = DataCsv().testing_file() 
        
        # SAVE DATASETS
        PickleDumpLoad().save_config((x_train, user_lbl_train), 'acc_train_full.pickle')
        PickleDumpLoad().save_config((x_test, user_lbl_test), 'acc_test_full.pickle')
        
        
    # FRAME
    def load_datasets(self):
        # LOAD THE SAVED DATASETS
        x_train, y_train, user_lbl_train = PickleDumpLoad().load_config('frame_train_full.pickle')
        x_test, y_test, user_lbl_test = PickleDumpLoad().load_config('frame_test_full.pickle')
         
        return x_train, y_train, user_lbl_train, x_test, y_test, user_lbl_test
    
    # ACCEL
    def load_csv(self):
        # LOAD THE SAVED DATASETS
        x_train, user_lbl_train = PickleDumpLoad().load_config('acc_train_full.pickle')
        x_test, user_lbl_test = PickleDumpLoad().load_config('acc_test_full.pickle') 
        return x_train, user_lbl_train, x_test, user_lbl_test
    
    def concat(self):
        # FRAME
        x_train_fr, y_train_fr, user_lbl_train_fr = PickleDumpLoad().load_config('frame_train_full.pickle')
        x_test_fr, y_test_fr, user_lbl_test_fr = PickleDumpLoad().load_config('frame_test_full.pickle')
        
        
            
        # ACC
        x_train_acc, user_lbl_train_acc = PickleDumpLoad().load_config('acc_train_full.pickle')
        x_test_acc, user_lbl_test_acc = PickleDumpLoad().load_config('acc_test_full.pickle') 
        
        #print(np.array(x_train_fr).shape, np.array(x_train_acc).shape)
        #print(np.array(x_train_fr[2]).shape, np.array(x_train_acc[2]).shape)
        #input()
         
        for x in range(len(user_lbl_train_fr)): 
            print('ACC:', np.array(x_train_acc[x]).shape, 'FRAME:', np.array(x_train_fr[x]).shape, (int(np.array(x_train_acc[x]).shape[0]) // int(np.array(x_train_fr[x]).shape[0])))
        
        print(user_lbl_train_acc)
        
        x_train_concat = np.concatenate([x_train_fr, x_train_acc], axis=0)
        x_test_concat = np.concatenate([x_test_fr, x_test_acc], axis=0)
        #user_lbl_concat = np.concatenate()
        
        return x_train_concat, x_test_concat, y_train_fr, y_test_fr, user_lbl_train_acc, user_lbl_test_acc, user_lbl_train_fr, user_lbl_test_fr
    
    def run(self):    
        # LOAD DATASETS
        x_train_concat, x_test_concat, user_lbl_train_acc, user_lbl_test_acc, y_train_fr, y_test_fr= self.concat()
        
        # SHUFFLE THE DATA
        x_train_concat, y_train_fr = shuffle(x_train_concat, y_train_fr)
        x_test_concat, user_label_test_fr = shuffle(x_test_concat, y_test_fr)
        
        x_train_concat, x_test_concat = x_train_concat.astype(np.float32), x_test_concat.astype(np.float32)
#       #y_train, y_test = y_train.astype(np.int32), y_test.astype(np.int32)
#        
#        #print(x_train.shape, y_train.shape)
#        #print(x_test.shape, y_test.shape)
         
        # TRAINING SECTION
         
        filepath = "media/marvela/fix_research/weights-improvement-{epoch:02d}-{loss:.4f}-bigger.hdf5"
        checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=True, mode='min')
        callbacks_list = [checkpoint]   
        
        model = CNN_Model().vgg19_lstm(dimension=self.img_dim, seq=self.seq, classes=self.num_classes) 
        for ep in range(self.epochs):
            print('epoch', ep, 'from', self.epochs)
            model.fit(x_train_concat, y_train_fr, batch_size=128, verbose=1, epochs=1, callbacks=callbacks_list)
            pred = model.predict(x_test_concat)  
            predicted = np.argmax(pred, axis=1)
            accuracy, balanced_accuracy, sensitivity, specitivity, conf_matrix = Assesment().get_performance_evaluation(predicted, np.argmax(y_test_fr, axis=1))
            print('Test: {}' . format(accuracy))
    

run = Main()

# TO EXTRACT DATASETS "FRAME"
run.dataset_to_pickle_frame()

# EXTRACTING DATASETS "ACC"
run.dataset_to_pickle_acc()

# TO TRAIN AND TEST    
run.run() 
