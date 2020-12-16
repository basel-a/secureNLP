# -*- coding: utf-8 -*-
"""
Created on Sun Nov 22 13:02:25 2020

@author: babdeen
"""


from tensorflow.keras.models import Model 
from tensorflow.keras.layers import Dense, Input, LSTM ,Bidirectional , TimeDistributed, Flatten
from sklearn.metrics import confusion_matrix,f1_score,recall_score, precision_score,accuracy_score
from tensorflow.keras.layers import MaxPooling1D,Conv1D, GlobalMaxPooling1D
import numpy as np


class NN_models:
    def lstm_seq_tag(parameters):
        
        lstm_dims = parameters['lstm_dims']
        FC_dims = parameters['FC_dims']
        input_shape = parameters['input_shape']
        outputs = parameters['outputs']
        
        inp = Input(shape=(input_shape))
        model = Bidirectional(LSTM(units=lstm_dims[0], return_sequences=True))(inp)  # variational biLSTM
        for l in lstm_dims[1:]:
            model = Bidirectional(LSTM(units=l, return_sequences=True))(model)  # variational biLSTM
        for l in FC_dims:
            model = TimeDistributed(Dense(l,activation= 'relu'))(model)  # softmax output layer
        out = TimeDistributed(Dense(outputs,activation= 'softmax'))(model)  # softmax output layer
        model = Model(inp, out)
        model.summary()
        return model

    def eval_seq_tag(model,x_test,y_test, thr = .5):
        y_pred = model.predict(x_test)
        pred = [0 if i <thr else 1 for i in y_pred.ravel()]
        true = y_test.ravel()

        print('general evaluation')
        print('Acc: ',accuracy_score(true,pred))
        print('F1: ' ,f1_score(true,pred))
        print('Precision: ',precision_score(true,pred))
        print('Recall: ',recall_score(true,pred))
        print('Confusion_matrix: \n',confusion_matrix(true,pred))
        
        return (accuracy_score(true,pred),f1_score(true,pred),precision_score(true,pred),recall_score(true,pred),confusion_matrix(true,pred)  )
        
    def lstm_SC(parameters):
        lstm_dims = parameters['lstm_dims']
        FC_dims = parameters['FC_dims']
        input_shape = parameters['input_shape']
        inp = Input(shape=(input_shape))
        model = Bidirectional(LSTM(units=lstm_dims[0], return_sequences=True))(inp)  # variational biLSTM
        for l in lstm_dims[1:]:
            model = Bidirectional(LSTM(units=l, return_sequences=True))(model)  # variational biLSTM
        model = Flatten()(model)  

        for l in FC_dims:
            model = (Dense(l,activation= 'relu'))(model)  # softmax output layer
        out = (Dense(1,activation= 'sigmoid'))(model)  # softmax output layer
        model = Model(inp, out)
        model.summary()
        return model
    
    def lstm_cnn_SC(parameters):
        lstm_dims = parameters['lstm_dims']
        FC_dims = parameters['FC_dims']
        cnn_filters = parameters['cnn_filters']
        kernel_size = parameters['kernel_size']
        pool_size = parameters['pool_size']
        
        input_shape = parameters['input_shape']
        inp = Input(shape=(input_shape))
        model = Bidirectional(LSTM(units=lstm_dims[0], return_sequences=True))(inp)  # variational biLSTM
        for l in lstm_dims[1:]:
            model = Bidirectional(LSTM(units=l, return_sequences=True))(model)  # variational biLSTM        
        
        model = Conv1D(cnn_filters[0], kernel_size[0] , activation='relu')(model)
        model = MaxPooling1D(pool_size[0])(model)
        for l in range(1,len(cnn_filters)):
            model = Conv1D(cnn_filters[l], kernel_size[l] , activation='relu')(model)
            model = MaxPooling1D(pool_size[l])(model)            
            
            
        model = GlobalMaxPooling1D()(model)  
        
        for l in FC_dims:
            model = (Dense(l,activation= 'relu'))(model)  # softmax output layer
        out = (Dense(1,activation= 'sigmoid'))(model)  # softmax output layer
        model = Model(inp, out)
        model.summary()
        return model
    
    def cnn_lstm_SC(parameters):
        lstm_dims = parameters['lstm_dims']
        FC_dims = parameters['FC_dims']
        cnn_filters = parameters['cnn_filters']
        kernel_size = parameters['kernel_size']
        pool_size = parameters['pool_size']
        
        input_shape = parameters['input_shape']
        inp = Input(shape=(input_shape))
        
        model = Conv1D(cnn_filters[0], kernel_size[0] , activation='relu')(inp)
        model = MaxPooling1D(pool_size[0])(model)
        for l in range(1,len(cnn_filters)):
            model = Conv1D(cnn_filters[l], kernel_size[l] , activation='relu')(model)
            model = MaxPooling1D(pool_size[l])(model) 

            
        model = Bidirectional(LSTM(units=lstm_dims[0], return_sequences=True))(model)  # variational biLSTM
        for l in lstm_dims[1:]:
            model = Bidirectional(LSTM(units=l, return_sequences=True))(model)  # variational biLSTM        
        
        model = GlobalMaxPooling1D()(model)  

        
        for l in FC_dims:
            model = (Dense(l,activation= 'relu'))(model)  # softmax output layer
        out = (Dense(1,activation= 'sigmoid'))(model)  # softmax output layer
        model = Model(inp, out)
        model.summary()
        return model
    
    
    def eval_SC(model,x_test,true, thr = .5):
        y_pred = model.predict(x_test)
        # y_pred_list = y_pred.squeeze().tolist()
        pred = [1 if i > thr else 0 for i in y_pred]
        
        print('Acc: ',accuracy_score(true,pred))
        print('F1: ' ,f1_score(true,pred))
        print('Precision: ',precision_score(true,pred))
        print('Recall: ',recall_score(true,pred))
        print('Confusion_matrix: \n',confusion_matrix(true,pred))
        return (accuracy_score(true,pred),f1_score(true,pred),precision_score(true,pred),recall_score(true,pred),confusion_matrix(true,pred)  )
    
    def predict_SC(model,x_test,y_test, thr = .5):
        y_pred = model.predict(x_test)
        return  [1 if i > thr else 0 for i in y_pred]
        
    def get_sents_results(y_true, y_pred):
        TP = []
        FP = []
        TN = []
        FN = []
        for i in range(len(y_true)): 
            if y_true[i]==y_pred[i]==1:
               TP.append(i)
            if y_pred[i]==1 and y_true[i]!=y_pred[i]:
               FP.append(i)
            if y_true[i]==y_pred[i]==0:
               TN.append(i)
            if y_pred[i]==0 and y_true[i]!=y_pred[i]:
               FN.append(i)
    
        return(TP, FP, TN, FN)
    
    def loadGloveModel(File):
        print("Loading Glove Model")
        f = open(File, 'r', encoding = 'utf-8')
        gloveModel = {}
        for line in f:
            splitLines = line.split()
            word = splitLines[0]
            wordEmbedding = np.array([float(value) for value in splitLines[1:]])
            gloveModel[word] = wordEmbedding
        print(len(gloveModel)," words loaded!")
        return gloveModel
