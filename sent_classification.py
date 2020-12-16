# -*- coding: utf-8 -*-
"""
Created on Sat Nov 14 17:31:18 2020

@author: babdeen
"""
import tensorflow as tf
physical_devices = tf.config.list_physical_devices('GPU')
try:
  tf.config.experimental.set_memory_growth(physical_devices[0], True)
except:
  print('NO GPU')
  
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from data_class import Data
from sklearn.utils import class_weight
from tf_models import NN_models
from sklearn.metrics import recall_score, precision_score
from gensim.models import Word2Vec
import funs
###read data
data_path = './data/'
data = Data(data_path)


###extract SVO from data
max_len = 20
pos_srl_SVO = {}
neg_srl_SVO = {}

pos_srl_tag = {}
neg_srl_tag = {}

for sent in data.all_pos_sents_srl:
    for v in data.all_pos_sents_srl[sent]:
        if 'V' in data.all_pos_sents_srl[sent][v]:  
            srl_sent,srl_tag = data.get_SVO_from_srl(data.all_pos_sents_srl[sent] ,v)

            if sent in data.sent_malicious_verbs and data.all_pos_sents_srl[sent][v]['V']['text'] in data.sent_malicious_verbs[sent]:
                pos_srl_tag[srl_sent] = srl_tag[:max_len] +  ['pad' for i in  range(max_len - len(srl_sent.split()) )]
                if sent in pos_srl_SVO:
                    pos_srl_SVO[sent].append(srl_sent)
                else:
                    pos_srl_SVO[sent] = [srl_sent]
            
            else:
                neg_srl_tag[srl_sent] = srl_tag[:max_len] + ['pad' for i in  range(max_len - len(srl_sent.split()) )]
                if sent in neg_srl_SVO:
                    neg_srl_SVO[sent].append(srl_sent)
                else:
                    neg_srl_SVO[sent] = [srl_sent]
                    

for sent in data.all_neg_sents_srl:
    for v in data.all_neg_sents_srl[sent]:
        if 'V' in data.all_neg_sents_srl[sent][v]: 
            srl_sent,srl_tag = data.get_SVO_from_srl(data.all_neg_sents_srl[sent] ,v)
            neg_srl_tag[srl_sent] = srl_tag[:max_len] + ['pad' for i in  range(max_len - len(srl_sent.split()) )]
            if sent in neg_srl_SVO:
                neg_srl_SVO[sent].append(srl_sent)
            else:
                neg_srl_SVO[sent] = [srl_sent]


### One hot vector encoding SVO tags
# all_labels = set([i for j in list(neg_srl_tag.values()) for i in j ])
# label_to_num = {l:i for i,l in enumerate(all_labels)}
label_to_num = funs.read_json_as_dict('srl_ohv.json')

pos_srl_tag_OHV = {}
for sent in pos_srl_tag:
    pos_srl_tag_OHV[sent] = data.get_OHV(pos_srl_tag[sent],label_to_num)

neg_srl_tag_OHV = {}
for sent in neg_srl_tag:
    neg_srl_tag_OHV[sent] = data.get_OHV(neg_srl_tag[sent],label_to_num)


###get all SVOs
all_pos_srl_SVO = list(set([i for j in list(pos_srl_SVO.values()) for i in j]))
all_neg_srl_SVO = list(set([i for j in list(neg_srl_SVO.values()) for i in j]))


### read word embedding model
emb_model = Word2Vec.load("cyber.model")

###extract WE features
pos_sentence_features_w_emb = {}
for sent in all_pos_srl_SVO:
    pos_sentence_features_w_emb[sent] = data.get_word_emb(sent,emb_model, max_len )
        
neg_sentence_features_w_emb = {}
for sent in all_neg_srl_SVO:
   neg_sentence_features_w_emb[sent] = data.get_word_emb(sent,emb_model, max_len )


###split data to train/val/test
train_pos_SVO = list(set([i for j in [pos_srl_SVO[i] for i in pos_srl_SVO if i in list(data.train_sents_pos.values())] for i in j]))
train_neg_SVO =  list(set([i for j in [neg_srl_SVO[i] for i in neg_srl_SVO if i in list(data.train_sents_neg.values())] for i in j]))

val_pos_SVO =  list(set([i for j in [pos_srl_SVO[i] for i in pos_srl_SVO if i in list(data.val_sents_pos.values())] for i in j]))
val_neg_SVO =  list(set([i for j in [neg_srl_SVO[i] for i in neg_srl_SVO if i in list(data.val_sents_neg.values())] for i in j]))

test_pos_SVO =  list(set([i for j in [pos_srl_SVO[i] for i in pos_srl_SVO if i in list(data.test_sents_pos.values())] for i in j]))
test_neg_SVO =  list(set([i for j in [neg_srl_SVO[i] for i in neg_srl_SVO if i in list(data.test_sents_neg.values())] for i in j]))


###get data featres (WE + SRL)
X_train_emb = np.array([(pos_sentence_features_w_emb[i]) for i in train_pos_SVO] + [neg_sentence_features_w_emb[i] for i in train_neg_SVO])
X_train_srl_tag = np.array([pos_srl_tag_OHV[i] for i in train_pos_SVO] + [neg_srl_tag_OHV[i] for i in train_neg_SVO])
X_train = np.concatenate((X_train_emb,X_train_srl_tag),axis = 2)

X_val_emb = np.array([pos_sentence_features_w_emb[i] for i in val_pos_SVO] + [neg_sentence_features_w_emb[i]  for i in val_neg_SVO])
X_val_srl_tag = np.array([pos_srl_tag_OHV[i] for i in val_pos_SVO] + [neg_srl_tag_OHV[i] for i in val_neg_SVO])
X_val = np.concatenate((X_val_emb,X_val_srl_tag),axis = 2)

X_test_emb = np.array([pos_sentence_features_w_emb[i] for i in test_pos_SVO] + [neg_sentence_features_w_emb[i][:max_len]  for i in test_neg_SVO])
X_test_srl_tag = np.array([pos_srl_tag_OHV[i] for i in test_pos_SVO] + [neg_srl_tag_OHV[i] for i in test_neg_SVO])
X_test= np.concatenate((X_test_emb,X_test_srl_tag),axis = 2)

### labels
y_train = [1 for i in range(len(train_pos_SVO))] + [0 for i in range(len(train_neg_SVO))]
y_val =  [1 for i in range(len(val_pos_SVO))] + [0 for i in range(len(val_neg_SVO))]
y_test =  [1 for i in range(len(test_pos_SVO))] + [0 for i in range(len(test_neg_SVO))]


### get each class weights for training 
class_weights = class_weight.compute_class_weight('balanced', np.unique(y_train), y_train)
class_weights_dict = {i : class_weights[i] for i in range(len(class_weights))}
        

###build model
input_shape = X_train.shape[1:]
parameters  = {'input_shape':input_shape , 'FC_dims':[16,4], 'lstm_dims':[4], 'cnn_filters':[16], 'kernel_size':[3], 'pool_size':[2]}
model = NN_models.lstm_cnn_SC(parameters)

#compile model
model.compile(loss='binary_crossentropy',
          optimizer='adam',
          metrics=['accuracy'])
   
#train model
batch_size = 8
epochs = 500
lr = .0005
patience = 5
opt = tf.keras.optimizers.Adam(learning_rate=lr)
model.compile(optimizer=opt, loss="binary_crossentropy", metrics=["accuracy"])

###early stopping 
callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=patience, restore_best_weights= True)

y_train, y_val = np.array(y_train), np.array(y_val)
###train
model.fit(X_train, y_train,
      batch_size=batch_size,
      epochs=epochs,
      shuffle=True,
      validation_data=(X_val, y_val),
      callbacks = [callback],
      class_weight= class_weights_dict)

# model.save('./SVO_model.h5')
###evaluate 
NN_models.eval_SC(model,X_test,y_test)


all_acc = []
all_f1 = []
all_presicion = []
all_recall = []
for thr in range(11):

###evalute per senteces
    y_pred = model.predict(X_test)
    svo_pred = dict( zip(test_pos_SVO+test_neg_SVO,np.squeeze(y_pred).tolist()) )
    y_true = [1 for i in data.test_sents_pos if data.test_sents_pos[i] in pos_srl_SVO] + \
                [0 for i in data.test_sents_neg if data.test_sents_neg[i] in neg_srl_SVO]
        
    y_pred = []
    for i in data.test_sents_pos:
            sent = data.test_sents_pos[i]
            if sent in pos_srl_SVO:
                y_pred.append(0)
                for SVO in pos_srl_SVO[sent]:
                    if svo_pred[SVO] > thr * .1:
                        y_pred[-1] = 1
                        break
            
            
    for i in data.test_sents_neg:
            sent = data.test_sents_neg[i]
            if sent in neg_srl_SVO:
                y_pred.append(0)
                for SVO in neg_srl_SVO[sent]:
                    if svo_pred[SVO] > thr * .1:
                        y_pred[-1] = 1
                        break
    all_f1.append(f1_score(y_true,y_pred))
    all_presicion.append(precision_score(y_true,y_pred))
    all_recall.append(recall_score(y_true,y_pred))
    all_acc.append(accuracy_score(y_true,y_pred))
    
    
    print('Acc: ',accuracy_score(y_true,y_pred))
    print('F1: ' ,f1_score(y_true,y_pred))
    print('Precision: ',precision_score(y_true,y_pred))
    print('Recall: ',recall_score(y_true,y_pred))
    print('Confusion_matrix: \n',confusion_matrix(y_true,y_pred))





###

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt

acc,f1,pre,rec,cm = NN_models.eval_SC(model,X_test,y_test)
results = {'acc':acc, 'f1':f1,'precision':pre,'recall':rec}


#results vs threshold
all_acc = []
all_f1 = []
all_presicion = []
all_recall = []
for thr in range(11):
    acc,f1,pre,rec,cm = NN_models.eval_SC(model,X_test,y_test,thr * .1)
    all_f1.append(f1)
    all_presicion.append(pre)
    all_recall.append(rec)
    all_acc.append(acc)




plt.plot([i * .1 for i in range(11)],all_f1, color = 'g')
plt.plot([i * .1 for i in range(11)],all_presicion, color = 'r')
plt.plot([i * .1 for i in range(11)],all_recall, color = 'b')
plt.plot([i * .1 for i in range(11)],all_acc, color = 'k')
red_patch = mpatches.Patch(color='green', label='F1 score')
blue_patch = mpatches.Patch(color='red', label='Presicion')
black_patch = mpatches.Patch(color='blue', label='Recall')
green_patch = mpatches.Patch(color='black', label='Accuracy')
plt.legend(handles=[black_patch, red_patch,blue_patch,green_patch])
plt.xlabel('Threshold')
plt.ylabel('Score')
plt.title('Threshold Vs Scores')




        


    