# -*- coding: utf-8 -*-
"""
Created on Thu Nov 26 15:34:53 2020

@author: babdeen
"""


import tensorflow as tf
physical_devices = tf.config.list_physical_devices('GPU')
try:
  tf.config.experimental.set_memory_growth(physical_devices[0], True)
except:
  print('NO GPU')
  
from sklearn.metrics import confusion_matrix,f1_score,recall_score, precision_score,accuracy_score
import funs
import os
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences
import utils
from tf_models import NN_models
from gensim.models import Word2Vec
import random

#parsers one hot vector encoders
OHV_encoders_dict = funs.read_json_as_dict('parseres_ohv.json')


###read data
data_tagged_dict = {}
for folder in ['train','dev','test_1','test_2','test_3']:
    files = os.listdir('./data/MalwareTextDB-2.0/data/'+ folder +'/tokenized')
    data_tagged_dict[folder] = []
    for f in files:
        with open('./data/MalwareTextDB-2.0/data/'+ folder +'/tokenized/' + f, encoding = 'utf-8') as file:
            data_tagged = file.read()
            data_tagged_dict[folder].append(data_tagged)
         
            
max_len = 30
number_of_verbs = 4

train_sentences, train_senteces_labels, train_mv = utils.get_pos_sents(data_tagged_dict['train'])
val_sentences, val_senteces_labels, val_mv = utils.get_pos_sents(data_tagged_dict['dev'])
test_sentences, test_senteces_labels, test_mv = utils.get_pos_sents(data_tagged_dict['test_1'] + data_tagged_dict['test_2'] + data_tagged_dict['test_3'] )

senteces_sets = {'train': train_sentences, 'val':val_sentences, 'test': test_sentences}
sentences_labels = {** train_senteces_labels, **val_senteces_labels,**test_senteces_labels}
sentences_mv = {**train_mv,**val_mv,**test_mv}
sentences = train_sentences + val_sentences + test_sentences

pos_sents = funs.read_json_as_dict('./data/pos_sents.json')
pos_sents_srl = funs.read_json_as_dict('./data/pos_sents_srl.json')
pos_sents_srl = {pos_sents[i]:pos_sents_srl[i] for i in pos_sents_srl }

pos_sents_val = funs.read_json_as_dict('./data/pos_sentences_val.json')
pos_sents_srl_val = funs.read_json_as_dict('./data/pos_sentences_val_srl.json')
pos_sents_srl_val = {pos_sents_val[i]:pos_sents_srl_val[i] for i in pos_sents_srl_val }

pos_sents_test = funs.read_json_as_dict('./data/pos_sentences_test.json')
pos_sents_srl_test = funs.read_json_as_dict('./data/pos_sentences_test_srl.json')
pos_sents_srl_test = {pos_sents_test[i]:pos_sents_srl_test[i] for i in pos_sents_srl_test }


pos_sents_srl.update(pos_sents_srl_val)
pos_sents_srl.update(pos_sents_srl_test)






important_tags = ['pad','O','B-V','B-ARG0','B-ARG1','B-ARG2','B-ARG3','I-ARG0','I-ARG1','I-ARG2','I-ARG3']
labeles_to_num = {label:i for i,label in enumerate(important_tags)}

labeles_to_num = {'pad':0,'O':1,'B-V':2,'B-ARG0':3,'B-ARG1':3,'B-ARG2':3,'B-ARG3':3,'I-ARG0':4,'I-ARG1':4,'I-ARG2':4,'I-ARG3':4}
number_of_num = len(set(labeles_to_num.values()))

empty_tag = [labeles_to_num['O'] for i in range(max_len)]


sents_srl_features = {}
for sent in pos_sents_srl:
    srl = pos_sents_srl[sent]
    k = 0
    srl_features = []
    for v in srl['verbs']:
        tags = utils.clean_list(v['tags'],important_tags,'O' )
        tags = tags[:max_len] + ['pad' for i in range(max_len - len(tags))]
        tags_num = [labeles_to_num[i] for i in tags]
        srl_features.append(tags_num)
        k+=1
        if k == number_of_verbs:
            break
    srl_features = srl_features + [empty_tag for i in range(number_of_verbs - len(srl_features))]
    sents_srl_features[sent] = tf.keras.utils.to_categorical(np.array(srl_features),num_classes=number_of_num )
    sents_srl_features[sent] = sents_srl_features[sent].reshape(30,number_of_num *number_of_verbs )








###parse sentences
max_len = 30
sentences_tokens = {}
for sent in sentences:
    sentences_tokens[sent]= sent.split()
sentences_parsing = utils.parse_senteces(sentences_tokens)

### align labels 
new_sentences_labels = {}
for sent in sentences_parsing:
    old_labels = sentences_labels[sent]
    old_tokens = sentences_tokens[sent]
    new_tokens = sentences_parsing[sent]['words']
    new_labels = utils.align_tokens(old_tokens,new_tokens,old_labels)
    new_sentences_labels[sent] = new_labels

###pad sentences
new_sentences_labels_padded = {}
sentences_parsing_padded = {}
keys = ['words','pos','dep']
for sent in sentences_parsing:
    sentences_parsing_padded[sent] = {}
    for t in keys:
        sentences_parsing_padded[sent][t] = pad_sequences([sentences_parsing[sent][t]], maxlen = max_len,padding = 'post',truncating = 'post',dtype=object, value = 'pad')[0]
    new_sentences_labels_padded[sent] =  pad_sequences([new_sentences_labels[sent]], maxlen = max_len,padding = 'post',truncating = 'post',dtype=object, value = 'pad')[0]


####W@V
w2v_model = Word2Vec.load("cyber.model")



###extract sentences features
sentences_features = utils.extract_senteces_features(sentences_parsing_padded, w2v_model, OHV_encoders_dict)

all_features = {}
for sent in sents_srl_features:
    if sent not in sentences_features:
        print(1)
        continue 
    print(2)
    all_features[sent] = np.concatenate((sentences_features[sent], sents_srl_features[sent]),axis = 1)
    # all_features[sent] = sentences_features[sent]

x_train = np.array([np.array(all_features[sent]) for sent in senteces_sets['train'] if sent in all_features])
x_val = np.array([np.array(all_features[sent]) for sent in senteces_sets['val']if sent in all_features])
x_test = np.array([np.array(all_features[sent]) for sent in senteces_sets['test']if sent in all_features])

 
### one hot vector encoding 
train_labels = np.array([np.array(new_sentences_labels_padded[sent]) for sent in  senteces_sets['train'] if sent in all_features])
val_labels = np.array([np.array(new_sentences_labels_padded[sent]) for sent in senteces_sets['val'] if sent in all_features])
test_labels = np.array([np.array(new_sentences_labels_padded[sent]) for sent in senteces_sets['test'] if sent in all_features])

all_labels = set([i for j in list(new_sentences_labels_padded.values()) for i in j])
label_to_num = {l:i for i,l in enumerate(all_labels)}

y_train_class = np.vectorize(lambda x:label_to_num[x] )(train_labels)
y_val_class =np.vectorize(lambda x:label_to_num[x] )(val_labels)
y_test_class =np.vectorize(lambda x:label_to_num[x] )(test_labels)

y_train = tf.keras.utils.to_categorical(y_train_class )
y_val = tf.keras.utils.to_categorical(y_val_class)
y_test = tf.keras.utils.to_categorical(y_test_class)


####


# indicies of the verb 
# senteces_mv_ind = {}
# for sent in sentences_parsing_padded:
#     senteces_mv_ind[sent] = np.zeros(max_len)
#     for i,w in enumerate(sentences_parsing_padded[sent]['words']):
#         if w in sentences_mv[sent]:
#             senteces_mv_ind[sent][i] = 1
            
# X_train_mv = np.expand_dims([senteces_mv_ind[sent] for sent in senteces_sets['train']if sent in all_features],axis = 2)
# X_val_mv = np.expand_dims([senteces_mv_ind[sent] for sent in senteces_sets['val']if sent in all_features],axis = 2)
# X_test_mv = np.expand_dims([senteces_mv_ind[sent] for sent in senteces_sets['test']if sent in all_features],axis = 2)


# x_train = np.concatenate((x_train, X_train_mv), axis = 2)
# x_val = np.concatenate((x_val, X_val_mv), axis = 2)
# x_test = np.concatenate((x_test, X_test_mv), axis = 2)

###
input_shape = x_train.shape[1:]
            
parameters  = {'input_shape':input_shape ,  'lstm_dims':[16], 'FC_dims':[8,8], 'outputs': 8 } 
model = NN_models.lstm_seq_tag(parameters)



#######
#parameters 
batch_size = 4
epochs = 500
lr = .0001
patience = 5

opt = tf.keras.optimizers.Adam(learning_rate=lr)
model.compile(optimizer=opt, loss="categorical_crossentropy", metrics=["accuracy"])
callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=patience,restore_best_weights= True)

##training
model.fit(x_train, y_train,
      batch_size=batch_size,
      epochs=epochs,
      validation_data=(x_val, y_val),
      callbacks = [callback])





y_pred = model.predict(x_test)
y_pred_class = np.argmax(y_pred,axis = 2)
pred = y_pred_class.ravel()
true = y_test_class.ravel()

pred = pred[true!= 2]
true = true[true!= 2]

print('general evaluation')
print('Acc: ',accuracy_score(true,pred))
print('F1: ' ,f1_score(true, pred, average= 'macro'))
print('Precision: ',precision_score(true,pred,average= 'macro'))
print('Recall: ',recall_score(true,pred,average= 'macro'))
print('Confusion_matrix: \n',confusion_matrix(true,pred))




new_f1 = {}
for change in [['pos'],['dep'],['emb'],['srl']]:
    x_test_tmp = x_test.copy()
    for i in range(len(x_test_tmp)):
        for j in range(len(x_test_tmp[i])):
            if 'pos' in change:
                pos = np.zeros(17)
                pos[np.random.randint(0,17)] = 1
                x_test_tmp[i][j][:17] = pos   
        
            if 'dep' in change:
                    dep = np.zeros(27)
                    dep[np.random.randint(0,27)] = 1
                    x_test_tmp[i][j][17 :17  + 27] = dep
            if 'emb' in change:
                random_token = random.choice(w2v_model.wv.index2entity)
                token_emb = w2v_model[random_token]
                x_test_tmp[i][j][17  + 27: 17  + 27 + 50] = token_emb
    
            if 'srl' in change:
                srl = np.zeros(20)
                srl[np.random.randint(0,20)] = 1
                x_test_tmp[i][j][17 + 27 + 50:] = srl
            
    y_pred = model.predict(x_test_tmp)
    y_pred_class = np.argmax(y_pred,axis = 2)
    pred = y_pred_class.ravel()
    true = y_test_class.ravel()
    
    pred = pred[true!= 2]
    true = true[true!= 2]

    f1 = f1_score(true, pred, average= 'macro')
    
    new_f1[change[0]] = f1

