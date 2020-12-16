# -*- coding: utf-8 -*-
"""
Created on Sat Dec  5 16:47:38 2020

@author: babdeen
"""

import funs
import os
from parse_class import Parser
import tensorflow as tf
import numpy as np
class Data():
    def __init__(self, path):
        malware_data_path = path + 'MalwareTextDB-2.0/data/'
        self.data_tagged_dict = {}
        for folder in ['train','dev','test_1','test_2','test_3']:
            files = os.listdir(malware_data_path + folder +'/tokenized')
            self.data_tagged_dict[folder] = []
            for f in files:
                with open(malware_data_path + folder +'/tokenized/' + f, encoding = 'utf-8') as file:
                    data_tagged = file.read()
                    self.data_tagged_dict[folder].append(data_tagged)
        
        self.train_sentences = self.data_tagged_dict['train'] 
        self.val_sentences = self.data_tagged_dict['dev']
        self.test_sentences = self.data_tagged_dict['test_1'] + self.data_tagged_dict['test_2']+ self.data_tagged_dict['test_3']
        self.data_tagged_full = self.train_sentences + self.val_sentences + self.test_sentences
        
        self.sent_malicious_verbs = {}
        self.pos_sents = []
        self.neg_sents = []
        for data_tagged in self.data_tagged_full:
            data_tagged = data_tagged.strip()
            data_tagged = data_tagged.replace('\n\n', '\n \n')
            sents = data_tagged.split('\n \n')
            
            for sent in sents:
                tokens = sent.split('\n')
                sent_text = ' '.join([j.split()[0] for j in sent.split('\n')])
                for token in tokens:
                    if len(token.split()) != 2:
                        continue 
                    label = token.split()[1]
                    ent = token.split()[0]
                    if label[2:] == 'Action':
                        # sent_text = ''.join(sent_text.split())
                        if sent_text not in self.sent_malicious_verbs:
                            self.sent_malicious_verbs[sent_text] = [ent]
                        else:
                            self.sent_malicious_verbs[sent_text].append(ent)
                            
            pos = [i for i in sents if i.find('B-Entity') != -1]
            neg = [i for i in sents if i.find('B-Entity') == -1]
            self.pos_sents.extend([' '.join([j.split()[0] for j in i.split('\n')]) for i in pos])
            self.neg_sents.extend([' '.join([j.split()[0] for j in i.split('\n')]) for i in neg])
            
        self.train_sents_pos = funs.read_json_as_dict(path + 'pos_sents.json')
        self.train_sents_neg = funs.read_json_as_dict(path + 'neg_sents.json')
        self.train_sents_pos_srl = funs.read_json_as_dict(path + 'pos_sents_srl.json')
        self.train_sents_neg_srl = funs.read_json_as_dict(path + 'neg_sents_srl.json')
        
        self.val_sents_pos = funs.read_json_as_dict(path + 'pos_sentences_val.json')
        self.val_sents_neg = funs.read_json_as_dict(path + 'neg_sentences_val.json')
        self.val_sents_pos_srl = funs.read_json_as_dict(path + 'pos_sentences_val_srl.json')
        self.val_sents_neg_srl = funs.read_json_as_dict(path + 'neg_sentences_val_srl.json')
        
        self.test_sents_pos = funs.read_json_as_dict(path + 'pos_sentences_test.json')
        self.test_sents_neg = funs.read_json_as_dict(path + 'neg_sentences_test.json')
        self.test_sents_pos_srl = funs.read_json_as_dict(path + 'pos_sentences_test_srl.json')
        self.test_sents_neg_srl = funs.read_json_as_dict(path + 'neg_sentences_test_srl.json')

        for i in  self.train_sents_pos_srl:
            Parser.add_v_id_srl( self.train_sents_pos_srl[i])
        self.train_pos_sents_srl_dict = {i:Parser.srl_to_dict( self.train_sents_pos_srl[i]) for i in  self.train_sents_pos_srl}
        
        for i in  self.train_sents_neg_srl:
            Parser.add_v_id_srl( self.train_sents_neg_srl[i])
        self.train_neg_sents_srl_dict = {i:Parser.srl_to_dict( self.train_sents_neg_srl[i]) for i in  self.train_sents_neg_srl}
        
        for i in  self.val_sents_pos_srl:
            Parser.add_v_id_srl( self.val_sents_pos_srl[i])
        self.val_sents_pos_srl_dict = {i:Parser.srl_to_dict( self.val_sents_pos_srl[i]) for i in  self.val_sents_pos_srl}
        
        for i in  self.val_sents_neg_srl:
            Parser.add_v_id_srl( self.val_sents_neg_srl[i])
        self.val_sents_neg_srl_dict = {i:Parser.srl_to_dict( self.val_sents_neg_srl[i]) for i in  self.val_sents_neg_srl}
        
        for i in  self.test_sents_pos_srl:
            Parser.add_v_id_srl( self.test_sents_pos_srl[i])
        self.test_sents_pos_srl_dict = {i:Parser.srl_to_dict( self.test_sents_pos_srl[i]) for i in  self.test_sents_pos_srl}
        
        for i in self. test_sents_neg_srl:
            Parser.add_v_id_srl( self.test_sents_neg_srl[i])
        self.test_sents_neg_srl_dict = {i:Parser.srl_to_dict( self.test_sents_neg_srl[i]) for i in  self.test_sents_neg_srl}
        
        
        self.all_pos_sents_srl = {self.train_sents_pos[i]: self.train_pos_sents_srl_dict[i] for i in self.train_sents_pos if i in self.train_pos_sents_srl_dict}
        self.all_pos_sents_srl.update({self.val_sents_pos[i]: self.val_sents_pos_srl_dict[i] for i in self.val_sents_neg if i in self.val_sents_pos_srl_dict}) 
        self.all_pos_sents_srl.update({self.test_sents_pos[i]: self.test_sents_pos_srl_dict[i] for i in self.test_sents_pos if i in self.test_sents_pos_srl_dict}) 
        
        self.all_neg_sents_srl = {self.train_sents_neg[i]: self.train_neg_sents_srl_dict[i] for i in self.train_sents_neg if i in self.train_sents_neg_srl}
        self.all_neg_sents_srl.update({self.val_sents_neg[i]: self.val_sents_neg_srl_dict[i] for i in self.val_sents_neg if i in self.val_sents_neg_srl}) 
        self.all_neg_sents_srl.update({self.test_sents_neg[i]: self.test_sents_neg_srl_dict[i] for i in self.test_sents_neg if i in self.test_sents_neg_srl}) 

    def get_SVO_from_srl(self,srl,v, tags = ['ARG0','V','ARG1','ARG2','ARG3']):
        args = []
        srl_tags = []
        for t in tags:
            if t in srl[v]:
                args.append(srl[v][t]['text'])
                srl_tags.extend([t for i in range(len(srl[v][t]['text'].split()))])
                
        srl_sent = ' '.join(args)
        srl_sent = ' '.join(srl_sent.split())
        return (srl_sent,srl_tags)
    
    def get_OHV(self,labels, label_to_num):
        map_label = [None for i in labels]
        for i,l in enumerate(labels):
            map_label[i] = label_to_num[l]
        OHV = tf.keras.utils.to_categorical(map_label, num_classes=len(label_to_num))
        return OHV

    def get_word_emb(self, tokens,emb_model,max_len = None):
        we = []
        for token in (tokens):
            we.append(emb_model[token] if token in emb_model and token != 'pad' else (np.zeros(50) if  token == 'pad' else np.ones(50) ))
        if max_len != None:
            we = we[:max_len] + [np.zeros(50) for i in range(max_len - len(we))]
        return we


            
            
            