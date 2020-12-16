# -*- coding: utf-8 -*-
"""
Created on Sat Dec  5 18:36:33 2020

@author: babdeen
"""

import tokenizations
from nlp_general import NLP
import numpy as np 
def del_arg(tags,ind):
    if tags[ind] == 'O':
        print('erorr')
        return 
    for j in range(ind+1,len(tags)):
        if tags[j] == 'O' or (j != ind and tags[j][0] == 'B'):
            break
        else:
            tags[j] = 'O'
    for j in range(ind,-1,-1):
        if tags[j] == 'O' or tags[j][0] == 'B':
            if tags[j][0] == 'B':
                tags[j] = 'O'
            break
        else:
            tags[j] = 'O'


def merge_srls(srl):
    new_srl =  ['O' for i in range(len(srl['words']))]
    for v in srl['verbs']:
        for i in range(len(v['tags'])):
            t = v['tags'][i]
            if t[2:] not in ['V','ARG0','ARG1','ARG2','ARG3']:
                continue 
            if t != 'O' and new_srl[i] !=  'O':
                del_arg(new_srl,i)
            if t != 'O':
                new_srl[i] = t
    return new_srl




def align_tokens(old_tokens,new_tokens,old_labels):
    a2b, b2a = tokenizations.get_alignments(old_tokens, new_tokens)
    l2 =  [None] * len(new_tokens)
    for i in range(len(a2b)):
        l = old_labels[i]
        for j in a2b[i]:
            l2[j] = l
    return l2

def parse_senteces(sentences):
    sentences_parsed = {}
    c= 0 
    for s in sentences:
        sent = ' '.join(sentences[s])
        sentences_parsed[s] = NLP.get_spacy_parse(sent)
        c+=1
        print(c)
    return sentences_parsed
        
def get_features(sentence_parse,OHV, parsers,is_pad = False):
       features = []
       for p in parsers:
           f = p + ('_pad' if is_pad == True else '')
           features.append([OHV[f][i] if i in OHV[f] else [0 for i in range(OHV[f]['len_'])] for i in sentence_parse[p]])
       x_all = np.concatenate(features,axis = 1)
       
       return x_all 
   
def extract_senteces_features(sentences_parsed,w2v_model,OHV):
    sentence_features_w_emb = {}
    sentence_features_parsing = {} #parsing features
    sentence_features = {}
    for sent in sentences_parsed:
        sentence_features_parsing[sent] = get_features(sentences_parsed[sent],OHV,['pos','dep'],True)
        sentence_features_w_emb[sent] = []
        for i,word in enumerate(sentences_parsed[sent]['words']):
            sentence_features_w_emb[sent].append(w2v_model[word] if word in w2v_model and word != 'pad' else (np.zeros(50) if word == 'pad' else np.ones(50)))
        sentence_features[sent] = np.concatenate((sentence_features_parsing[sent], np.array(sentence_features_w_emb[sent])),axis = 1)
    return sentence_features



def get_pos_sents(data_tagged_full):
    pos_sentences = []
    senteces_labels = {}
    malicious_verbs = {}
    for data_tagged in data_tagged_full:
        data_tagged = data_tagged.strip()
        data_tagged = data_tagged.replace('\n\n', '\n \n')
        sents = data_tagged.split('\n \n')
        
        for sent in sents:
            tokens = sent.split('\n')
            labels = [i.split()[1] for i in tokens if len(i.split()) == 2]
            words =  [i.split()[0] for i in tokens if len(i.split()) == 2]
            
            sent_text = ' '.join(words)
            if 'B-Action' in labels:
                pos_sentences.append(sent_text)
                senteces_labels[sent_text] = labels
            
                for i,label in enumerate(labels):
                    if label[2:] == 'Action':
                        if sent_text not in malicious_verbs:
                            malicious_verbs[sent_text] = [words[i]]
                        else:
                            malicious_verbs[sent_text].append(words[i])
                       
    return (pos_sentences,senteces_labels,malicious_verbs)

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

def clean_list(l, keep, fill):
    return [i if i in keep else fill  for i in l]

class DisjSet: 
    def __init__(self, n): 
        # Constructor to create and 
        # initialize sets of n items 
        self.rank = [1] * n 
        self.parent = [i for i in range(n)] 
  
  
    # Finds set of given item x 
    def find(self, x): 
          
        # Finds the representative of the set 
        # that x is an element of 
        if (self.parent[x] != x): 
              
            # if x is not the parent of itself 
            # Then x is not the representative of 
            # its set, 
            self.parent[x] = self.find(self.parent[x]) 
              
            # so we recursively call Find on its parent 
            # and move i's node directly under the 
            # representative of this set 
  
        return self.parent[x] 
  
  
    # Do union of two sets represented 
    # by x and y. 
    def Union(self, x, y): 
          
        # Find current sets of x and y 
        xset = self.find(x) 
        yset = self.find(y) 
  
        # If they are already in same set 
        if xset == yset: 
            return
  
        # Put smaller ranked item under 
        # bigger ranked item if ranks are 
        # different 
        if self.rank[xset] < self.rank[yset]: 
            self.parent[xset] = yset 
  
        elif self.rank[xset] > self.rank[yset]: 
            self.parent[yset] = xset 
  
        # If ranks are same, then move y under 
        # x (doesn't matter which one goes where) 
        # and increment rank of x's tree 
        else: 
            self.parent[yset] = xset 
            self.rank[xset] = self.rank[xset] + 1
            
def compare_oi(oi1, oi2):
    added = {}
    for v in oi2:
        if v not in oi1:
            added[v] = 1
        else:
            added[v] = {}
            for arg in oi2[v]:
                if arg not in oi1[v]:
                    added[v][arg] = 1
                elif arg != 'O':
                    if oi1[v][arg] != oi2[v][arg]:
                        added[v][arg] = oi2[v][arg]
                else:
                    added[v]['O'] = []
                    for o in oi2[v][arg]:
                        if o not in oi1[v][arg]:
                             added[v]['O'].append(o)
    return added
                


def srl_to_oi(srl):
    oi = {}
    for v in srl:
        oi[v] = {}
        for arg in srl[v]:
            if arg == 'ARG0':
                oi[v]['S'] = srl[v][arg]['text']
            elif arg == 'V':
                oi[v]['A'] = srl[v][arg]['text']
            elif arg in ['ARG1','ARG2','ARG3','ARG4']:
                if 'O' in oi[v]:
                    oi[v]['O'].append(srl[v][arg]['text'])
                else:
                    oi[v]['O'] = [srl[v][arg]['text']]
    return oi 

        
def get_jaccard_sim(str1, str2): 
    a = set(str1.split()) 
    b = set(str2.split())
    c = a.intersection(b)
    return float(len(c)) / (len(a) + len(b) - len(c))