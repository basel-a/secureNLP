# -*- coding: utf-8 -*-
"""
Created on Wed Mar 18 12:15:21 2020

@author: babdeen
"""
import spacy
ner_spacy = spacy.load('en_core_web_lg',disable=['parser', 'tagger']) 
pos_tagger = spacy.load("en_core_web_lg",disable=['ner', 'parser'])
parser = spacy.load("en_core_web_lg",disable=['ner'])

from spacy.lang.en import English
nlp = English()
nlp.add_pipe(nlp.create_pipe('sentencizer')) 
from nltk.corpus import stopwords
stop_words = set(stopwords.words('english'))
import string
from nltk.corpus import wordnet as wn
import re
from collections import Counter
import pickle
from nltk.stem.wordnet import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()

class NLP:
    def srl_to_dict(srl):
        SRLDict = {}
        for verb in srl['verbs']:
            verb_str = verb['id']
            SRLDict[verb_str] = {}
            for ind,tag in enumerate(verb['tags']):
                if tag != 'O':
                    if tag[0] == 'B':
                        newTag = tag[tag.find('-')+1:]
                        if newTag not in SRLDict[verb_str]:
                            SRLDict[verb_str][newTag] = {'text': srl['words'][ind] }
                        else:
                            SRLDict[verb_str][newTag]['text'] += ('/ ' + srl['words'][ind]) 
                        if newTag == 'V':
                            SRLDict[verb_str][newTag]['index'] = ind
                    else :   
                        newTag = tag[tag.find('-')+1:]
                        if newTag not in SRLDict[verb_str]:
                            continue
                        SRLDict[verb_str][newTag]['text'] += (' ' + srl['words'][ind]) 
    
        return SRLDict

    def add_v_id_srl(srl):
        verbs = set()
        counter = {}
    
        for v in srl["verbs"]:
            if v["verb"] not in verbs:
                verbs.add(v["verb"])
                counter[v["verb"]] = 1
                v['id'] = v["verb"] 
            else:
                counter[v["verb"]] += 1
                v['id'] = v["verb"] + '_' + str(counter[v["verb"]])
                
    def get_lemma(word,is_verb):
        return lemmatizer.lemmatize(word.lower(), wn.VERB if is_verb else wn.NOUN)
    
    def get_ner_dict(sentence):
        doc = ner_spacy(sentence)
        ner_dict = {}
        for ent in doc.ents:
            for word in ent.text.split():
                ner_dict[word] = ent.label_   
        return ner_dict
    def get_ner_list(sentence):
        doc = ner_spacy(sentence)
        ner_list = []
        for ent in doc.ents:
            ner_list.append((str(ent), ent.label_))
        return ner_list
    
    def seperate_sentences(text):
        doc = nlp(text)
        sentences = [sent.string.strip() for sent in doc.sents]
        return sentences
    
    def get_stop_words():
        return stop_words
    
    def add_comment_to_words(text, comments , sep = '()'):
        tokens = text.split()
        new_text = ''
        for i in tokens:
            new_text += i
            word = i.translate(str.maketrans('', '', string.punctuation))
            if word in comments:
                new_text += ' ' + sep[0] + str(comments[word]) + sep[1]
            new_text+=' '
        return new_text
    def pos_tag(text): 
        pos = {}
        doc = pos_tagger(text)
        for token in doc:
            tag = token.pos_
            token = token.text
            if token not in pos:
                pos[token] = {'tag' : tag , 'text': token}
            else:
                i = 2
                new_token = token + '_' + str(i)
                while new_token in pos:
                    i+=1
                    new_token = token + '_' + str(i)
                pos[new_token] = {'tag' : tag , 'text': token}
        return pos
    ##
    def remove_text_inside_brackets(text, brackets="()"):
        count = [0] * (len(brackets) // 2) # count open/close brackets
        saved_chars = []
        for character in text:
            for i, b in enumerate(brackets):
                if character == b: # found bracket
                    kind, is_close = divmod(i, 2)
                    count[kind] += (-1)**is_close # `+1`: open, `-1`: close
                    if count[kind] < 0: # unbalanced bracket
                        count[kind] = 0  # keep it
                    else:  # found bracket to remove
                        break
            else: # character is not a [balanced] bracket
                if not any(count): # outside brackets
                    saved_chars.append(character)
        return ''.join(saved_chars)
    def insert_between_elements(_list,element):
         return [(_list[i//2] if i % 2 == 0 else element) for i in range(len(_list))]
     
    def check_if_meaningful_sentence(sentence,min_tokens = 2):
        tokens = set(sentence.translate(str.maketrans('', '', string.punctuation)).lower().split())
        return len(tokens.intersection(stop_words)) != 0 and len(tokens) > min_tokens

    def filter_non_meaningful_sentences(sentences,keep_empty = False, min_tokens =2):
        meaningful_sents = []
        for sent in sentences:
            if keep_empty and sent == '':
                meaningful_sents.append(sent)
                continue
            if NLP.check_if_meaningful_sentence(sent,min_tokens):
                meaningful_sents.append(sent)
        return meaningful_sents
    
    def get_count(word):
        c = 0
        syns = wn.synsets(word)
        if len(syns) == 0:
            return -1
        for syn in syns:
            for lem in syn.lemmas():
                c+= lem.count()
        return c
    
    def rank_words(sentence):
        sentence = sentence.translate(str.maketrans('','',string.punctuation))
        w_count = {}
        for word in set(sentence.split()):
            count = NLP.get_count(word)
            if count != -1:
                w_count[word] = count
#        w_count = list(zip(w_count.keys(), w_count.values())) 
        return sorted(w_count, key = lambda x:w_count[x])

    def clean_html(raw_html):
      cleanr = re.compile('<.*?>')
      cleantext = re.sub(cleanr, ' ', raw_html)
      return cleantext
  
    def rank_words_from_counter(text,src):
        with open(src,'rb') as file:
            counter = pickle.load(file)
        text = text.translate(str.maketrans('','',string.punctuation))
        w_count = {}
        for word in text.split():
            if word in counter:
                count = counter[word]
                w_count[word] = count
#        w_count = list(zip(w_count.keys(), w_count.values())) 
        return sorted(w_count, key = lambda x:w_count[x])
    
    def get_synonym(word,tag = 'all'):
        return [i for i in wn.synsets(word) if (tag == 'all' or i.name().split('.')[1] == tag)]
    def get_synonym2(word,tag , threshold = 10):
        if tag not in ('VERB','NOUN','ADJ','ADV'):
            return []
        wn_tag = {'VERB': 'v','NOUN':'n','ADJ':'a','ADV':'r'}
        synms_count =  {}
        for v in wn.synsets(word , pos = wn_tag[tag]):
            synms_count[v.name()] = 0
            for lemma in v.lemmas():
                synms_count[v.name()] += lemma.count()
            
        count_all= sum([synms_count[key] for key in synms_count.keys()])
        if count_all == 0:
            return []
        synms_percent = {i:int(synms_count[i]/count_all * 100) for i in synms_count}
        return [i for i in synms_percent if synms_percent[i] > threshold]
    
    
    def rank_words_from_counter_wn(text,src, f = {}):
        with open(src,'rb') as file:
            counter = pickle.load(file)
            
        f = {'VERB':1.1023854760396876,
        'NOUN': 1.0,
        'ADJ': 3.753683958308374,
        'ADV': 4.564685314685315}

        pos = NLP.pos_tag(text)
        text = text.translate(str.maketrans('','',string.punctuation))
        w_count = {}
        for word in text.split():
            if word not in pos:
                continue
            if word in counter:
                count = counter[word]
                w_count[word] = count
                for sys in set([i[:i.find('.')] for i in NLP.get_synonym2(word,pos[word]['tag'])]):
                    if sys in counter and sys != word:
                        w_count[word] += counter[sys]
        if f != {}:
            for word in w_count:
                if pos[word]['tag'] in f:
                    w_count[word] *= f[pos[word]['tag']]
        return sorted(w_count, key = lambda x:w_count[x])
    def spacy_lemma(text):
        doc = pos_tagger(text)
        # Extract the lemma for each token and join
        return " ".join([(token.lemma_ if token.pos_ == 'NOUN' or token.pos_=='VERB' else token.text) for token in doc])
        
    def extract_pos(text, pos):
        doc = pos_tagger(text)
        # Extract the lemma for each token and join
        return " ".join([(token.lemma_ if token.pos_ == 'NOUN' or token.pos_=='VERB' else token.text) for token in doc if token.pos_ == pos])
       
    def clean_punc(text):
        return text.translate(str.maketrans('', '', string.punctuation))
    def clean_edges_punc(text):      
        return re.findall('^['+string.punctuation+']*(.*?)['+string.punctuation+']*$',text)[0]

    
    def space_pad(text):
        return text.translate(str.maketrans({key: " {0} ".format(key) for key in string.punctuation}))
            
    def dict_To_list(d):
        return [d[i] for i in d]

    def dict_of_dicts_to_dict(dd):
        out = {}
        for i in dd:
            for j in dd[i]:
                out[j] = dd[i][j]
        return out

    def get_spacy_parse(text):
        doc = parser(text)
        out = []

        for token in doc:
            out.append([token.text,token.lemma_,token.pos_, token.tag_, token.dep_])
        parse = list(zip(*out))
        words,lemma,pos,tag,dep = list(parse[0]), list(parse[1]), list(parse[2]), list(parse[3]), list(parse[4])

        return {'words':words,'lemma':lemma,'pos':pos,'tag':tag,'dep':dep}
    
