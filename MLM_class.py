# -*- coding: utf-8 -*-
"""
Created on Mon Mar 16 16:54:46 2020

@author: babdeen
"""

# -*- coding: utf-8 -*-
"""
Created on Fri Mar  6 14:06:00 2020

@author: babdeen
"""

import torch
from pytorch_pretrained_bert import BertTokenizer,BertForMaskedLM,BertModel
from nltk.corpus import stopwords
stop_words = set(stopwords.words('english'))
import string
from nlp_general import NLP
from scipy.special import softmax 


class MLM:
    def __init__(self,model_path = 'bert-base-uncased',length = None, cased = False):
        self.length = length
        self.cased = cased
        if cased == True :
            model_path = 'bert-base-cased'
            self.tokenizer = BertTokenizer.from_pretrained('bert-base-cased',do_lower_case = False)
        else:
            self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.model = BertForMaskedLM.from_pretrained(model_path)
        self.base_model = BertModel.from_pretrained(model_path)
        self.model.eval()
        self.base_model.eval()
        self.vocab = dict(self.tokenizer.vocab)
        sw_vocab = stop_words.intersection(set(self.vocab.keys()))
        self.sw_indecies = self.tokenizer.convert_tokens_to_ids(list(sw_vocab))
        puncs = list(string.punctuation)
        self.puncs_indecies = self.tokenizer.convert_tokens_to_ids(puncs)
    def bert_tokenize(self,text):
        tokenized_text = self.tokenizer.tokenize(text)
        if self.length != None:
            tokenized_text =['[CLS]'] + tokenized_text[:self.length-2] + ['[SEP]'] + ['[PAD]' for i in range(self.length - len(tokenized_text)-2 )]
        else:
            tokenized_text =['[CLS]'] + tokenized_text + ['[SEP]']
        return tokenized_text
    
    def bert_tokens_to_ids(self,tokens,length = None):

        indexed_tokens = self.tokenizer.convert_tokens_to_ids(tokens)
        return indexed_tokens
    
    def mask_whole_word(self, tokens, mask_index):
        output = [i for i in tokens]
        output[mask_index] = '[MASK]'
        for i in range(mask_index + 1,len(tokens)):
            if output[i][:2] == '##':
                output[i] = '[MASK]'
            else:
                break
        return output
                
    def get_whole_word_from_bert_tokens(self,tokens,ind):
        ww = tokens[ind]
        for i in range(ind+1,len(tokens)):
            if tokens[i][:2] == '##':
                ww += tokens[i][2:]
            else:
                break
        return ww

    def get_every_token_prop(self, text , discard = set(), remove_sw = True, thr = -1):
        prop = []
        tokens = self.bert_tokenize(text) 
        if len(tokens) > 400:
            return []
        org_tokens_ids = self.bert_tokens_to_ids(tokens)   
        segments_ids = [0] * len(tokens)
        segments_tensors = torch.tensor([segments_ids])
        last_discard = False
        if remove_sw:
            discard.update(stop_words)
            discard.update(set(string.punctuation))

        for i in range(1,len(tokens)-1):
            if tokens[i][:2] == '##':
                if not last_discard:
                    prop[-1][0] += tokens[i][2:]
                continue
            if self.get_whole_word_from_bert_tokens(tokens , i) in discard:
                last_discard = True
                continue
            else:
                last_discard = False
            tokens_masked = self.mask_whole_word(tokens,i)
            for j in range(len(tokens_masked)-1,-1,-1):
                if tokens[i] == tokens_masked[j]:
                    tokens_masked = self.mask_whole_word(tokens_masked,j)
            tokens_ids = self.bert_tokens_to_ids(tokens_masked)   
            tokens_tensor = torch.tensor([tokens_ids])
            with torch.no_grad():
                predictions =  self.model(tokens_tensor, segments_tensors)
                preds = predictions.tolist()[0][i]
                #
                if remove_sw:
                    mini = min(preds)
                    for j in (self.sw_indecies + self.puncs_indecies):
                        preds[j] = mini
                #
                preds_softmaxed = softmax(preds)
                if thr == -1:
                    prop.append([tokens[i],preds_softmaxed[org_tokens_ids[i]]])
                else:
                    prop.append([tokens[i],preds_softmaxed[org_tokens_ids[i]], sum(i > thr for i in preds_softmaxed)  ])
        return prop
    
    def get_element_indicies(self,_list, element):
        out = []
        count = _list.count(element)
        last_occu = -1
        for c in range(count):
            last_occu = _list[last_occu+1:].index(element) + last_occu + 1
            out.append(last_occu)
        return out
    

    def get_mlm(self,text, filter_sw = True):
        tokens = self.bert_tokenize(text)
        mask_indicies = self.get_element_indicies(tokens,'[MASK]')
        
        segments_ids = [0] * len(tokens)
        segments_tensors = torch.tensor([segments_ids])
        
        tokens_ids = self.bert_tokens_to_ids(tokens)   
        tokens_tensor = torch.tensor([tokens_ids])
        with torch.no_grad():
            predictions =  self.model(tokens_tensor, segments_tensors)
        masks_pred = [] 
        for mask_ind in mask_indicies:
            preds = predictions.tolist()[0][mask_ind]
            #
            if filter_sw:
                mini = min(preds)
                for i in (self.sw_indecies + self.puncs_indecies):
                    preds[i] = mini
            #
            preds_softmaxed = softmax(preds)
            
            r =  [(self.tokenizer.convert_ids_to_tokens([i[0]])[0],i[1]) for i in enumerate(preds_softmaxed)]
            r.sort(key = lambda x: x[1], reverse = True)  
            masks_pred.append(r)
        return masks_pred
    
    def get_most_predictable(self,prop_dict,thr):
        high_pred = []
        for i in prop_dict:
            if prop_dict[i] > thr:
                high_pred.append(i)
        return high_pred
    
    def sentence_info(self,prop,word = ''):
        summ = 0
        for i in prop:
            if i[0] != word:
                summ += i[1]
        return summ/(len(prop) - (1 if word in prop else 0))
    def number_of_unpred(self,prop, thr = .01):
        return len([i for i in prop if i[1] > thr])
    
    def get_to_mask_words(self,text):
        ners = NLP.get_ner_list(text)
        ners_words = []
        for ne in ners:
            ners_words.extend([i.lower() for i in ne[0].split()])
        prop = self.get_every_token_prop(text, discard = set())
        pos = NLP.pos_tag(text)
        sents_pro= {'ners': ners, 'prop' : prop, 'pos': pos}
        to_mask = []
        v = sents_pro['ners']
        to_mask.extend([i for i in v])
        sorted_prop = sorted(prop, key = lambda x :x[1])
        to_mask.extend([i for i in sorted_prop])
        return to_mask
    
    def get_vector(self, text):
        tokens = self.bert_tokenize(text)
        ids = self.bert_tokens_to_ids(tokens)
        input_ids = torch.tensor(ids).unsqueeze(0) 
        outputs = self.base_model(input_ids)
        last_hidden_states = outputs[0][0][0]
        return last_hidden_states

    def get_token_vector(self, text):
        tokens = self.bert_tokenize(text)
        index = tokens.index('*')
        del tokens[index]
        ids = self.bert_tokens_to_ids(tokens)
        input_ids = torch.tensor(ids).unsqueeze(0) 
        outputs = self.base_model(input_ids)
        vector = outputs[0][0][0][index]
        return vector
    
    def get_avg_vector(self, text):
        tokens = self.bert_tokenize(text)
        ids = self.bert_tokens_to_ids(tokens)
        input_ids = torch.tensor(ids).unsqueeze(0) 
        outputs = self.base_model(input_ids)
        last_hidden_states = outputs[0][0][0]
        avg = last_hidden_states[1:last_hidden_states.size()[0]-1].sum(dim=0) / (last_hidden_states.size()[0] - 2)
        return avg
    
    def get_max_vector(self, text):
        tokens = self.bert_tokenize(text)
        ids = self.bert_tokens_to_ids(tokens)
        input_ids = torch.tensor(ids).unsqueeze(0) 
        outputs = self.base_model(input_ids)
        last_hidden_states = outputs[0][0][0]
        maxi = last_hidden_states[1:last_hidden_states.size()[0]-1].max(dim=0).values
        return maxi
    
    def cos_sim_torch(self,v1,v2):
        return (torch.dot(v1,v2) / (torch.norm(v1) * torch.norm(v2) )).item() 
    

    
    def extract_all_2(self,text):
        mem_wordnet = [i.lower() for i in NLP.rank_words(text)]
        mem_enron = [i.lower() for i in NLP.rank_words_from_counter_wn(text,'C:/basel/models/enron_counter.pkl')]
        releventness = [i[0] for i in self.w_s_similarity(text)]
        prop_thr = .01
        most_prop_thr = 7
        words_properties = {}
        text = ' '.join(text.split())
        ners = NLP.get_ner_list(text)
        all_ner_words = []
        for ne in ners:
            new_ners = [i.lower() for i in ne[0].split()]
            all_ner_words.extend(new_ners)
        prop = self.get_every_token_prop(text, discard = set(all_ner_words),thr = prop_thr)
        prop_score = [(i[0],(0 if i[1] < prop_thr else (1 if i[2] > most_prop_thr else 2 ))) for i in prop]
        prop_score =  dict(prop_score)
        for i in prop_score:
            if i in words_properties:
                continue
            words_properties[i] = {'sentence':1,'releventness': releventness.index(i) if i in releventness else  99, 'prop_score':prop_score[i] if i in prop_score else -1, 'mem_enron': mem_enron.index(i) if i in mem_enron and i not in all_ner_words else -1 , 'mem_wordnet': mem_wordnet.index(i) if i in mem_wordnet and i not in all_ner_words else -1 }
        for i in ners:
            words_properties[i[0]] = {'sentence':1,'releventness': -1, 'prop_score': -1, 'mem_enron': -1 , 'mem_wordnet': -1 }
        ranked = sorted(words_properties , key = lambda x: (words_properties[x]['prop_score'], words_properties[x]['mem_enron']) )
        return {'words_properties': words_properties,'ranked': ranked}


    def w_s_similarity(self,email):
        tokens = self.bert_tokenize(email)
        vectors = self.get_vector(email)
        org_embedding = torch.mean(torch.tensor([vectors[i].tolist() for i in range(1,len(tokens)-1) if tokens[i] not in stop_words]),axis = 0 )   
        sim = []
        for i in range(1,len(tokens) -1 ):
            if tokens[i][:2] == '##':
                sim[-1][0] += tokens[i][2:]
            else:
                sim.append([tokens[i], self.cos_sim_torch(org_embedding,vectors[i])])
    
        return sorted(sim, key = lambda x:x[1],reverse = True)

