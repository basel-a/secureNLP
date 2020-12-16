# -*- coding: utf-8 -*-
"""
Created on Tue Jul 21 23:21:08 2020

@author: babdeen
"""


# -*- coding: utf-8 -*-
"""
Created on Sun Jul  5 23:01:55 2020

@author: babdeen
"""

from allennlp.predictors.predictor import Predictor

# import allennlp_models.syntax.srl
SRL = Predictor.from_path("https://storage.googleapis.com/allennlp-public-models/bert-base-srl-2020.03.24.tar.gz")
DT = Predictor.from_path("https://storage.googleapis.com/allennlp-public-models/biaffine-dependency-parser-ptb-2020.04.06.tar.gz")
CT = Predictor.from_path("https://storage.googleapis.com/allennlp-public-models/elmo-constituency-parser-2020.02.10.tar.gz")
CoRef = Predictor.from_path("https://storage.googleapis.com/allennlp-public-models/coref-spanbert-large-2020.02.27.tar.gz")



class Parser():
    def extract_srl(text):
        srl = SRL.predict(text)
        Parser.add_v_id_srl(srl)
        srl_dict = Parser.srl_to_dict(srl)
        return srl_dict
    def srl_predict(text):
        srl = SRL.predict(text)
        Parser.add_v_id_srl(srl)
        return srl
    def extract_dt(text):
        return DT.predict(text)
    def extract_ct(text):
        return CT.predict(text)
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
      
    def get_coref(text):
        object_prounous = {'his','her','its','our','thier','my','your'}
        subject_prounous = {'he','she','it','we','they','I','you'}

        
        cor = CoRef.predict( document=text)
        clusters = cor["clusters"]
        words = cor["document"]
        words2 = [i for i in words]
        for cluster in clusters:
            original = " ".join(words[cluster[0][0]:cluster[0][-1]+1]) if cluster[0][0] != cluster[0][1] else words[cluster[0][0]]
            original = original.replace("\'s" , "")
            if original in subject_prounous or original in object_prounous :
                continue
            for i in cluster:
                if i[0] == i[1]:
                    if words2[i[0]] not in (list(object_prounous) + list(subject_prounous)):
                        continue
                    if words2[i[0]] in object_prounous:
                        words2[i[0]] = (original + '\'s') if words2[i[0]] != 'my' else 'my'
                    else:
                        words2[i[0]] = original
        result = " ".join(words2)
        return result
    