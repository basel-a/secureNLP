# -*- coding: utf-8 -*-
"""
Created on Sat Nov 14 17:34:05 2020

@author: babdeen
"""


import utils
import os 
import networkx as nx
import json

from parse_class import Parser

files = os.listdir('./data/MalwareTextDB-2.0/data/train/tokenized')
    
with open('./data/MalwareTextDB-2.0/data/train/tokenized/Anthem_hack_all_roads_lead_to_China.tokens') as file:
    token_tagged = file.read()

with open('./data/MalwareTextDB-2.0/data/train/annotations/Anthem_hack_all_roads_lead_to_China.ann', encoding = 'utf-8') as file:
    data_relations = file.read().strip()
    
with open('./data/MalwareTextDB-2.0/data/train/annotations/AdversaryIntelligenceReport_DeepPanda_0 (1).txt', encoding = 'utf-8') as file:
    txt = file.read()

sents_SAO_extraction_all = {}
sent_id_all = {}
id_sent_all = {}

for f in files:
    with open('./data/MalwareTextDB-2.0/data/train/tokenized/' + f, encoding = 'utf-8') as file:
        token_tagged = file.read()
    
    with open('./data/MalwareTextDB-2.0/data/train/annotations/'+ f[:-7] +'.ann', encoding = 'utf-8') as file:
        data_relations = file.read().strip()
        
    with open('./data/MalwareTextDB-2.0/data/train/annotations/'+f[:-7]+'.txt', encoding = 'utf-8') as file:
        txt = file.read()

    
    relations = {}
    T_dict = {}
    all_T = set()
    for record in data_relations.split('\n'):
        record = record.strip()
        record_split = record.split('\t')
        record_type = record_split[0]
    
        if record_type[0] == 'T': #entity
            tag = record_split[1].split()[0]
            start = record_split[1].split()[1]
            end = record_split[1].split()[-1]
            T_dict[record_type] = {'tag': tag, 'start': start, 'end': end,'text': record_split[-1]}
            
        if record_type[0] == 'R': #relations
            r1 = record_split[1].split()[-2]
            r2 = record_split[1].split()[-1]
            r1_tag,r1_T = r1.split(':')
            r2_tag,r2_T = r2.split(':')
    
            all_T.add(r1_T)
            all_T.add(r2_T)
    
            if r1_T in relations:
                relations[r1_T].append((r1_tag,r2_T,r2_tag))
            else:
                relations[r1_T] = [(r1_tag,r2_T,r2_tag)]
            
            if r2_T in relations:
                relations[r2_T].append((r2_tag,r1_T,r1_tag))
            else:
                relations[r2_T] = [(r2_tag,r1_T,r1_tag)]
                
    T_id = {i:j for j,i in enumerate (all_T)}
    id_T = {j:i for j,i in enumerate (all_T)}
    
    
    edges = []
    for record in data_relations.split('\n'):
        record_split = record.split('\t')
        label = record_split[0]
        
        if label[0] == 'R':
            r1 = record_split[1].split()[-2]
            r2 = record_split[1].split()[-1]
            r1_rel,r1_tag = r1.split(':')
            r2_rel,r2_tag = r2.split(':')
            if r1_rel != 'Referer' and r2_rel != 'Referer':
                # edges.append((tech_id[r1_tag],tech_id[r2_tag]))
                edges.append((T_id[r1_tag],T_id[r2_tag]))
    
    
    obj = utils.DisjSet(len(all_T)) 
    for i in edges:
        obj.Union(i[0], i[1]) 
    
    
    groups = {}
    groups_id = {}
    for i in range(len(all_T)):
        group = obj.find(i)
        if group in groups:
            groups[group].add(id_T[i])
            groups_id[group].add(i)
        else:
            groups[group] = set([id_T[i]])
            groups_id[group] = set([i])
            
            
    group_sentences = {}
    for g in groups:
        T = list(groups[g])[0]
        x = int(T_dict[T]['start'])
        sent_st = txt[:x].rfind('. ') if txt[:x].rfind('. ') > txt[:x].rfind('.\n') else txt[:x].rfind('.\n') 
        sent_en = (txt[x:].find('. ') + x) if txt[x:].find('. ') < txt[x:].find('.\n') else (txt[x:].find('.\n')+x)
        group_sentences[g] = txt[sent_st+1:sent_en] 
    
    
    
    G = nx.Graph()
    G.add_edges_from(edges)
    
    SAO_extraction = {}
    for g in groups_id:
        G_sub =  G.subgraph(groups_id[g])
        G_sub_nodes = list(G_sub.nodes())
        S = [i for i in G_sub_nodes if T_dict[id_T[i]]['tag'] == 'Subject']
        O = [i for i in G_sub_nodes if T_dict[id_T[i]]['tag'] == 'Object']
        A = [i for i in G_sub_nodes if T_dict[id_T[i]]['tag'] == 'Action']
        SAO_extraction[g] = {}
        for action in A:
            
            action_text = T_dict[id_T[action]]['text']
            SAO_extraction[g][action_text] =  {'A':T_dict[id_T[action]]['text']}
            for s in S:
                SA = list(nx.all_simple_paths(G, source=s, target=action, cutoff = 1))                
                if len(SA) == 1:
                    SAO_extraction[g][action_text]['S'] = T_dict[id_T[SA[0][0]]]['text']
            for o in O:
                AO = list(nx.all_simple_paths(G, source=action, target=o))  
                for oa in AO:
                    if T_dict[id_T[oa[1]]]['tag'] != 'Subject':
                        path = [T_dict[id_T[i]]['tag'] for i in list(nx.all_simple_paths(G, source=action, target=o))[0] ]
                        if 'Action' in path[1:]:
                            continue
                        if 'O' in SAO_extraction[g][action_text]:
                            SAO_extraction[g][action_text]['O'].append(' '.join([T_dict[id_T[j]]['text'] for j in oa[1:]]))
                        else:
                            SAO_extraction[g][action_text]['O'] = [' '.join([T_dict[id_T[j]]['text'] for j in oa[1:]])]
    
    
    
    sentence_to_g = {}
    for g in groups_id:
        sent = group_sentences[g]
        if sent in sentence_to_g:
            sentence_to_g[sent].append(g)
        else:
            sentence_to_g[sent] = [g]
     


    sent_id = {s:i for i,s in enumerate(sentence_to_g)}
    id_sent = {i:s for i,s in enumerate(sentence_to_g)}
    
    
    sents_SAO_extraction = {}
    for sent in sentence_to_g:
        sents_SAO_extraction[sent_id[sent]] = []
        for g in sentence_to_g[sent] :
            if SAO_extraction[g] not in sents_SAO_extraction[sent_id[sent]]:
                sents_SAO_extraction[sent_id[sent]].append(SAO_extraction[g])
            

    sents_SAO_extraction_all[f] = sents_SAO_extraction
    sent_id_all[f] = sent_id
    id_sent_all[f] = id_sent


all_extractions ={}
for f in sents_SAO_extraction_all:
    for sent_id in sents_SAO_extraction_all[f]:
        all_extractions[id_sent_all[f][sent_id]] = sents_SAO_extraction_all[f][sent_id]
 

# all_sentences_srl = {}
# error = []
# c = 0
# for sent in all_extractions:
#     if sent in all_sentences_srl:
#         continue  
#     cleanr = re.compile('<.*>')
#     cleantext = re.sub(cleanr, ' ', sent.replace('\n',' '))
#     clean_sent = ' '.join(cleantext.split())
#     try:
#         srl = Parser.extract_srl(clean_sent)
#         all_sentences_srl[sent] = srl
#     except:
#         error.append(sent)
#     print(c)
#     c+=1

with open('./all_sentences_srl.json','w') as outfile:
    json.dump(all_sentences_srl, outfile)


#####SRL

from MLM_class import MLM
mlm  = MLM()

undetected_verbs = []
sims = [] 
for sent in all_extractions:
    oi = all_extractions[sent]
    oi_dict = {k: v for d in oi for k, v in d.items()}
    
    srl = all_sentences_srl[sent]

    oi_srl = utils.srl_to_oi(srl)
    oi_srl_2 = {i:oi_srl[i] for i in oi_srl if i in oi_dict}
    

    for v in oi_dict:
        srl_v = v.split()[-1]
        if srl_v not in oi_srl_2:
            undetected_verbs.append((v,sent))
        else:
            if 'A' not in oi_srl_2[srl_v]:
                continue 
            s1 = (oi_srl_2[srl_v]['S'] if 'S' in oi_srl_2[srl_v] else '') +' '+ (oi_srl_2[srl_v]['A']) + ' ' + (' '.join([i for i in oi_srl_2[srl_v]['O']]) if 'O' in oi_srl_2[srl_v] else '' )
            s2 = (oi_dict[v]['S'] if 'S' in oi_dict[v] else '') +' '+ (oi_dict[v]['A']) + ' ' + (' '.join([i for i in oi_dict[v]['O']]) if 'O' in oi_dict[v] else '' )
            sims.append((utils.get_jaccard_sim(s1,s2),v,sent))


with open('./algo_results1.txt', 'w', encoding='utf-8') as f:
    for item in sims:
        f.write(str(item))
            
score = sum([i[0] for i in sims])/len(sims)



######
mlm  = MLM()
cyber_mlm = MLM('./models/cyber_bert_1')


def sent_aug(sent):
    sent = sent.strip()[:-1] + ' .'  
    words = sent.split()
    org_words = words.copy()
    words_to_mask = []
    for i in range(len(words)):
        if len(mlm.bert_tokenize(words[i])) > 4:
            words_to_mask.append(words[i])
    
    changed_inds = []
    for i in range(len(words)):
        if words[i] in words_to_mask:
             words[i] = '[MASK]'
             changed_inds.append(i)
             
    mlms = mlm.get_mlm(' '.join(words))
    changes = {}
    for i,ind in enumerate(changed_inds):
        cands =  mlms[i]
        for i in cands:
            if i[0] not in words:
                words[ind] = i[0]
                break
        if words[ind] not in changes:
            changes[words[ind]] = org_words[ind]
            
    new_sent = ' '.join(words)
    return (new_sent, changes)




def aug_srl(srl,changes):
    new_srl = {}
    for v in oi_srl_2:
        new_srl[v] = {}
        if 'S' in oi_srl_2[v]:
            S_tokens = oi_srl_2[v]['S'].split()
            for i in range(len(S_tokens)):
                if S_tokens[i] in changes:
                    S_tokens[i] = changes[S_tokens[i]]
            new_srl[v]['S'] = ' '.join(S_tokens)
        
        if 'A' in oi_srl_2[v]:
            A_tokens = oi_srl_2[v]['A'].split()
            for i in range(len(A_tokens)):
                if A_tokens[i] in changes:
                    A_tokens[i] = changes[A_tokens[i]]
                
            new_srl[v]['A'] = ' '.join(A_tokens)
        
        if 'O' in oi_srl_2[v]:
            new_os = []
            for O in oi_srl_2[v]['O']:
                O_tokens = O.split()
                for i in range(len(O_tokens)):
                    if O_tokens[i] in changes:
                        O_tokens[i] = changes[O_tokens[i]]
                new_os.append(' '.join(O_tokens))
            new_srl[v]['O'] = new_os
    return new_srl


new_sent, changes = sent_aug(text)

srl = Parser.extract_srl(new_sent)
oi_srl = utils.srl_to_oi(srl)
oi_srl_2 = {i:oi_srl[i] for i in oi_srl if i in oi_dict}

aug_srl(oi_srl_2,changes)

clean_sent = 'During installation, the sample attempts to use documented APIs such as java and java to initialize itself as a persistent Windows service'
normal = mlm.get_vector(clean_sent).detach().numpy()
cyber = cyber_mlm.get_vector(clean_sent).detach().numpy()
tokens = mlm.bert_tokenize(clean_sent)



