import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
import torch.nn.utils as utils
import matplotlib as mpl
import matplotlib.pyplot as plt
from random import sample
from event_tensors.glove_utils import Glove
import itertools
import torch.nn.functional as F
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from models_func import BiLSTM_CRF,getsentence,form_batch,prepare_sequence,transfrom_sentence_to_embed,acc,get_label_stat
import pickle
torch.manual_seed(1)
if __name__== '__main__':
    glove = Glove('glove.6B.100d.ext.txt')
    data = pd.read_csv("GMB_dataset.txt", sep="\t", header=None, encoding="latin1")
    data.columns = data.iloc[0]
    START_TAG = "<START>"
    STOP_TAG = "<STOP>"
    data = data[1:]
    data.columns = ['Index','Sentence #','Word','POS','Tag']
    data = data.reset_index(drop=True)
    data[~data['Tag'].isin(['O'])]
    sample_num=1
    acc_step=8
    getter = getsentence(data)
    sentences = getter.sentences
    EMBEDDING_DIM = 150
    HIDDEN_DIM = 256
    loss=[]
    # Make up some training data
    word_to_ix = {}
    for long_sentence in sentences:
        for word in long_sentence:
            if word[0] not in word_to_ix:
                word_to_ix[word[0]] = len(word_to_ix)
    syntax_to_ix = {}
    for long_sentence in sentences:
        for word in long_sentence:
            if word[1] not in syntax_to_ix:
                syntax_to_ix[word[1]] = len(syntax_to_ix)
    train_data=pickle.load(open('train_data','rb'))
    val_data=pickle.load(open('val_data','rb'))
    test_data=pickle.load(open('test_data','rb'))
    syntax_embeds = nn.Embedding(len(syntax_to_ix),50)
    tag_to_ix = {'B-geo':0,'B-gpe':1,'B-tim':2,'B-org':3,'I-geo':4,'B-per':5,'I-per':6,'I-org':7,'B-nat':8,'I-tim':9,'I-gpe':10,'I-nat':11,'B-art':12,'I-art':13,'B-eve':14,'I-eve':15,'O':16,START_TAG: 17, STOP_TAG: 18}
    model = BiLSTM_CRF(len(word_to_ix), tag_to_ix, EMBEDDING_DIM, HIDDEN_DIM).cuda(0)
    optimizer = optim.Adam(itertools.chain(model.parameters(),syntax_embeds.parameters()), lr=0.01, weight_decay=1e-4)
    for epoch in range(10):  
        print(epoch)
        batch_index=0
        for time in range(len(train_data)):
            batched_data,batch_index=form_batch(train_data,sample_num,batch_index)
            targets = prepare_sequence(batched_data,tag_to_ix,'tag')
            batch_word_embedding,sentence_length=transfrom_sentence_to_embed(batched_data,syntax_to_ix,glove,syntax_embeds,targets)
            loss = model.neg_log_likelihood(batch_word_embedding, targets)/acc_step
            loss.backward()
            if time%acc_step==0:
                optimizer.step()
                optimizer.zero_grad()
            if time % 100==0:
                val_right_num=0
                val_total_num=0
                test_right_num=0
                test_total_num=0
                tag_to_ix_val={'B-geo':0,'B-gpe':0,'B-tim':0,'B-org':0,'I-geo':0,'B-per':0,'I-per':0,'I-org':0,'B-nat':0,'I-tim':0,'I-gpe':0,'I-nat':0,'B-art':0,'I-art':0,'B-eve':0,'I-eve':0,'O':0}
                tag_to_ix_test={'B-geo':0,'B-gpe':0,'B-tim':0,'B-org':0,'I-geo':0,'B-per':0,'I-per':0,'I-org':0,'B-nat':0,'I-tim':0,'I-gpe':0,'I-nat':0,'B-art':0,'I-art':0,'B-eve':0,'I-eve':0,'O':0}
                val_stat=get_label_stat(val_data)
                test_stat=get_label_stat(test_data)
                val_predict_list=[]
                val_targets_list=[]
                test_predict_list=[]
                test_targets_list=[]
                with torch.no_grad():
                    batch_index=0
                    for time_val in range(len(val_data)):
                        batched_data,batch_index=form_batch(val_data,sample_num,batch_index)
                        batch_word_embedding,sentence_length=transfrom_sentence_to_embed(batched_data,syntax_to_ix,glove,syntax_embeds,targets=None)
                        targets = prepare_sequence(batched_data,tag_to_ix,'tag')
                        predict_output=model(batch_word_embedding)
                        val_predict_list.extend(predict_output[1])
                        val_targets_list.extend(targets[0].tolist())
                        right_predicit_num,total_num,tag_to_ix_val=acc(predict_output,targets,tag_to_ix_val)
                        val_right_num=val_right_num+right_predicit_num
                        val_total_num=val_total_num+total_num
                with torch.no_grad():
                    batch_index=0
                    for time_test in range(len(test_data)):
                        batched_data,batch_index=form_batch(test_data,sample_num,batch_index)
                        batch_word_embedding,sentence_length=transfrom_sentence_to_embed(batched_data,syntax_to_ix,glove,syntax_embeds,targets=None)
                        targets = prepare_sequence(batched_data,tag_to_ix,'tag')
                        predict_output=model(batch_word_embedding)
                        test_predict_list.extend(predict_output[1])
                        test_targets_list.extend(targets[0].tolist())
                        right_predicit_num,total_num,tag_to_ix_test=acc(predict_output,targets,tag_to_ix_test)
                        test_right_num=test_right_num+right_predicit_num
                        test_total_num=test_total_num+total_num
                dict_key=list(tag_to_ix_val.keys())
                print('total right percentage in val set',int(val_right_num)/int(val_total_num))
                print('precisison macro',precision_score(val_targets_list,val_predict_list,average='macro'))
                print('recal macro',recall_score(val_targets_list,val_predict_list,average='macro'))
                print('f1',f1_score(val_targets_list,val_predict_list,average='macro'))
                for dkey in dict_key:
                    try:
                        print(dkey,'percentage',tag_to_ix_val[dkey]/val_stat[dkey])
                    except:
                        print(dkey,'in stat is 0')
                print('total right percentage in test set',int(test_right_num)/int(test_total_num))
                print('precisison macro',precision_score(test_targets_list,test_predict_list,average='macro'))
                print('recal macro',recall_score(test_targets_list,test_predict_list,average='macro'))
                print('f1',f1_score(test_targets_list,test_predict_list,average='macro'))
                for dkey in dict_key:
                    try:
                        print(dkey,'percentage',tag_to_ix_test[dkey]/test_stat[dkey])
                    except:
                        print(dkey,'in stat is 0')
    torch.save(model,'bilistm.pkl')
    torch.save(syntax_embeds,'syntax_embeds.pkl')
                        

    