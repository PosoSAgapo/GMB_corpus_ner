import pandas as pd
import numpy as np
#Data visualisation
import matplotlib.pyplot as plt
import seaborn as sns
#Modeling
from sklearn.model_selection import cross_val_predict, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn_crfsuite import CRF, scorers, metrics
from sklearn_crfsuite.metrics import flat_classification_report
from sklearn.metrics import classification_report, make_scorer
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import recall_score

import scipy.stats
import eli5
from sklearn.model_selection import train_test_split
from sklearn.externals import joblib
import torch
from models_func import BiLSTM_CRF,getsentence,word2features,sent2features,sent2labels,form_batch,transfrom_sentence_to_embed,prepare_sequence,get_label_stat,acc,form_standard_list,word_feature_map
from random import sample
from event_tensors.glove_utils import Glove
from collections import Counter
import pickle
def num_tag(lable_list,ix_to_tag):
    for num in range(len(lable_list)):
        lable_list[num]=ix_to_tag[lable_list[num]]
    return lable_list
model=torch.load('attn_bilstm_crf.pkl')
syntax_embeds=torch.load('syntax_embeds')
data = pd.read_csv("GMB_dataset.txt", sep="\t", header=None, encoding="latin1")
data.columns = data.iloc[0]
data = data[1:]
data.columns = ['Index','Sentence #','Word','POS','Tag']
data = data.reset_index(drop=True)
crf_2=joblib.load('crf1.pickle')
crf_1=joblib.load('crf2.pickle')
xgb=joblib.load('xgb')
rf=joblib.load('rf')
labels=['B-geo', 'B-gpe', 'B-tim', 'B-org', 'I-geo', 'B-per', 'I-per', 'I-org', 'B-nat', 'I-tim', 'I-gpe', 'I-nat', 'B-art', 'I-art', 'B-eve', 'I-eve','O']
getter = getsentence(data)
sentences = getter.sentences
test_data=pickle.load(open('test_data','rb'))
word_list=[]
lable_list=[]
for sentence in test_data:
    for word in sentence:
        word_list.append(word[0])
        lable_list.append(word[2])
glove = Glove('glove.6B.100d.ext.txt')
X = [sent2features(s) for s in test_data]
y = [sent2labels(s) for s in test_data]
words = [word_feature_map(word) for word in word_list]
sorted_labels = sorted(
    labels,
    key=lambda name: (name[1:], name[0]))
y_crf1_pred = crf_1.predict(X)
y_crf2_pred = crf_2.predict(X)
y_xgb_pred=xgb.predict(np.array(words))
y_rf_pred=rf.predict(np.array(words))
y_xgb_pred_standard=form_standard_list(y_xgb_pred,test_data)
y_rf_pred_standard=form_standard_list(y_rf_pred,test_data)
sample_num=1
syntax_to_ix = {}
START_TAG = "<START>"
STOP_TAG = "<STOP>"
tag_to_ix = {'B-geo':0,'B-gpe':1,'B-tim':2,'B-org':3,'I-geo':4,'B-per':5,'I-per':6,'I-org':7,'B-nat':8,'I-tim':9,'I-gpe':10,'I-nat':11,'B-art':12,'I-art':13,'B-eve':14,'I-eve':15,'O':16,START_TAG: 17, STOP_TAG: 18}
ix_to_tag={0:'B-geo',1:'B-gpe',2:'B-tim',3:'B-org',4:'I-geo',5:'B-per',6:'I-per',7:'I-org',8:'B-nat',9:'I-tim',10:'I-gpe',11:'I-nat',12:'B-art',13:'I-art',14:'B-eve',15:'I-eve',16:'O',17:START_TAG,18: STOP_TAG}
for long_sentence in sentences:
    for word in long_sentence:
        if word[1] not in syntax_to_ix:
            syntax_to_ix[word[1]] = len(syntax_to_ix)
with torch.no_grad():
    test_right_num=0
    test_total_num=0
    tag_to_ix_test={'B-geo':0,'B-gpe':0,'B-tim':0,'B-org':0,'I-geo':0,'B-per':0,'I-per':0,'I-org':0,'B-nat':0,'I-tim':0,'I-gpe':0,'I-nat':0,'B-art':0,'I-art':0,'B-eve':0,'I-eve':0,'O':0}
    test_stat=get_label_stat(test_data)
    test_predict_list=[]
    test_targets_list=[]
    batch_index=0
    for test_val in range(len(test_data)):
        test_predict_list.append([])
        test_targets_list.append([])
        batched_data,batch_index=form_batch(test_data,sample_num,batch_index)
        batch_word_embedding,sentence_length=transfrom_sentence_to_embed(batched_data,syntax_to_ix,glove,syntax_embeds,targets=None)
        targets = prepare_sequence(batched_data,tag_to_ix,'tag')
        predict_output=model(batch_word_embedding)
        test_predict_list[test_val].extend(num_tag(predict_output[1],ix_to_tag))
        test_targets_list[test_val].extend(num_tag(targets[0].tolist(),ix_to_tag))
voting_true_label=[]
for (c1_pred,c2_pred,bi_lstm_pred,xgb_pred,rf_pred,num) in zip(y_crf1_pred,y_crf2_pred,test_predict_list,y_xgb_pred_standard,y_rf_pred_standard,range(len(test_predict_list))):
    voting_true_label.append([])
    for (c1_label,c2_label,bi_label,xgb_label,rf_label) in zip(c1_pred,c2_pred,bi_lstm_pred,xgb_pred,rf_pred):
        count=Counter([c1_label,c2_label,bi_label,xgb_label,rf_label])
        count_value=list(count.values())
        max_count=max(count_value)
        max_index=count_value.index(max_count)
        count_key=list(count.keys())
        voting_true_label[num].extend([count_key[max_index]])
print(metrics.flat_classification_report(y, voting_true_label, digits=3))
predict_right=0
total_right=0
pred_list=[]
true_list=[]
for y_true,vote_label in zip(y,voting_true_label):
    predict_right=predict_right+np.sum(np.array(y_true)==np.array(vote_label))
    total_right=total_right+len(y_true)
    true_list.extend(y_true)
    pred_list.extend(vote_label)

recall_score