import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
import torch.nn.utils as utils
import matplotlib as mpl
import matplotlib.pyplot as plt
from random import sample
from transformers import *
import itertools
import pickle
torch.manual_seed(1)
class BiLSTM_CRF(nn.Module):
    def __init__(self, vocab_size, tag_to_ix, embedding_dim, hidden_dim):
        super(BiLSTM_CRF, self).__init__()
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size
        self.tag_to_ix = tag_to_ix
        self.tagset_size = len(tag_to_ix)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim // 2,
                            num_layers=1, bidirectional=True)

        # Maps the output of the LSTM into tag space.
        self.hidden2tag = nn.Linear(hidden_dim, self.tagset_size).cuda(0)

        # Matrix of transition parameters.  Entry i,j is the score of
        # transitioning *to* i *from* j.
        self.transitions = nn.Parameter(torch.randn(self.tagset_size, self.tagset_size)).cuda(0)

        # These two statements enforce the constraint that we never transfer
        # to the start tag and we never transfer from the stop tag
        self.transitions.data[tag_to_ix[START_TAG], :] = -10000
        self.transitions.data[:, tag_to_ix[STOP_TAG]] = -10000

        self.hidden = self.init_hidden()
    def init_hidden(self):
        return (torch.randn(2, 1, self.hidden_dim // 2),
                torch.randn(2, 1, self.hidden_dim // 2))
    def _forward_alg(self, feats):
        # Do the forward algorithm to compute the partition function
        init_alphas = torch.full([self.tagset_size], -10000.).cuda(0)
        # START_TAG has all of the score.
        init_alphas[self.tag_to_ix[START_TAG]] = 0.

        # Wrap in a variable so that we will get automatic backprop
        # Iterate through the sentence
        forward_var_list=[]
        forward_var_list.append(init_alphas)
        feats=feats.squeeze()
        for feat_index in range(feats.shape[0]):
            gamar_r_l = torch.stack([forward_var_list[feat_index]] * feats.shape[1]).cuda(0)
            t_r1_k = torch.unsqueeze(feats[feat_index],0).transpose(0,1).cuda(0)
            aa = gamar_r_l + t_r1_k + self.transitions
            forward_var_list.append(torch.logsumexp(aa,dim=1))
        terminal_var = forward_var_list[-1] + self.transitions[self.tag_to_ix[STOP_TAG]]
        terminal_var = torch.unsqueeze(terminal_var,0)
        alpha = torch.logsumexp(terminal_var, dim=1)[0].cuda(0)
        return alpha
    def _get_lstm_features(self, sentence):
        self.hidden = self.init_hidden()
        lstm_out, self.hidden = self.lstm(sentence)
        seq_unpacked, lens_unpacked=utils.rnn.pad_packed_sequence(lstm_out)
        lstm_out = seq_unpacked.view(lens_unpacked[0], self.hidden_dim)
        lstm_feats = self.hidden2tag(seq_unpacked)
        return lstm_feats.cuda(0)
    def _score_sentence(self, feats, tags):
        # Gives the score of a provided tag sequence
        score = torch.zeros(1).cuda(0)
        tags = torch.cat([torch.tensor([self.tag_to_ix[START_TAG]], dtype=torch.long), tags[0]]).cuda(0)
        for i, feat in enumerate(feats):
            score = score + \
                self.transitions[tags[i + 1], tags[i]] + feat[0][tags[i + 1]]
        score = score + self.transitions[self.tag_to_ix[STOP_TAG], tags[-1]]
        return score
    def _viterbi_decode(self, feats):
        backpointers = []
        # Initialize the viterbi variables in log space
        init_vvars = torch.full((1, self.tagset_size), -10000.).cuda(0)
        init_vvars[0][self.tag_to_ix[START_TAG]] = 0

        # forward_var at step i holds the viterbi variables for step i-1
        forward_var_list = []
        forward_var_list.append(init_vvars)

        for feat_index in range(feats.shape[0]):
            gamar_r_l = torch.stack([forward_var_list[feat_index]] * feats.shape[2]).cuda(0)
            gamar_r_l = torch.squeeze(gamar_r_l).cuda(0)
            next_tag_var = gamar_r_l + self.transitions
            viterbivars_t,bptrs_t = torch.max(next_tag_var,dim=1)

            t_r1_k = torch.unsqueeze(feats[feat_index], 0).cuda(0)
            forward_var_new = torch.unsqueeze(viterbivars_t,0) + t_r1_k.squeeze()

            forward_var_list.append(forward_var_new)
            backpointers.append(bptrs_t.tolist())

        # Transition to STOP_TAG
        terminal_var = forward_var_list[-1] + self.transitions[self.tag_to_ix[STOP_TAG]]
        best_tag_id = torch.argmax(terminal_var).tolist()
        path_score = terminal_var[0][best_tag_id]

        # Follow the back pointers to decode the best path.
        best_path = [best_tag_id]
        for bptrs_t in reversed(backpointers):
            best_tag_id = bptrs_t[best_tag_id]
            best_path.append(best_tag_id)
        # Pop off the start tag (we dont want to return that to the caller)
        start = best_path.pop()
        assert start == self.tag_to_ix[START_TAG]  # Sanity check
        best_path.reverse()
        return path_score, best_path
    def neg_log_likelihood(self, sentence, tags):
        feats = self._get_lstm_features(sentence)
        forward_score = self._forward_alg(feats)
        gold_score = self._score_sentence(feats, tags)
        dif=forward_score - gold_score
        return dif.cuda(0)
    def forward(self, sentence):  # dont confuse this with _forward_alg above.
        lstm_feats = self._get_lstm_features(sentence)
        score, tag_seq = self._viterbi_decode(lstm_feats)
        return score, tag_seq
class getsentence(object):
    def __init__(self, data):
        self.n_sent = 1.0
        self.data = data
        self.empty = False
        agg_func = lambda s: [(w, p, t) for w, p, t in zip(s["Word"].values.tolist(),
                                                           s["POS"].values.tolist(),
                                                           s["Tag"].values.tolist())]
                                                           
        self.grouped = self.data.groupby("Sentence #").apply(agg_func)
        self.sentences = [s for s in self.grouped]
def form_batch(sentences,sample_num,word_to_ix,batch_index):
    batched_data=sentences[batch_index:batch_index+sample_num]
    return batched_data,batch_index+sample_num
def transfrom_sentence_to_embed(seq, to_ix,model):
    word_list=[]
    index_data=[]
    sentence_length=[]
    sample_num=len(seq)
    data=torch.zeros(80,sample_num,3072)
    for i in range(sample_num):
        word_list.append([])
        sentence=seq[i]
        for word in sentence:
            word_list[i].extend([word[0]])
        #index_data.append(torch.tensor(idxs[i], dtype=torch.long))
    for (word,i) in zip(word_list[0],range(len(word_list[0]))):
        word_ids=torch.LongTensor(tokenizer.encode(word))
        word_ids = word_ids.unsqueeze(0)
        out = model(input_ids=word_ids.cuda(0))
        hidden_states = out[2]
        last_four_layers = [hidden_states[i] for i in (-1, -2, -3, -4)]
        cat_hidden_states = torch.cat(tuple(last_four_layers), dim=-1)
        cat_sentence_embedding = torch.mean(cat_hidden_states, dim=1).squeeze()        
        data[i,0] =cat_sentence_embedding.squeeze()
    sentence_length.append(len(word_list[0]))
    return data,sentence_length
def prepare_sequence(seq, to_ix,extract_label):
    idxs=[]
    index_data=[]
    sample_num=len(seq)
    for i in range(sample_num):
        idxs.append([])
        sentence=seq[i]
        for word in sentence:
            idxs[i].extend([to_ix[word[2]]])
        index_data.append(torch.tensor(idxs[i], dtype=torch.long))
    return index_data
def get_label_stat(sentence_list):
    label_dict={'B-geo':0,'B-gpe':0,'B-tim':0,'B-org':0,'I-geo':0,'B-per':0,'I-per':0,'I-org':0,'B-nat':0,'I-tim':0,'I-gpe':0,'I-nat':0,'B-art':0,'I-art':0,'B-eve':0,'I-eve':0,'O':0}
    for sentence in sentence_list:
        for word in sentence:
            label_dict[word[2]]=label_dict[word[2]]+1
    return label_dict
        

def acc(packed_predict,label,tag_to_ix):
    predict=packed_predict[1]
    label=label[0].tolist()
    right_predicit_num=torch.sum(torch.tensor(predict)==torch.tensor(label))
    total_num=len(predict)
    key=list(tag_to_ix.keys())
    for pred,lab in zip(predict,label):
        if (pred==lab)&(lab==0):
            tag_to_ix[key[0]]=tag_to_ix[key[0]]+1
        if (pred==lab)&(lab==1):
            tag_to_ix[key[1]]=tag_to_ix[key[1]]+1
        if (pred==lab)&(lab==2):
            tag_to_ix[key[2]]=tag_to_ix[key[2]]+1
        if (pred==lab)&(lab==3):
            tag_to_ix[key[3]]=tag_to_ix[key[3]]+1
        if (pred==lab)&(lab==4):
            tag_to_ix[key[4]]=tag_to_ix[key[4]]+1
        if (pred==lab)&(lab==5):
            tag_to_ix[key[5]]=tag_to_ix[key[5]]+1
        if (pred==lab)&(lab==6):
            tag_to_ix[key[6]]=tag_to_ix[key[6]]+1
        if (pred==lab)&(lab==7):
            tag_to_ix[key[7]]=tag_to_ix[key[7]]+1
        if (pred==lab)&(lab==8):
            tag_to_ix[key[8]]=tag_to_ix[key[8]]+1
        if (pred==lab)&(lab==9):
            tag_to_ix[key[9]]=tag_to_ix[key[9]]+1
        if (pred==lab)&(lab==10):
            tag_to_ix[key[10]]=tag_to_ix[key[10]]+1
        if (pred==lab)&(lab==11):
            tag_to_ix[key[11]]=tag_to_ix[key[11]]+1
        if (pred==lab)&(lab==12):
            tag_to_ix[key[12]]=tag_to_ix[key[12]]+1
        if (pred==lab)&(lab==13):
            tag_to_ix[key[13]]=tag_to_ix[key[13]]+1
        if (pred==lab)&(lab==14):
            tag_to_ix[key[14]]=tag_to_ix[key[14]]+1
        if (pred==lab)&(lab==15):
            tag_to_ix[key[15]]=tag_to_ix[key[15]]+1
        if (pred==lab)&(lab==16):
            tag_to_ix[key[16]]=tag_to_ix[key[16]]+1
    return right_predicit_num,total_num,tag_to_ix
    
if __name__== '__main__':
    data = pd.read_csv("GMB_dataset.txt", sep="\t", header=None, encoding="latin1")
    data.columns = data.iloc[0]
    data = data[1:]
    data.columns = ['Index','Sentence #','Word','POS','Tag']
    data = data.reset_index(drop=True)
    sample_num=1
    acc_step=2
    test_step=5
    getter = getsentence(data)
    sentences = getter.sentences
    START_TAG = "<START>"
    STOP_TAG = "<STOP>"
    EMBEDDING_DIM = 3072
    HIDDEN_DIM = 3072
    loss=[]
    tokenizer=BertTokenizer.from_pretrained('bert-base-uncased')
    bert_model=BertModel.from_pretrained('bert-base-uncased',output_hidden_states=True).cuda(0)
    # Make up some training data
    word_to_ix = {}
    for long_sentence in sentences:
        for word in long_sentence:
            if word[0] not in word_to_ix:
                word_to_ix[word[0]] = len(word_to_ix)
    val_data=sample(sentences,300)
    for vd in val_data:
        sentences.remove(vd)
    test_data=sample(sentences,300)
    for td in test_data:
        sentences.remove(td)
    train_data=sentences
    tag_to_ix = {'B-geo':0,'B-gpe':1,'B-tim':2,'B-org':3,'I-geo':4,'B-per':5,'I-per':6,'I-org':7,'B-nat':8,'I-tim':9,'I-gpe':10,'I-nat':11,'B-art':12,'I-art':13,'B-eve':14,'I-eve':15,'O':16,START_TAG: 17, STOP_TAG: 18}
    tag_to_ix_no_start_end = {'B-geo':0,'B-gpe':1,'B-tim':2,'B-org':3,'I-geo':4,'B-per':5,'I-per':6,'I-org':7,'B-nat':8,'I-tim':9,'I-gpe':10,'I-nat':11,'B-art':12,'I-art':13,'B-eve':14,'I-eve':15,'O':16}
    model_lstm = BiLSTM_CRF(len(word_to_ix), tag_to_ix, EMBEDDING_DIM, HIDDEN_DIM).cuda(0)
    optimizer = optim.Adagrad(itertools.chain(bert_model.parameters(),model_lstm.parameters()), lr=0.05, weight_decay=1e-4)
    loss_list=[]
    val_stat=get_label_stat(val_data)
    test_stat=get_label_stat(test_data)
    dict_key=list(tag_to_ix_no_start_end.keys())
    for epoch in range(5):  # again, normally you would NOT do 300 epochs, it is toy data
        batch_index=0
        print(epoch)
        for time in range(len(train_data)):
            batched_data,batch_index=form_batch(train_data,sample_num,word_to_ix,batch_index)
            batch_word_embedding,sentence_length=transfrom_sentence_to_embed(batched_data,word_to_ix,bert_model)
            packed=utils.rnn.pack_padded_sequence(batch_word_embedding,sentence_length,enforce_sorted=False)
            targets = prepare_sequence(batched_data,tag_to_ix,'tag')
            loss = model_lstm.neg_log_likelihood(packed.cuda(0), targets)/acc_step
            loss.backward()
            if time%acc_step==0:
                loss_list.append(int(loss))
                optimizer.step()
                optimizer.zero_grad()
            if time % test_step==0:
                print(time)
                bert_model.eval()
                model_lstm.eval()
                val_right_num=0
                val_total_num=0
                test_right_num=0
                test_total_num=0
                tag_to_ix_val ={'B-geo':0,'B-gpe':0,'B-tim':0,'B-org':0,'I-geo':0,'B-per':0,'I-per':0,'I-org':0,'B-nat':0,'I-tim':0,'I-gpe':0,'I-nat':0,'B-art':0,'I-art':0,'B-eve':0,'I-eve':0,'O':0}
                tag_to_ix_test={'B-geo':0,'B-gpe':0,'B-tim':0,'B-org':0,'I-geo':0,'B-per':0,'I-per':0,'I-org':0,'B-nat':0,'I-tim':0,'I-gpe':0,'I-nat':0,'B-art':0,'I-art':0,'B-eve':0,'I-eve':0,'O':0}
                batch_val_index=0
                for time_val in range(len(val_data)):
                    batched_data,batch_val_index=form_batch(val_data,sample_num,word_to_ix,batch_val_index)
                    batch_word_embedding,sentence_length=transfrom_sentence_to_embed(batched_data,word_to_ix,bert_model)
                    packed=utils.rnn.pack_padded_sequence(batch_word_embedding,sentence_length,enforce_sorted=False)
                    targets = prepare_sequence(batched_data,tag_to_ix,'tag')
                    predict_output=model_lstm(packed.cuda(0))
                    right_predicit_num,total_num,tag_to_ix_val=acc(predict_output,targets,tag_to_ix_val)
                    val_right_num=val_right_num+right_predicit_num
                    val_total_num=val_total_num+total_num
                print('vallllllllllllll',val_right_num)
                batch_test_index=0
                for time_test in range(len(test_data)):
                    batched_data,batch_test_index=form_batch(test_data,sample_num,word_to_ix,batch_test_index)
                    batch_word_embedding,sentence_length=transfrom_sentence_to_embed(batched_data,word_to_ix,bert_model)
                    packed=utils.rnn.pack_padded_sequence(batch_word_embedding,sentence_length,enforce_sorted=False)
                    targets = prepare_sequence(batched_data,tag_to_ix,'tag')
                    predict_output=model_lstm(packed.cuda(0))
                    right_predicit_num,total_num,tag_to_ix_test=acc(predict_output,targets,tag_to_ix_test)
                    test_right_num=test_right_num+right_predicit_num
                    test_total_num=test_total_num+total_num
                bert_model.train()
                model_lstm.train()
                print('total right percentage in val set',int(val_right_num)/int(val_total_num))
                for dkey in dict_key:
                    try:
                        print(dkey,'percentage',tag_to_ix_val[dkey]/val_stat[dkey])
                    except:
                        print(dkey,'in stat is 0')
                print('total right percentage in test set',int(test_right_num)/int(test_total_num))
                for dkey in dict_key:
                    try:
                        print(dkey,'percentage',tag_to_ix_val[dkey]/val_stat[dkey])
                    except:
                        print(dkey,'in stat is 0')
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                

    