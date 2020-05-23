import torch
import torch.nn as nn
import pandas as pd
import numpy as np 
START_TAG = "<START>"
STOP_TAG = "<STOP>"

class BiLSTM_CRF(nn.Module):
    def __init__(self, vocab_size, tag_to_ix, embedding_dim, hidden_dim):
        super(BiLSTM_CRF, self).__init__()
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size
        self.tag_to_ix = tag_to_ix
        self.tagset_size = len(tag_to_ix)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim // 2,num_layers=1, bidirectional=True).cuda(0)
        self.hidden2tag = nn.Linear(hidden_dim, self.tagset_size).cuda(0)
        self.transitions = nn.Parameter(torch.randn(self.tagset_size, self.tagset_size)).cuda(0)
        self.transitions.data[tag_to_ix[START_TAG], :] = -10000
        self.transitions.data[:, tag_to_ix[STOP_TAG]] = -10000
        self.hidden = self.init_hidden()
        self.tanh = nn.Tanh()
    def init_hidden(self):
        return (torch.randn(2, 1, self.hidden_dim // 2).cuda(0),
                torch.randn(2, 1, self.hidden_dim // 2).cuda(0))
    def _forward_alg(self, feats):
        init_alphas = torch.full([self.tagset_size], -10000.)
        init_alphas[self.tag_to_ix[START_TAG]] = 0.
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
        self.hidden[0].cuda(0)
        self.hidden[1].cuda(0)
        lstm_out, self.hidden = self.lstm(sentence,self.hidden)
        lstm_out = lstm_out.view(lstm_out.size()[0], self.hidden_dim)
        lstm_feats = self.hidden2tag(lstm_out)
        return lstm_feats.cuda(0)
    def _score_sentence(self, feats, tags):
        score = torch.zeros(1).cuda(0)
        tags = torch.cat([torch.tensor([self.tag_to_ix[START_TAG]], dtype=torch.long), tags[0]]).cuda(0)
        for i, feat in enumerate(feats):
            score = score + self.transitions[tags[i + 1], tags[i]] + feat[tags[i + 1]]
        score = score + self.transitions[self.tag_to_ix[STOP_TAG], tags[-1]]
        return score
    def _viterbi_decode(self, feats):
        backpointers = []
        init_vvars = torch.full((1, self.tagset_size), -10000.)
        init_vvars[0][self.tag_to_ix[START_TAG]] = 0
        forward_var_list = []
        forward_var_list.append(init_vvars)

        for feat_index in range(feats.shape[0]):
            gamar_r_l = torch.stack([forward_var_list[feat_index]] * feats.shape[1]).cuda(0)
            gamar_r_l = torch.squeeze(gamar_r_l).cuda(0)
            next_tag_var = gamar_r_l + self.transitions
            viterbivars_t,bptrs_t = torch.max(next_tag_var,dim=1)

            t_r1_k = torch.unsqueeze(feats[feat_index], 0)
            forward_var_new = torch.unsqueeze(viterbivars_t,0) + t_r1_k.squeeze()

            forward_var_list.append(forward_var_new)
            backpointers.append(bptrs_t.tolist())

        terminal_var = forward_var_list[-1] + self.transitions[self.tag_to_ix[STOP_TAG]]
        best_tag_id = torch.argmax(terminal_var).tolist()
        path_score = terminal_var[0][best_tag_id]
        best_path = [best_tag_id]
        for bptrs_t in reversed(backpointers):
            best_tag_id = bptrs_t[best_tag_id]
            best_path.append(best_tag_id)
        start = best_path.pop()
        assert start == self.tag_to_ix[START_TAG]  # Sanity check
        best_path.reverse()
        return path_score, best_path
    def neg_log_likelihood(self, sentence, tags):
        feats = self._get_lstm_features(sentence)
        forward_score = self._forward_alg(feats)
        gold_score = self._score_sentence(feats, tags)
        return forward_score - gold_score
    def forward(self, sentence): 
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
def word2features(sent, i):
    word = sent[i][0]
    postag = sent[i][1]
    features = {
        'bias': 1.0,
        'word.lower()': word.lower(),
        'word[-3:]': word[-3:],
        'word[-2:]': word[-2:],
        'word.isupper()': word.isupper(),
        'word.istitle()': word.istitle(),
        'word.isdigit()': word.isdigit(),
        'postag': postag,
        'postag[:2]': postag[:2],
    }
    if i > 0:
        word1 = sent[i-1][0]
        postag1 = sent[i-1][1]
        features.update({
            '-1:word.lower()': word1.lower(),
            '-1:word.istitle()': word1.istitle(),
            '-1:word.isupper()': word1.isupper(),
            '-1:postag': postag1,
            '-1:postag[:2]': postag1[:2],
        })
    else:
        features['BOS'] = True

    if i < len(sent)-1:
        word1 = sent[i+1][0]
        postag1 = sent[i+1][1]
        features.update({
            '+1:word.lower()': word1.lower(),
            '+1:word.istitle()': word1.istitle(),
            '+1:word.isupper()': word1.isupper(),
            '+1:postag': postag1,
            '+1:postag[:2]': postag1[:2],
        })
    else:
        features['EOS'] = True
    return features
def sent2features(sent):
    return [word2features(sent, i) for i in range(len(sent))]
def sent2labels(sent):
    return [label for token, postag, label in sent]
def form_batch(sentences,sample_num,batch_index):
    batched_data=sentences[batch_index:batch_index+sample_num]
    return batched_data,batch_index+sample_num
def transfrom_sentence_to_embed(seq,syntax_to_ix,glove,syntax_embeds,targets=None):
    idxs=[]
    index_data=[]
    sentence_length=[]
    sample_num=len(seq)
    data=torch.zeros(len(seq[0]),sample_num,150).cuda(0)
    for i in range(sample_num):
        sentence=seq[i]
        for word,idx in zip(sentence,range(len(sentence))):
            try:
                emb=torch.tensor(glove.embedding(word[0].lower()))
                syn_emb=syntax_embeds(torch.tensor(syntax_to_ix[word[1]]))
                total_emb=torch.cat((emb,syn_emb),dim=0)
            except:
                emb=torch.randn(100)
                syn_emb=syntax_embeds(torch.tensor(syntax_to_ix[word[1]]))
                total_emb=torch.cat((emb,syn_emb),dim=0)
            data[idx,0]=total_emb.cuda(0)
    sentence_length.append(len(seq[0]))
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
def feature_map(sentence):
    feature_list=[]
    for word in sentence:
        feature_list.extend([word[0].istitle(), word[0].islower(), word[0].isupper(), len(word[0]), word[0].isdigit(),  word[0].isalpha()])
    return np.array(feature_list)
def word_feature_map(word):
    return np.array([word.istitle(), word.islower(), word.isupper(), len(word), word.isdigit(),  word.isalpha()])
def form_standard_list(lable_list,train_data):
    reformed_list=[]
    flag=0
    for sentence,sentence_num in zip(train_data,range(len(train_data))):
        reformed_list.append([])
        reformed_list[sentence_num].extend(lable_list[flag:flag+len(sentence)])
        flag=flag+len(sentence)
    return reformed_list