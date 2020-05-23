import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import f1_score
from sklearn_crfsuite.metrics import flat_classification_report
from sklearn.metrics import classification_report, make_scorer
import copy
import random
from models_func import getsentence,word2features,sent2labels,feature_map
import pickle
from sklearn.externals import joblib
labels=['B-geo', 'B-gpe', 'B-tim', 'B-org', 'I-geo', 'B-per', 'I-per', 'I-org', 'B-nat', 'I-tim', 'I-gpe', 'I-nat', 'B-art', 'I-art', 'B-eve', 'I-eve']
train_data=pickle.load(open('train_data','rb'))
word_list=[]
lable_list=[]
for sentence in train_data:
    for word in sentence:
        word_list.append(word[0])
        lable_list.append(word[2])
def word_feature_map(word):
    return np.array([word.istitle(), word.islower(), word.isupper(), len(word), word.isdigit(),  word.isalpha()])
words = [word_feature_map(word) for word in word_list]
model=XGBClassifier(class_weight='balanced')
# model.fit(np.array(words),lable_list)
# y_pred=model.predict(np.array(x_test))
# report = classification_report(y_pred=y_pred, y_true=y_test)
learning_rate = [0.0001, 0.001, 0.01, 0.1, 0.2, 0.3]
param_grid = dict(learning_rate=learning_rate)
kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=7)
grid_search = GridSearchCV(model, param_grid, scoring="recall_macro", n_jobs=-1, cv=kfold)
grid_result = grid_search.fit(np.array(words), lable_list)
model.fit(np.array(words), lable_list)
y_pred = model.predict(np.array(words))
print(classification_report(lable_list, y_pred,digits=3))
reformed_list=[]
flag=0
joblib.dump(model,'xgb')
for sentence,sentence_num in zip(train_data,range(len(train_data))):
    reformed_list.append([])
    reformed_list[sentence_num].extend(lable_list[flag:flag+len(sentence)])
    flag=flag+len(sentence)
