import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import cross_val_predict, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn_crfsuite import CRF, scorers, metrics
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import GridSearchCV
from sklearn_crfsuite.metrics import flat_classification_report
from sklearn.metrics import classification_report, make_scorer
from sklearn.model_selection import RandomizedSearchCV
import sklearn
import scipy.stats
import eli5
import pickle
from sklearn.externals import joblib

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
model=RandomForestClassifier(class_weight='balanced')
n_estimators = [100, 300, 500,1000]
learning_rate=[0.0001, 0.001, 0.01, 0.1, 0.2, 0.3]
param_grid = dict(n_estimators=n_estimators)
kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=7)
grid_search = GridSearchCV(model, param_grid, scoring="f1_macro", n_jobs=-1, cv=kfold)
grid_result = grid_search.fit(np.array(words), lable_list)
model.fit(np.array(words), lable_list)
y_pred = model.predict(np.array(words))
print(classification_report(lable_list, y_pred,digits=3))
reformed_list=[]
flag=0
joblib.dump(model,'rf')
for sentence,sentence_num in zip(train_data,range(len(train_data))):
    reformed_list.append([])
    reformed_list[sentence_num].extend(lable_list[flag:flag+len(sentence)])
    flag=flag+len(sentence)
    
