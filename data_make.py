import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import cross_val_predict, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn_crfsuite import CRF, scorers, metrics
from sklearn_crfsuite.metrics import flat_classification_report
from sklearn.metrics import classification_report, make_scorer
from sklearn.model_selection import RandomizedSearchCV
import scipy.stats
import eli5
from models_func import getsentence
import pickle
from random import sample
data = pd.read_csv("GMB_dataset.txt", sep="\t", header=None, encoding="latin1")
data.columns = data.iloc[0]
data = data[1:]
data.columns = ['Index','Sentence #','Word','POS','Tag']
data = data.reset_index(drop=True)
getter = getsentence(data)
sentences = getter.sentences
val_data=sample(sentences,150)
for vd in val_data:
    sentences.remove(vd)
test_data=sample(sentences,500)
for td in test_data:
    sentences.remove(td)
train_data=sentences
f=open('train_data','wb')
pickle.dump(train_data,f)
f.close()
f=open('val_data','wb')
pickle.dump(val_data,f)
f.close()
f=open('test_data','wb')
pickle.dump(test_data,f)
f.close()

