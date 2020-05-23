import pandas as pd
import numpy as np
#Data visualisation
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(color_codes=True)
sns.set(font_scale=1)
#Modeling
from sklearn.model_selection import cross_val_predict, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn_crfsuite import CRF, scorers, metrics
from sklearn_crfsuite.metrics import flat_classification_report
from sklearn.metrics import classification_report, make_scorer
from sklearn.model_selection import RandomizedSearchCV
import scipy.stats
import eli5
from sklearn.model_selection import train_test_split
from sklearn.externals import joblib
import pickle
from models_func import getsentence,word2features,sent2features,sent2labels
data = pd.read_csv("GMB_dataset.txt", sep="\t", header=None, encoding="latin1")
data.columns = data.iloc[0]
data = data[1:]
data.columns = ['Index','Sentence #','Word','POS','Tag']
data = data.reset_index(drop=True)
def sent2features(sent):
    return [word2features(sent, i) for i in range(len(sent))]
def sent2labels(sent):
    return [label for token, postag, label in sent]
labels=['B-geo', 'B-gpe', 'B-tim', 'B-org', 'I-geo', 'B-per', 'I-per', 'I-org', 'B-nat', 'I-tim', 'I-gpe', 'I-nat', 'B-art', 'I-art', 'B-eve', 'I-eve','O']
train_data=pickle.load(open('train_data','rb'))
X = [sent2features(s) for s in train_data]
y = [sent2labels(s) for s in train_data]
crf = CRF(algorithm='lbfgs',
          c1=0.1,
          c2=0.1,
          max_iterations=100,
          all_possible_transitions=False)
pred = cross_val_predict(estimator=crf, X=X, y=y, cv=5)
report = flat_classification_report(y_pred=pred, y_true=y)
print(report)
crf3 = CRF(algorithm='lbfgs',
          max_iterations=100,
          all_possible_transitions=False)
pred = cross_val_predict(estimator=crf, X=X, y=y, cv=5)
params_space = {
    'c1': scipy.stats.expon(scale=1.5),
    'c2': scipy.stats.expon(scale=0.05),
}
f1_scorer = make_scorer(metrics.flat_f1_score,average='weighted', labels=labels)
rs = RandomizedSearchCV(crf, params_space,
                        cv=3,
                        verbose=1,
                        n_jobs=-1,
                        n_iter=50,
                        scoring=f1_scorer)
rs.fit(X, y)
sorted_labels = sorted(
    labels,
    key=lambda name: (name[1:], name[0]))
crf3 = rs.best_estimator_
train_test_split
y_pred = crf3.predict(X)
print(metrics.flat_classification_report(y, y_pred, labels=sorted_labels, digits=3))
crf3.fit(X,y)
joblib.dump(crf3,'crf2.pickle')


