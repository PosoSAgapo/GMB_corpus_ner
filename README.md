# GMB_corpus_ner
This is the repository fot the link https://www.kaggle.com/shoumikgoswami/ner-using-random-forest-and-crf
It uses a ensemble model combines (xgboos,crf,random forest,bilist+attn+crf) and a model based on Bert(since my computer does not support trainning a model like bert,my laptop is only a macbook and I have to use the old computer in my house to use cuda , so I do not have the full result of the Bert,but the code works fine just the speed is very slow which takes a long time to see a proper result)
In BiLSTM, I use the glove vector and concat it with a 50-dim vector which uses it to describe the syntax label,so the word embedding in the BiLSTM is actually 150 dim.

--------------------------------------------------------------------------------------------------------------------------

Running Procedure:
1.Run make_data.py which you can adjust the size of training,val,test dataset,in the experiment  I use training (2349 sentences),val (150 sentences),test (350 sentences) .
2.Run each model's traininig script,like ner_Xgboost.py,Rf.py,CRF.py,Copy_Attn.py,it will save the model automatically
3.Run the vote_classifer.py to get the hard_voting result on the test dataset
4.To balance the dataset, I put a 'balanced' parameter in the sklearn's model, also I give a small value to the  
Result:
On the test set,the result gives like below:
               precision    recall  f1-score   support

       B-art      0.500     0.100     0.167        10
       B-eve      0.750     0.273     0.400        11
       B-geo      0.726     0.869     0.791       335
       B-gpe      0.877     0.758     0.813       198
       B-nat      0.500     0.500     0.500         2
       B-org      0.733     0.664     0.697       211
       B-per      0.749     0.753     0.751       182
       B-tim      0.916     0.840     0.877       169
       I-art      0.000     0.000     0.000         6
       I-eve      1.000     0.111     0.200         9
       I-geo      0.753     0.733     0.743        75
       I-gpe      0.000     0.000     0.000         6
       I-nat      1.000     1.000     1.000         1
       I-org      0.742     0.685     0.712       168
       I-per      0.813     0.880     0.845       217
       I-tim      0.750     0.488     0.592        43
           O      0.990     0.994     0.992      9591
    accuracy      0.958     0.957     0.960     11234
   macro avg      0.694     0.567     0.593     11234
weighted avg      0.959     0.960     0.959     11234

We see that the classifer:
1.reach the weighted accuracy weight of 0.959 and non weighted accuracy 0.958: 
2.reach the f1 score of 0.96:
3.due to the severly unbalanced dataset,like I-art or I-nat could only count for 20~40 counts in this dataset that contains more than 60,000 words,therefore, event I try to balance the dataset ,due to the lack of data,all those models do not perform well on these unbalanced labels.

--------------------------------------------------------------------------------------------------------------------------

Scirpt and file Descibe:
1.models_func.py，contains the all the model and function that used in the NER process
2.data_make.py, spilit the data under the num that you give
3.CRF,ner_Xgboost,Copy_Attn,Rf,Bert_BiLSTM_CRF are different scirpts that you can run
4.event_tensors is the Glove Vector Package,in order to use this package,you also need to download a glove vector,which I use a 100d Glove Vector. You can also choose to use a random word vector,however,based on the accuracy on the BiLSTM, using Glove vector will raise the accuracy for serveral points.
5.GMB_dataset.txt is the dataset used in this task
6.non_O_BiLSTM_CRF_constrained.py is the BiLSTM that does not use O labeled words, however,maybe due to the lack of context,the performance is bad, whichi I did not use it as a model.
7.crf1,crf2,xgb,rf,attn_bilstm_crf are model files which you can load using sklearn's joinlib or torch's torch.load
8.syntax_embeds is the embedding that trained to describe the syntax label which is used to cancat with word vector.
9.train data,val data,test data,are the datas that spilited using make_data.py

--------------------------------------------------------------------------------------------------------------------------

For Glove vector,due to its size,I can not upload it to the github,you can download it at https://nlp.stanford.edu/projects/glove/
I am still training a deeper BiLSTM model,since I did not adjust any hyperparameter (only a 1 layer BiLSTM with 256 hidden dims) to find a nice setting,the result still has space to raise since the BiLSTM model could be improved.
