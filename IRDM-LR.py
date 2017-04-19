import pandas as pd 
import numpy as np
from sklearn import preprocessing

# load and split
train_data= pd.read_csv('../irdm/MSLR-WEB10K/Fold1/train.txt', header=None)
vali_data = pd.read_csv('../irdm/MSLR-WEB10K/Fold1/vali.txt', header=None)
test_data = pd.read_csv('../irdm/MSLR-WEB10K/Fold1/test.txt', header=None)
train_df = pd.DataFrame(train_data[0].str.split(' ',2).tolist(),columns = ['relevancy','qid','features'])
vali_df = pd.DataFrame(vali_data[0].str.split(' ',2).tolist(),columns = ['relevancy','qid','features'])
test_df = pd.DataFrame(test_data[0].str.split(' ',2).tolist(),columns = ['relevancy','qid','features'])

# splits 136 features
features = list(range(1, 137))
train_feature_df = pd.DataFrame(train_df['features'].str.split(' ',135).tolist(), columns = features)
vali_feature_df = pd.DataFrame(vali_df['features'].str.split(' ',135).tolist(), columns = features)
test_feature_df = pd.DataFrame(test_df['features'].str.split(' ',135).tolist(), columns = features)

# drop old feature column
train_df = train_df.drop(['features'], axis=1)
vali_df = vali_df.drop(['features'], axis=1)
test_df = test_df.drop(['features'], axis=1)

# add new divided feature column to main df
train_df = pd.concat([train_df,train_feature_df], axis=1)
vali_df = pd.concat([vali_df,vali_feature_df], axis=1)
test_df = pd.concat([test_df,test_feature_df], axis=1)

# rel type int
train_df.relevancy = train_df.relevancy.astype(np.int64)
vali_df.relevancy = vali_df.relevancy.astype(np.int64)
test_df.relevancy = test_df.relevancy.astype(np.int64)

train_feat = train_df.columns[2:138]
vali_feat = vali_df.columns[2:138]
test_feat = test_df.columns[2:138]

train_df_all_feat = train_df[train_feat]
vali_df_all_feat = vali_df[vali_feat]
test_df_all_feat = test_df[test_feat]

# convert values to integers (removes : )
# + prepares input for lr
le = preprocessing.LabelEncoder()

train_df_all_feat = train_df_all_feat.apply(le.fit_transform)
vali_df_all_feat = vali_df_all_feat.apply(le.fit_transform)
test_df_all_feat = test_df_all_feat.apply(le.fit_transform)

#train_df_small_feat = train_df_all_feat[1,3].values

train_df_small_feat = train_df_all_feat.drop([2,4,5,6,7,8,9,10,11,12,13,14,15,17,19,20,22,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,47,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63,64,65,66,67,68,69,70,76,77,78,79,80,81,82,83,84,85,86,87,88,89,90,91,92,93,94,95,96,97,98,99], axis=1)
vali_df_small_feat = vali_df_all_feat.drop([2,4,5,6,7,8,9,10,11,12,13,14,15,17,19,20,22,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,47,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63,64,65,66,67,68,69,70,76,77,78,79,80,81,82,83,84,85,86,87,88,89,90,91,92,93,94,95,96,97,98,99], axis=1)
test_df_small_feat = test_df_all_feat.drop([2,4,5,6,7,8,9,10,11,12,13,14,15,17,19,20,22,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,47,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63,64,65,66,67,68,69,70,76,77,78,79,80,81,82,83,84,85,86,87,88,89,90,91,92,93,94,95,96,97,98,99], axis=1)


import seaborn as sns
mean, cov = [0, 1], [(1, .5), (.5, 1)]

x, y = np.random.multivariate_normal(mean, cov, 1000).T
with sns.axes_style("darkgrid"):
    f = sns.jointplot(x=train_df_small_feat[16], y=train_df['relevancy'], kind="reg", color="k");

ax = sns.boxplot(x=train_df_small_feat[16], y=train_df['relevancy'], linewidth=2.5)
f.set_axis_labels('tf idf body', 'relevancy')

train_df_targets = np.array(train_df.relevancy).T
vali_df_targets = np.array(vali_df.relevancy).T

from sklearn.linear_model import LogisticRegression
lr = LogisticRegression()
lr.fit(train_df_small_feat, train_df_targets)


pd.set_option('display.float_format', lambda x: '%.3f' % x)

pred_rel = []
highest_prob = []
rel_prob = []
all_prob = []

for i in range(0, len(vali_df)):
    five_prob=[]
    query = vali_df_small_feat._slice(slice(i,i+1))    
    rel = query.iloc[0]

    query_df_all_feat = query[query.columns.values]

    predicted_rel = lr.predict(query_df_all_feat)
    pred_rel.append(predicted_rel)
    predicted_rel_prob = lr.predict_proba(query_df_all_feat)
    
    rel_prob.append(predicted_rel_prob[0])
    five_prob.append(predicted_rel_prob[0][0])
    five_prob.append(predicted_rel_prob[0][1])
    five_prob.append(predicted_rel_prob[0][2])
    five_prob.append(predicted_rel_prob[0][3])
    five_prob.append(predicted_rel_prob[0][4])
    all_prob.append(five_prob)
    highest_prob.append(max(five_prob))
df_pred_rel = pd.DataFrame(pred_rel)
df_rel_prob = pd.DataFrame(rel_prob)

pd_highest_prob = pd.DataFrame(highest_prob, columns = ['highest'])
pd_all_prob = pd.DataFrame(all_prob, columns =['0','1','2','3','4'])

from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support, log_loss
from sklearn import metrics

print("Accuracy Score LR:", accuracy_score(vali_df_targets, df_pred_rel))
print("Confusion Matrix LR: \n", confusion_matrix(vali_df_targets, df_pred_rel))
print(metrics.classification_report(vali_df_targets, df_pred_rel))

# str qid -> int qid

train_df['qid'] = train_df.qid.str.replace('qid:' , '').astype('int')
vali_df['qid'] = vali_df.qid.str.replace('qid:' , '').astype('int')
test_df['qid'] = test_df.qid.str.replace('qid:' , '').astype('int')

vali_predict = vali_df.copy()
vali_predict['relevancy'] = df_pred_rel
vali_predict = pd.concat([vali_predict, pd_highest_prob], axis=1)
vali_predict.head(10)

# add final decider to df
sorted_df = vali_predict.sort_values(by=['qid','relevancy','highest'], ascending=[True, False, False])
sorted_df.head(10)

def dcg(predicted_rel, rank):
    predicted_rel = np.asarray(predicted_rel)[:rank]
    n_relevances = len(predicted_rel)
    if n_relevances == 0:
        return float(0)

    discounts = np.log2(np.arange(n_relevances) + 2)
    return np.sum(predicted_rel / discounts)


def ndcg(predicted_rel, rank):
    high_sc_dcg = dcg(sorted(predicted_rel, reverse=True), rank)
    if high_sc_dcg == 0:
        return float(0)
    print(high_sc_dcg)

    return dcg(predicted_rel, rank) / high_sc_dcg


# lets test our final ranking results

# True Relevance DCG
dcg([0, 0, 1, 0, 1, 2, 1, 1, 0, 0], rank=10)

# Predicted Relevance DCG
dcg([0, 0, 1, 0, 3, 1, 1, 3, 0, 1], rank=10)


# True Relevance NDCG
ndcg([0, 0, 1, 0, 1, 2, 1, 1, 0, 0], rank=10)


# Predicted NDCG
ndcg([0, 0, 1, 0, 3, 1, 1, 3, 0, 1], rank=10)


# Mean Avg Precision (MAP)
# Results from df output for first 10

tr = [0, 0, 1, 0, 1, 2, 1, 1, 0, 0]
pred = [0, 0, 1, 0, 3, 1, 1, 3, 0, 1]

def avg_pred(true, pred, rank=10):
    if (len(pred) > rank):
        pred = pred[:rank]

    score = 0.0
    num_hits = 0.0

    for i,p in enumerate(pred):
        if p in true and p not in pred[:i]:
            num_hits += 1.0
            score += num_hits / (i+1.0)

    if not true:
        return 0.0

    return score / min(len(true), rank)

def mean_avg_pred(true, pred, rank=10):

    return np.mean([avg_pred(true,pred,rank) for a,p in zip(true, pred)])

mean_avg_pred(tr, pred)

def ROC(label,result):
    %matplotlib inline
    from sklearn.utils import shuffle
    from sklearn.metrics import roc_curve, auc, precision_score, roc_auc_score
    from sklearn.preprocessing import label_binarize
    import pylab as pl
    import numpy as py
    Y = np.array(label)
    truth = label_binarize(label, classes=[0,1,2,3,4])
    n_classes = truth.shape[1]
    pred = label_binarize(result, classes=[0,1,2,3,4])
    # Compute ROC curve and ROC area for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(truth[:, i], pred[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
        
    # Plot of a ROC curve for a specific class
    pl.figure()
    pl.plot(fpr[2], tpr[2], label='ROC curve (area = %0.2f)' % roc_auc[2])
    pl.plot([0, 1], [0, 1], 'k--')
    pl.xlim([0.0, 1.0])
    pl.ylim([0.0, 1.05])
    pl.xlabel('FP')
    pl.ylabel('TP')
    pl.legend(loc="lower right")
    pl.show()

    for i in range(n_classes):
        pl.plot(fpr[i], tpr[i], label='ROC curve of class {0} (area = {1:0.2f})'
                                       ''.format(i, roc_auc[i]))

    pl.plot([0, 1], [0, 1], 'k--')
    pl.xlim([0.0, 1.0])
    pl.ylim([0.0, 1.05])
    pl.xlabel('FP')
    pl.ylabel('TP')
    pl.legend(loc="lower right")
    pl.show()


ROC(vali_df['relevancy'], vali_predict['relevancy'])

