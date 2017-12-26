import pickle #實驗組
import numpy as np
from pprint import pprint
from sklearn.metrics import roc_auc_score #<=====AUC
f = open('model.pkl', 'rb')
clf = pickle.load(f)
pprint(clf)
valid_X = np.load('validationX.npy')
valid_Y = np.load('validationY.npy')
print(valid_X.shape)
print(valid_Y.shape)
y_score=clf.predict_proba(valid_X)[:, 1]
y_decision=clf.decision_function(valid_X)
print(y_score.shape)
print('AUC: {:.3f}'.format(roc_auc_score(valid_Y, y_score)))#<=====AUC
print('AUC(2): {:.3f}'.format(roc_auc_score(valid_Y, y_decision)))#<=====AUC
y_pred = clf.predict(valid_X)
print('Test Accuracy: {:.3f}'.format(clf.score(valid_X, valid_Y)))
