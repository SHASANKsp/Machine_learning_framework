#import
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.cluster import  KMeans
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, classification_report, confusion_matrix


#providing data (for a flat file)
data = pd.read_csv() #read the file

X = data.drop('target', axis=1)
y = data['target']

#clean data for missing data and other noice
#
# scaling the features - refer to the scalling.py script
# 
#  
# feature selectio/engineering if needed 
# selecting features based on variance, corelation or PCA or any other selection method - - refer to the feature selection.py script
#
#

#Spliting the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=30)

#use elbow method to find the optimal n_cluster value
n_clusters=4
clf = KMeans(n_clusters)
clf.fit(X_train,y_train)


pred = clf.predict(X_test)

print(clf.cluster_centers_)
print(clf.labels_)
