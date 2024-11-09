#Writing a py code for the naive bayes

#importing required modules
import numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score

#providing data here x represent the train features data and y represents the labels for training
x = np.array([[-1,-1],[-2,-1],[-3,-2],[1,1],[2,1],[3,2]])
y = np.array([1,1,1,2,2,2])
#gaussian naive bayes was called and train data were given
clf = GaussianNB()
clf.fit(x,y)
#an array of values can be given in place of A SINGLE DATA to predict, there might a variable containing test data
pred = clf.predict([[-0.8,-1]])

#evaluating the model
# score = accuracy_score(pred,test_labels)
