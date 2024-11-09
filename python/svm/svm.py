#Writing a py code for the naive bayes

#importing required modules
import numpy as np
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score


#providing data here x represent the train features data and y represents the labels for training
x = np.array()
y = np.array()
#svc was called and train data were given
clf = SVC(kernel="linear")
clf.fit(x,y)
#an array of values can be given in place of A SINGLE DATA to predict, there might a variable containing test data
pred = clf.predict()

#evaluating the model
# score = accuracy_score(pred,test_labels)
