#import
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, root_mean_squared_error
import matplotlib.pyplot as plt



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


clf = LinearRegression()
clf.fit(X_train,y_train)

pred = clf.predict(X_test)


rmse = root_mean_squared_error(y_test, pred)
r2_xgb = r2_score(y_test, pred)
print(f"RMSE: {rmse:.2f}")
print(f"R-squared (R²): {r2_xgb:.2f}")

plt.figure()
plt.scatter(y_test,pred)
plt.ylabel("Predicted")
plt.ylabel("Actual")
plt.show()
