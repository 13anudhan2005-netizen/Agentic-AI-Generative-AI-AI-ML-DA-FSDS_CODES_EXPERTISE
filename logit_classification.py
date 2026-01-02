#logit classification
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv(r"C:\Users\mohur\Downloads\logit classification.csv")
 
x = dataset.iloc[:, [2, 3]].values
y = dataset.iloc[:, -1].values

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.20, random_state=0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler() 
X_train = sc.fit_transform(x_train)
X_test = sc.transform(x_test)

# Training the Logistic Regression model on the Training set
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression()
classifier.fit(x_train, y_train)


y_pred = classifier.predict(x_test)


# Predicting the Test set results
y_pred = classifier.predict(X_test)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
cm


from sklearn.metrics import accuracy_score
ac = accuracy_score(y_test, y_pred)
ac

# This is to get the Classification Report
from sklearn.metrics import classification_report
cr = classification_report(y_test, y_pred)
cr


#------- future predition------


dataset1= pd.read_csv(r"C:\Users\mohur\Downloads\2.LOGISTIC REGRESSION CODE\2.LOGISTIC REGRESSION CODE\final1.csv")

dataset1 = dataset1.iloc[:, [2, 3]].values


from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
M = sc.fit_transform(dataset1)

y_pred1 = pd.DataFrame()


d2 ['y_pred1'] = classifier.predict(M)

d2.to_csv('final1.csv')









#**********************************************************************


from sklearn.metrics import roc_auc_score, roc_curve
y_pred_prob = classifire .predict_proba(x_test)[:,1]

auc_score = roc_auc_score(y_true, y_pred_prob)
auc_score

fpr, trp, thresholds = roc_curve(y_test, y_pred_prob)

plt.figure(figsize=(8,6))
plt.plot(fpr,tpr, label=f'Logistic Regrssion (AUC ={auc_score:.2f})')
plt.plot([0,1], [0,1]),'k---'
plt.xlabel('False Positive Rate')
plt.ylabel('True Psitive Rate')
plt.title('ROC curve')














