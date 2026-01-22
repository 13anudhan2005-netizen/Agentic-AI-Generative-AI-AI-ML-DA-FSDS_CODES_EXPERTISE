import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


dataset = pd.read_csv(r"Z:\FSDS\Classroom. 9AM Datascience\30-31 dec logictick regression\15. Logistic regression with future prediction\15. Logistic regression with future prediction\Social_Network_Ads.csv")
dataset

x=dataset.iloc[:,[2,3]].values
y=dataset.iloc[:,-1]

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.20,random_state=0)

from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
x_train=sc.fit_transform(x_train)
x_test=sc.transform(x_test)

from sklearn.linear_model import LogisticRegression
classifier=LogisticRegression()
classifier.fit(x_train,y_train)

y_pred=classifier.predict(x_test)

from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, y_pred)
print(cm)

from sklearn.metrics import accuracy_score
ac=accuracy_score(y_test,y_pred)
print(ac)

from sklearn.metrics import classification_report
cr =classification_report(y_test,y_pred)
print(cr)

bias = classifier.score(x_train, y_train)
bias

variance = classifier.score(x_test, y_test)
variance

# Visualising the Training set results

from matplotlib.colors import ListedColormap
x_set, y_set = x_train, y_train
x1, x2 = np.meshgrid(np.arange(start = x_set[:, 0].min() - 1, stop = x_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = x_set[:, 1].min() - 1, stop = x_set[:, 1].max() + 1, step = 0.01))
plt.contourf(x1, x2, classifier.predict(np.array([x1.ravel(), x2.ravel()]).T).reshape(x1.shape),
             alpha = 0.75, cmap = ListedColormap(('red', 'green')))
plt.xlim(x1.min(), x1.max())
plt.ylim(x2.min(), x2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(x_set[y_set == j, 0], x_set[y_set == j, 1],
                c = ListedColormap(('red', 'green'))(i), label = j)
plt.title('Logistic Regression (Training set)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
plt.show()

# Visualising the Test set results

x_set, y_set = x_test, y_test
x1, x2 = np.meshgrid(np.arange(start = x_set[:, 0].min() - 1, stop = x_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = x_set[:, 1].min() - 1, stop = x_set[:, 1].max() + 1, step = 0.01))
plt.contourf(x1, x2, classifier.predict(np.array([x1.ravel(), x2.ravel()]).T).reshape(x1.shape),
             alpha = 0.75, cmap = ListedColormap(('red', 'green')))
plt.xlim(x1.min(), x1.max())
plt.ylim(x2.min(), x2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(x_set[y_set == j, 0], x_set[y_set == j, 1],
                c = ListedColormap(('red', 'green'))(i), label = j)
plt.title('Logistic Regression (Test set)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
plt.show()
# FUTURE PREDICTION
dataset1 = pd.read_csv(r"Z:\FSDS\Classroom. 9AM Datascience\30-31 dec logictick regression\15. Logistic regression with future prediction\15. Logistic regression with future prediction\Future prediction1.csv")
dataset1

d2=dataset1.copy()
dataset1 =dataset1.iloc[:,[2,3]].values

M=sc.transform(dataset1)

d2['y_pred1']=classifier.predict(M)
d2.to_csv('final1.csv',index=False)
print("Future prediction saved as final1.csv")
import os
os.getcwd()


dataset2 = pd.read_csv(r"Z:\FSDS\Classroom. 9AM Datascience\30-31 dec logictick regression\15. Logistic regression with future prediction\15. Logistic regression with future prediction\Future prediction1.csv")
df_final=dataset2.copy()
# Select columns by index
dataset2 = dataset2.iloc[:, [2, 3]].values
dataset2=sc.transform(dataset2)

print("Rows in dataset2:", len(dataset2))
print("Rows in df_final:", len(df_final))

# Predict
df_final['y_pred1'] = classifier.predict(dataset2)

# Save
df_final.to_csv('final.csv', index=False)


from sklearn.metrics import roc_auc_score, roc_curve
import matplotlib.pyplot as plt

# Get predicted probabilities for the positive class
y_pred_prob = classifier.predict_proba(x_test)[:, 1]

# Calculate AUC score
auc_score = roc_auc_score(y_test, y_pred_prob)
print("AUC Score:", auc_score)

# Get ROC curve values
fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob)

# Plot ROC curve
plt.figure(figsize=(8,6))
plt.plot(fpr, tpr, label=f'Logistic Regression (AUC={auc_score:.2f})')
plt.plot([0,1], [0,1], 'k--')  # diagonal line
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend(loc='lower right')
plt.grid()
plt.show()