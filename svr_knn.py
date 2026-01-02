import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv(r"C:\Users\mohur\Downloads\emp_sal.csv")

X = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, 2].values


# svr model
from sklearn.svm import SVR
svr_regressor = SVR(kernel='poly', degree = 4, gamma = 'auto', C=5 )
svr_regressor.fit(X,y)

svr_model_pred = svr_regressor.predict([[6.5]])
print(svr_model_pred)

# knn model 
from sklearn.neighbors import KNeighborsRegressor
knn_reg_model = KNeighborsRegressor(n_neighbors=2,weights='distance', leaf_size=30 )
knn_reg_model.fit(X,y)

knn_reg_pred = knn_reg_model.predict([[6.5]])
print(knn_reg_pred)

# decission tree 
from sklearn.tree import DecisionTreeRegressor
dtr_reg_model = DecisionTreeRegressor(criterion='poisson', max_depth=3, random_state=0)
dtr_reg_model.fit(X,y)

dtr_reg_pred = dtr_reg_model.predict([[6.5]])
print(dtr_reg_pred)


# Random forest 
from sklearn.ensemble import RandomForestRegressor
rfr_reg_model = RandomForestRegressor(n_estimators=6, random_state=6)
rfr_reg_model.fit(X,y)

rfr_reg_pred = rfr_reg_model.predict([[6.5]])
print(rfr_reg_pred)

# Create high-resolution grid for smooth plotting
X_grid = np.arange(min(X), max(X), 0.01)
X_grid = X_grid.reshape((len(X_grid), 1))

# Plot KNN Regression results
plt.figure()
plt.scatter(X, y)
plt.plot(X_grid, knn_reg_model.predict(X_grid), color='red')
plt.title("KNN Regression Graph")
plt.xlabel("Level / Experience")
plt.ylabel("Salary")
plt.show()

#SVR Graph
plt.figure()
plt.scatter(X, y, color='red')
plt.plot(X_grid, svr_regressor.predict(X_grid))
plt.title("SVR Regression (Polynomial Kernel)")
plt.xlabel("Level / Experience")
plt.ylabel("Salary")
plt.show()

#Decision Tree Regression Graph
plt.figure()
plt.scatter(X, y, color='green')
plt.plot(X_grid, dtr_reg_model.predict(X_grid))
plt.title("Decision Tree Regression")
plt.xlabel("Level / Experience")
plt.ylabel("Salary")
plt.show()


#Random Forest Regression Graph
plt.figure()
plt.scatter(X, y, color='blue')
plt.plot(X_grid, rfr_reg_model.predict(X_grid))
plt.title("Random Forest Regression")
plt.xlabel("Level / Experience")
plt.ylabel("Salary")
plt.show()




