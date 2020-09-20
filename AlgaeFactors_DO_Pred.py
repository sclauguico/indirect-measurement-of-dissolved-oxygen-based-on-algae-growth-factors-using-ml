# -*- coding: utf-8 -*-
"""
Created on Tue Jul 14 15:59:01 2020

@author: sclau
"""
# To import libraries
import numpy as np # for computation
import pandas as pd # for data manipulation
import matplotlib.pyplot as plt # for visualization

# To import dataset
dataset = pd.read_csv('Algae.csv')

# Preliminary analysis of dataset
# A. To know if there is missing data
dataset.isnull().sum().sort_values(ascending=False)

# To check column names and total records
dataset_count = dataset.count()

# C. To view the info about the dataset 
print(dataset.info())

# D. To view statistical salary of the dataset
statistics = dataset.describe() #one-way of checking for outliers

dataset.plot(kind='box', figsize=(8, 6))

plt.title('Box Plot of the Dataset on Algae Growth Factors')
plt.ylabel('Levels')

plt.show()

# To create the matrix of independent variable, x
X = dataset.iloc[:,0:3].values

# To create the matrix of dependent variable, y
Y = dataset.iloc[:,3:4].values

# To view the scatterplot of our dataset
import seaborn as sns
sns.pairplot(dataset)

# To determine the Pearson's Coefficient of Correlation for the whole dataset
dataset_correlation = dataset.corr()
##plt.figure(figsize=(3,3))
sns.heatmap(dataset_correlation, annot=True, linewidths=3)

# To split the whole dataset into training dataset and testing dataset
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size=0.2, random_state=0) #train_size=0.8, you can either still put this or not since test_size is already defined. By default, remaining is for training

# To perform feature scaling 
# A. For standardization feature scaling
from sklearn.preprocessing import StandardScaler # for not normally distributed samples
standard_scaler = StandardScaler ()
X_train_standard = X_train.copy()
X_test_standard = X_test.copy()
X_train_standard = standard_scaler.fit_transform(X_train_standard) # X_train_standard[:,3:5] -> for specifying features to be scaled (age and salary)
X_test_standard = standard_scaler.fit_transform(X_test_standard)  # X_test_standard[:,3:5] -> for specifying features to be scaled (age and salary)

# To fit the training dataset into a multiple linear regression model
from sklearn.linear_model import LinearRegression
mlr = LinearRegression()
mlr.fit(X_train_standard,Y_train)

# To predict the output of the testing dataset
Y_predict_mlr = mlr.predict(X_test_standard)

 plt.scatter


# To apply K-fold cross-validation for the simple linear regression model
from sklearn.model_selection import KFold
k_fold = KFold(n_splits=10, shuffle=True, random_state=0)

from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

from sklearn.preprocessing import StandardScaler # for not normally distributed samples
standard_scaler = StandardScaler ()
X_standard = X.copy()
X_standard = standard_scaler.fit_transform(X_standard)

# For the Mean Squared Error as scoring for for cross-validation 
MAE = (cross_val_score(estimator=mlr, X=X_standard, y=Y, cv=k_fold, scoring='neg_mean_absolute_error'))*-1

MAE_average = MAE.mean()
MAE_variance = MAE.std()

print('Mean Absolute Error of K-FOLDS:')
print (MAE)
print(' ')
print('Average Mean Absolute Error of K-FOLDS:')
print(MAE_average)
print(' ')
print('Mean Absolute Error Variance of K-FOLDS:')
print(MAE_variance)
print(' ')

# For the Mean Squared Error as scoring for for cross-validation 
MSE = (cross_val_score(estimator=mlr, X=X_standard, y=Y, cv=k_fold, scoring='neg_mean_squared_error'))*-1

MSE_average = MSE.mean()
MSE_variance = MSE.std()

print('Mean Squared Error of K-FOLDS:')
print (MSE)
print(' ')
print('Average Mean Squared Error of K-FOLDS:')
print(MSE_average)
print(' ')
print('Mean Squared Error Variance of K-FOLDS:')
print(MSE_variance)
print(' ')

# For the R Squared Error as scoring for for cross-validation 
R2 = (cross_val_score(estimator=mlr, X=X_standard, y=Y, cv=k_fold, scoring='r2'))

R2_average = R2.mean()
R2_variance = R2.std()

print('R Squared Error of K-FOLDS:')
print (R2)
print(' ')
print('Average R Squared Error of K-FOLDS:')
print(R2_average)
print(' ')
print('R Squared Error Variance of K-FOLDS:')
print(R2_variance)
print(' ')

# To evaluate the performance of the multiple linear regression model using holdout
# A. For the Mean Absolute Error (MAE)
from sklearn.metrics import mean_absolute_error
MAE_hold = mean_absolute_error(Y_test, Y_predict_mlr)
print('Mean Absolute Error: %.4f'
      % MAE_hold)
print(' ')

# B. For the Mean Squared Error (MSE)
from sklearn.metrics import mean_squared_error
MSE_hold = mean_squared_error(Y_test, Y_predict_mlr)
print('Mean Squared Error: %.4f'
      % MSE_hold)
print(' ')  

# C. For the Root Mean Squared Error (RMSE)/Deviation
from math import sqrt
RMSE_hold = sqrt(MSE_hold)
print('Root Mean Squared Error: %.4f'
      % RMSE_hold)
print(' ') 

# D. For the Explained Variance Score (EVS) -> ideal is 1
from sklearn.metrics import explained_variance_score
EVS_hold = explained_variance_score(Y_test, Y_predict_mlr)
print('Explained Variance Score: %.4f'
      % EVS_hold)
print(' ') 

# E. For the Coefficient of Determination Regression Score Function, R Squared Error (R2) -> ideal is 1
from sklearn.metrics import r2_score
R2_hold = r2_score(Y_test, Y_predict_mlr)
print('R2 Error: %.4f'
      % R2_hold)
print(' ')  

mlr.coef_

###############################################ANN################################################

# To fit the training dataset into a ANN regression model
from sklearn.neural_network import MLPRegressor
ann = MLPRegressor()
ann.fit(X_train_standard,Y_train)

# To predict the output of the testing dataset
Y_predict_ann = ann.predict(X_test_standard)

from sklearn.model_selection import KFold
k_fold = KFold(n_splits=10, shuffle=True, random_state=0)

from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

# For the Mean Squared Error as scoring for for cross-validation 
MAE = (cross_val_score(estimator=ann, X=X_standard, y=Y, cv=k_fold, scoring='neg_mean_absolute_error'))*-1

MAE_average = MAE.mean()
MAE_variance = MAE.std()

print('Mean Absolute Error of K-FOLDS:')
print (MAE)
print(' ')
print('Average Mean Absolute Error of K-FOLDS:')
print(MAE_average)
print(' ')
print('Mean Absolute Error Variance of K-FOLDS:')
print(MAE_variance)
print(' ')

# For the Mean Squared Error as scoring for for cross-validation 
MSE = (cross_val_score(estimator=ann, X=X_standard, y=Y, cv=k_fold, scoring='neg_mean_squared_error'))*-1

MSE_average = MSE.mean()
MSE_variance = MSE.std()

print('Mean Squared Error of K-FOLDS:')
print (MSE)
print(' ')
print('Average Mean Squared Error of K-FOLDS:')
print(MSE_average)
print(' ')
print('Mean Squared Error Variance of K-FOLDS:')
print(MSE_variance)
print(' ')

# For the R Squared Error as scoring for for cross-validation 
R2 = (cross_val_score(estimator=ann, X=X_standard, y=Y, cv=k_fold, scoring='r2'))

R2_average = R2.mean()
R2_variance = R2.std()

print('R Squared Error of K-FOLDS:')
print (R2)
print(' ')
print('Average R Squared Error of K-FOLDS:')
print(R2_average)
print(' ')
print('R Squared Error Variance of K-FOLDS:')
print(R2_variance)
print(' ')

# To evaluate the performance of the multiple linear regression model using holdout
# A. For the Mean Absolute Error (MAE)
from sklearn.metrics import mean_absolute_error
MAE_hold = mean_absolute_error(Y_test, Y_predict_ann)
print('Mean Absolute Error: %.4f'
      % MAE_hold)
print(' ')

# B. For the Mean Squared Error (MSE)
from sklearn.metrics import mean_squared_error
MSE_hold = mean_squared_error(Y_test, Y_predict_ann)
print('Mean Squared Error: %.4f'
      % MSE_hold)
print(' ')  

# C. For the Root Mean Squared Error (RMSE)/Deviation
from math import sqrt
RMSE_hold = sqrt(MSE_hold)
print('Root Mean Squared Error: %.4f'
      % RMSE_hold)
print(' ') 

# D. For the Explained Variance Score (EVS) -> ideal is 1
from sklearn.metrics import explained_variance_score
EVS_hold = explained_variance_score(Y_test, Y_predict_ann)
print('Explained Variance Score: %.4f'
      % EVS_hold)
print(' ') 

# E. For the Coefficient of Determination Regression Score Function, R Squared Error (R2) -> ideal is 1
from sklearn.metrics import r2_score
R2_hold = r2_score(Y_test, Y_predict_ann)
print('R2 Error: %.4f'
      % R2_hold)
print(' ')  

from ann_visualizer.visualize import ann_viz

ann_viz(ann, view=True, filename="network.gv", title= "Neural Network")

ann.layers

############################################### SVM R ################################################

# To fit the training dataset into a SVM regression model
from sklearn.svm import SVR
svr = SVR()
svr.fit(X_train_standard,Y_train)

# To predict the output of the testing dataset
Y_predict_svr = svr.predict(X_test_standard)
mglearn.plots.plot_svr

plt.scatter(X_test[:,0], Y_test, color = 'blue')
plt.scatter(X_test[:,0], svr.predict(X_test[:,1]), color = 'aquamarine')
plt.title('DO Level Prediction Using SVR')
plt.xlabel('pH Level')
plt.ylabel('DO Level')
plt.show()

plt.scatter(X_train[:, 0], X_train[:, 1], c=Y_train, s=50, cmap='autumn')
plot_svc_decision_function(svr)
plt.scatter(svr.support_vectors_[:, 0], svr.support_vectors_[:, 1],
            s=300, lw=1, facecolors='none');

from sklearn.model_selection import KFold
k_fold = KFold(n_splits=10, shuffle=True, random_state=0)

from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

# For the Mean Squared Error as scoring for for cross-validation 
MAE = (cross_val_score(estimator=svr, X=X_standard, y=Y, cv=k_fold, scoring='neg_mean_absolute_error'))*-1

MAE_average = MAE.mean()
MAE_variance = MAE.std()

print('Mean Absolute Error of K-FOLDS:')
print (MAE)
print(' ')
print('Average Mean Absolute Error of K-FOLDS:')
print(MAE_average)
print(' ')
print('Mean Absolute Error Variance of K-FOLDS:')
print(MAE_variance)
print(' ')

# For the Mean Squared Error as scoring for for cross-validation 
MSE = (cross_val_score(estimator=svr, X=X_standard, y=Y, cv=k_fold, scoring='neg_mean_squared_error'))*-1

MSE_average = MSE.mean()
MSE_variance = MSE.std()

print('Mean Squared Error of K-FOLDS:')
print (MSE)
print(' ')
print('Average Mean Squared Error of K-FOLDS:')
print(MSE_average)
print(' ')
print('Mean Squared Error Variance of K-FOLDS:')
print(MSE_variance)
print(' ')

# For the R Squared Error as scoring for for cross-validation 
R2 = (cross_val_score(estimator=svr, X=X_standard, y=Y, cv=k_fold, scoring='r2'))

R2_average = R2.mean()
R2_variance = R2.std()

print('R Squared Error of K-FOLDS:')
print (R2)
print(' ')
print('Average R Squared Error of K-FOLDS:')
print(R2_average)
print(' ')
print('R Squared Error Variance of K-FOLDS:')
print(R2_variance)
print(' ')

# To evaluate the performance of the multiple linear regression model using holdout
# A. For the Mean Absolute Error (MAE)
from sklearn.metrics import mean_absolute_error
MAE_hold = mean_absolute_error(Y_test, Y_predict_svr)
print('Mean Absolute Error: %.4f'
      % MAE_hold)
print(' ')

# B. For the Mean Squared Error (MSE)
from sklearn.metrics import mean_squared_error
MSE_hold = mean_squared_error(Y_test, Y_predict_svr)
print('Mean Squared Error: %.4f'
      % MSE_hold)
print(' ')  

# C. For the Root Mean Squared Error (RMSE)/Deviation
from math import sqrt
RMSE_hold = sqrt(MSE_hold)
print('Root Mean Squared Error: %.4f'
      % RMSE_hold)
print(' ') 

# D. For the Explained Variance Score (EVS) -> ideal is 1
from sklearn.metrics import explained_variance_score
EVS_hold = explained_variance_score(Y_test, Y_predict_svr)
print('Explained Variance Score: %.4f'
      % EVS_hold)
print(' ') 

# E. For the Coefficient of Determination Regression Score Function, R Squared Error (R2) -> ideal is 1
from sklearn.metrics import r2_score
R2_hold = r2_score(Y_test, Y_predict_svr)
print('R2 Error: %.4f'
      % R2_hold)
print(' ')  

############################################### KNN R ################################################

# To fit the training dataset into a KNN regression model
from sklearn.neighbors import KNeighborsRegressor
n_neighbors = 5
for i, weights in enumerate(['uniform', 'distance']):
    knn = KNeighborsRegressor(n_neighbors, weights=weights)
    Y_predict_knn=knn.fit(X_train_standard,Y_train).predict(X_test_standard)
    plt.subplot(2, 1, i + 1)
    plt.scatter(X_train[:,2], Y_train, color='darkorange', label='Actual')
    plt.plot(X_test[:,2], Y_predict_knn, color='navy', marker='x' label='Predicted')
    plt.axis('tight')
    plt.legend(["Actual", "Predicted"], loc=1)
    plt.title("KNeighborsRegressor (k = %i, weights = '%s')" % (n_neighbors,
                                                                weights))

plt.tight_layout()
plt.show()

fig, ax = plt.subplots(1, figsize=(12, 6))
plt.plot(X_test[:, 0], Y_test, marker='x', color='steelblue', linewidth=0)
plt.scatter(X_test[:, 0], Y_predict_gpr, marker='x', color='deeppink')
#plt.suptitle("$GaussianProcessRegressor(kernel=RBF)$ [default]", fontsize=20)
plt.title('DO Level Prediction Using GPR')
plt.xlabel('pH Level')
plt.ylabel('DO Level')
plt.legend(["Actual", "Predicted"], loc=1)
pass
for i, k in enumerate((1, 5, 10, 20)):
  # weights=distance - weight using distances
  knn = KNeighborsRegressor(k, weights='distance')

  # calculate y_test for all points in x_test
  Y_test = knn.fit(X_train_standard, Y_train).predict(X_test_standard)

  plt.subplot(2, 2, i + 1)

  plt.title("k = {}".format(k))

  plt.plot(X_train[:,0], Y_train, 'ro', X_test[:,0], Y_test, 'g');

plt.tight_layout()

# To predict the output of the testing dataset
Y_predict_knn = knn.predict(X_test_standard)

conda install mglearn
import mglearn 
import matplotlib.pyplot as plt
#mglearn.plots.plot_knn_regression(n_neighbors=5)
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt
import mglearn.plots

X_train, Y_train = mglearn.datasets.make_forge()
X_train, Y_train = make_blobs()
mglearn.discrete_scatter(X_train[:, 0], X_train[:, 1], Y_train)
plt.legend(["Class 0", "Class 1", "Class 2"], loc=1)
plt.xlabel("First feature")
plt.ylabel("Second feature")
#plt.plot(line, knn.predict(line))
print("X.shape: {}".format(X_train.shape))
#plt.show()
mglearn.plots.plot_knn_regression(n_neighbors=5)

fig, axes = plt.subplots(1, 3, figsize=(15, 4))
# create 1000 data points, evenly spaced between -3 and 3
line = np.linspace(-3, 3, 1000).reshape(-1, 1)
plt.suptitle("nearest_neighbor_regression")
for n_neighbors, ax in zip([1, 3, 9], axes):
 # make predictions using 1, 3 or 9 neighbors
 reg = KNeighborsRegressor(n_neighbors=n_neighbors).fit(X_train_standard, Y_train)
 ax.plot(X_train[:,2], Y_train, 'x')
 ax.plot(X_train[:,2], -3 * np.ones(len(X_train[:,2])), 'x')
 ax.plot(line, reg.predict(line))
 ax.set_title("%d neighbor(s)" % n_neighbors)

fig, axes = plt.subplots(1, 3, figsize=(10, 3))
for n_neighbors, ax in zip([1, 3, 9], axes):
    # the fit method returns the object self, so we can instantiate
    # and fit in one line
    clf = KNeighborsClassifier(n_neighbors=n_neighbors).fit(X, y)
    mglearn.plots.plot_2d_separator(clf, X, fill=True, eps=0.5, ax=ax, alpha=.4)
    mglearn.discrete_scatter(X[:, 0], X[:, 1], y, ax=ax)
    ax.set_title("{} neighbor(s)".format(n_neighbors))
    ax.set_xlabel("feature 0")
    ax.set_ylabel("feature 1")

axes[0].legend(loc=3)
plt.show()

mglearn.plots.plot_knn_classification(n_neighbors=5)
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
# create 1,000 data points, evenly spaced between -3 and 3
line = np.linspace(-3, 3, 1000).reshape(-1, 1)
for n_neighbors, ax in zip([1, 3, 9], axes):
    # make predictions using 1, 3, or 9 neighbors
    reg = KNeighborsRegressor(n_neighbors=n_neighbors)
    reg.fit(X_train_standard, Y_train)
    ax.plot(line, reg.predict(line))
    ax.plot(X_train, Y_train, '^', c=mglearn.cm2(0),   
             markersize=8)
    ax.plot(X_test_standard, Y_test, 'v', c=mglearn.cm2(1), markersize=8)
    ax.set_title("{} neighbor(s)\n train score: {:.2f} test  
              score: {:.2f}".format(n_neighbors,    
              reg.score(X_train_standard, Y_train),reg.score(X_test_standard, 
              Y_test)))
    ax.set_xlabel("Feature")
    ax.set_ylabel("Target")
axes[0].legend(["Model predictions", "Training data/target","Test   
    data/target"], loc="best")

from sklearn.model_selection import KFold
k_fold = KFold(n_splits=10, shuffle=True, random_state=0)

from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

# For the Mean Squared Error as scoring for for cross-validation 
MAE = (cross_val_score(estimator=knn, X=X_standard, y=Y, cv=k_fold, scoring='neg_mean_absolute_error'))*-1

MAE_average = MAE.mean()
MAE_variance = MAE.std()

print('Mean Absolute Error of K-FOLDS:')
print (MAE)
print(' ')
print('Average Mean Absolute Error of K-FOLDS:')
print(MAE_average)
print(' ')
print('Mean Absolute Error Variance of K-FOLDS:')
print(MAE_variance)
print(' ')

# For the Mean Squared Error as scoring for for cross-validation 
MSE = (cross_val_score(estimator=knn, X=X_standard, y=Y, cv=k_fold, scoring='neg_mean_squared_error'))*-1

MSE_average = MSE.mean()
MSE_variance = MSE.std()

print('Mean Squared Error of K-FOLDS:')
print (MSE)
print(' ')
print('Average Mean Squared Error of K-FOLDS:')
print(MSE_average)
print(' ')
print('Mean Squared Error Variance of K-FOLDS:')
print(MSE_variance)
print(' ')

# For the R Squared Error as scoring for for cross-validation 
R2 = (cross_val_score(estimator=knn, X=X_standard, y=Y, cv=k_fold, scoring='r2'))

R2_average = R2.mean()
R2_variance = R2.std()

print('R Squared Error of K-FOLDS:')
print (R2)
print(' ')
print('Average R Squared Error of K-FOLDS:')
print(R2_average)
print(' ')
print('R Squared Error Variance of K-FOLDS:')
print(R2_variance)
print(' ')

# To evaluate the performance of the multiple linear regression model using holdout
# A. For the Mean Absolute Error (MAE)
from sklearn.metrics import mean_absolute_error
MAE_hold = mean_absolute_error(Y_test, Y_predict_knn)
print('Mean Absolute Error: %.4f'
      % MAE_hold)
print(' ')

# B. For the Mean Squared Error (MSE)
from sklearn.metrics import mean_squared_error
MSE_hold = mean_squared_error(Y_test, Y_predict_knn)
print('Mean Squared Error: %.4f'
      % MSE_hold)
print(' ')  

# C. For the Root Mean Squared Error (RMSE)/Deviation
from math import sqrt
RMSE_hold = sqrt(MSE_hold)
print('Root Mean Squared Error: %.4f'
      % RMSE_hold)
print(' ') 

# D. For the Explained Variance Score (EVS) -> ideal is 1
from sklearn.metrics import explained_variance_score
EVS_hold = explained_variance_score(Y_test, Y_predict_knn)
print('Explained Variance Score: %.4f'
      % EVS_hold)
print(' ') 

# E. For the Coefficient of Determination Regression Score Function, R Squared Error (R2) -> ideal is 1
from sklearn.metrics import r2_score
R2_hold = r2_score(Y_test, Y_predict_knn)
print('R2 Error: %.4f'
      % R2_hold)
print(' ')  

############################################### GPR ################################################

# To fit the training dataset into a GPR model
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, RBF

# Define kernel parameters. 
l = 0.1
sigma_f = 2
sigma_n = 0.4

# Define kernel object. 
kernel = ConstantKernel(constant_value=sigma_f,constant_value_bounds=(1e-3, 1e3)) \
            * RBF(length_scale=l, length_scale_bounds=(1e-3, 1e3))

gpr = GaussianProcessRegressor(kernel=kernel, alpha=sigma_n**2, n_restarts_optimizer=10)
gpr.fit(X_train_standard,Y_train)

# To predict the output of the testing dataset
Y_predict_gpr = gpr.predict(X_test_standard)

fig, ax = plt.subplots(1, figsize=(12, 6))
plt.plot(X_test[:, 0], Y_test, marker='x', color='steelblue', linewidth=0)
plt.scatter(X_test[:, 0], Y_predict_gpr, marker='x', color='deeppink')
#plt.suptitle("$GaussianProcessRegressor(kernel=RBF)$ [default]", fontsize=20)
plt.title('DO Level Prediction Using GPR')
plt.xlabel('pH Level')
plt.ylabel('DO Level')
plt.legend(["Actual", "Predicted"], loc=1)
pass

%matplotlib inline

from gaussian_processes_util import plot_gp

# Finite number of points
#X = np.arange(-5, 5, 0.2).reshape(-1, 1)

# Mean and covariance of the prior
mu = np.zeros(X_train_standard.shape)
cov = kernel(X_train_standard, X_train_standard[:, 2])

# Draw three samples from the prior
samples = np.random.multivariate_normal(mu.ravel(), cov, 3)

# Plot GP mean, confidence interval and samples 
plot_gp(mu, cov, X_train_standard, samples=samples)

mu_s, cov_s = posterior_predictive(X, X_train_standard, Y_train, sigma_y=noise)

samples = np.random.multivariate_normal(mu_s.ravel(), cov_s, 3)
plot_gp(mu_s, cov_s, X, X_train_standard=X_train_standard, Y_train=Y_train, samples=samples)

def f(X_train_standard):
    f = np.sin((4*np.pi)*X_train_standard) + np.sin((7*np.pi)*X_train_standard)
    return(f)

f_x = f(X_train_standard)

fig, ax = plt.subplots()
# Plot "true" linear fit.
sns.lineplot(x=X_train_standard, y=f_x, color='red', label='f(x)', ax=ax)
# Plot prediction. 
sns.lineplot(x=X_train_standard, y=Y_predict_gpr, color='green', label='pred', ax=ax)
ax.set(title=f'Prediction GaussianProcessRegressor, sigma_f = {sigma_f} and l = {l}')
ax.legend(loc='upper right');




#y_predict_gpr, sigma = gpr.predict(X_test_standard, return_std=True)
#
#def f(X_test_standard):
#    """The function to predict."""
#    return X_test_standard * gpr.predict(X_test_standard)
#
#plt.figure()
#plt.plot(X_test_standard, f(X_test_standard), 'r:', label=r'$f(x) = x\,\sin(x)$')
#plt.plot(X_test_standard, Y_test, 'r.', markersize=10, label='Observations')
#plt.plot(X_test_standard, Y_predict_gpr, 'b-', label='Prediction')
#plt.fill(np.concatenate([X_test_standard, X_test_standard[::-1]]),
#         np.concatenate([Y_predict_gpr - 1.9600 * sigma,
#                        (Y_predict_gpr + 1.9600 * sigma)[::-1]]),
#         alpha=.5, fc='b', ec='None', label='95% confidence interval')
#plt.xlabel('$x$')
#plt.ylabel('$f(x)$')
#plt.ylim(-10, 20)
#plt.legend(loc='upper left')

from sklearn.model_selection import KFold
k_fold = KFold(n_splits=10, shuffle=True, random_state=0)

from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

# For the Mean Squared Error as scoring for for cross-validation 
MAE = (cross_val_score(estimator=gpr, X=X_standard, y=Y, cv=k_fold, scoring='neg_mean_absolute_error'))*-1

MAE_average = MAE.mean()
MAE_variance = MAE.std()

print('Mean Absolute Error of K-FOLDS:')
print (MAE)
print(' ')
print('Average Mean Absolute Error of K-FOLDS:')
print(MAE_average)
print(' ')
print('Mean Absolute Error Variance of K-FOLDS:')
print(MAE_variance)
print(' ')

# For the Mean Squared Error as scoring for for cross-validation 
MSE = (cross_val_score(estimator=gpr, X=X_standard, y=Y, cv=k_fold, scoring='neg_mean_squared_error'))*-1

MSE_average = MSE.mean()
MSE_variance = MSE.std()

print('Mean Squared Error of K-FOLDS:')
print (MSE)
print(' ')
print('Average Mean Squared Error of K-FOLDS:')
print(MSE_average)
print(' ')
print('Mean Squared Error Variance of K-FOLDS:')
print(MSE_variance)
print(' ')

# For the R Squared Error as scoring for for cross-validation 
R2 = (cross_val_score(estimator=gpr, X=X_standard, y=Y, cv=k_fold, scoring='r2'))

R2_average = R2.mean()
R2_variance = R2.std()

print('R Squared Error of K-FOLDS:')
print (R2)
print(' ')
print('Average R Squared Error of K-FOLDS:')
print(R2_average)
print(' ')
print('R Squared Error Variance of K-FOLDS:')
print(R2_variance)
print(' ')

# To evaluate the performance of the multiple linear regression model using holdout
# A. For the Mean Absolute Error (MAE)
from sklearn.metrics import mean_absolute_error
MAE_hold = mean_absolute_error(Y_test, Y_predict_gpr)
print('Mean Absolute Error: %.4f'
      % MAE_hold)
print(' ')

# B. For the Mean Squared Error (MSE)
from sklearn.metrics import mean_squared_error
MSE_hold = mean_squared_error(Y_test, Y_predict_gpr)
print('Mean Squared Error: %.4f'
      % MSE_hold)
print(' ')  

# C. For the Root Mean Squared Error (RMSE)/Deviation
from math import sqrt
RMSE_hold = sqrt(MSE_hold)
print('Root Mean Squared Error: %.4f'
      % RMSE_hold)
print(' ') 

# D. For the Explained Variance Score (EVS) -> ideal is 1
from sklearn.metrics import explained_variance_score
EVS_hold = explained_variance_score(Y_test, Y_predict_gpr)
print('Explained Variance Score: %.4f'
      % EVS_hold)
print(' ') 

# E. For the Coefficient of Determination Regression Score Function, R Squared Error (R2) -> ideal is 1
from sklearn.metrics import r2_score
R2_hold = r2_score(Y_test, Y_predict_gpr)
print('R2 Error: %.4f'
      % R2_hold)
print(' ')  
