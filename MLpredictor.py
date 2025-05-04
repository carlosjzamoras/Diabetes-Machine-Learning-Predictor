# import os 

# graphviz_path = "/opt/local/bin/"
# os.environ["PATH"] = graphviz_path + os.pathsep + os.environ["PATH"]
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import joblib
import statsmodels.api as sm
import statsmodels.formula.api as smf
# s


import xgboost as xgb 
import graphviz 
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.linear_model import Ridge, RidgeCV, Lasso, LassoCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, accuracy_score, confusion_matrix, precision_score, recall_score, ConfusionMatrixDisplay
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from scipy.stats import randint
from sklearn.tree import export_graphviz
from IPython.display import Image
from xgboost import XGBClassifier
from xgboost import plot_tree

from textwrap import wrap

#First step is to create EDA of variables 
#The goal is to predict Diabetes

#import data
health_data = pd.read_csv('/Users/carloszamora/Desktop/diabetes_012_health_indicators_BRFSS2015.csv')

# print(health_data.describe())
# #Creating a histogram of all the variables 
# health_data.hist(alpha=0.5, figsize=(20, 10))
# for ax in plt.gcf().axes:
#     ax.set_title(ax.get_title().replace('Histogram of',''))
#     ax.set_xlabel('Values')
#     ax.set_ylabel('Frequency')
# plt.tight_layout()
# plt.show()

# #Calculate Pearson Correlation Coeffcient 
# corr_matrix = health_data.corr()

# #plot the correlation coefficient
# plt.figure(figsize=(10,8))
# plt.matshow(corr_matrix,fignum = 1, cmap="seismic",vmin=-1,vmax=1)

# #add correlation coefficient numnber into cell 
# for i in range(len(corr_matrix.columns)):
#     for j in range(len(corr_matrix.columns)):
#         plt.text(j,i, f"{corr_matrix.iloc[i,j]:.2f}",
#                  ha='center',va='center',color='black',fontsize=8)
                   
# #Set tick positions and lables 
# tick_marks = np.arange(0,len(health_data.columns),1)
# x_lables= health_data.columns
# # x_lables = ['\n'.join(wrap(l,10)) for l in x_lables]
# plt.xticks(tick_marks,x_lables,rotation=90,ha='center')
# plt.yticks(tick_marks,health_data.columns)

# #add color bar gradient to represent Correlation Coefficient 
# color_bar = plt.colorbar()
# color_bar.set_label('Correlation Coefficient')

# #Add the title 
# plt.title("Correlation Matrix")
# plt.subplots_adjust(left=.5,top=.8,bottom=.3,right=.9)
# plt.show()

#Create a lasso regression 
#The first step in our lass regression is transforming our numerical 
#Combine PreDiabetes and Diabetes into one outcome

health_data['Diabetes_012'] = health_data['Diabetes_012'].replace({2:1})

# caterogical entries for Age,Education,Income, and General health 

health_data['Age'] = health_data['Age'].replace({1:'1-14',2:'18-24',3:"30-34",4:'35-39',5:'40-44',6:'45-49',7:'50-54',
                                                 8:'55-59',9:'60-64',10:'65-69',11:'70-74',12:'75-79',13:'80+'})
health_data['Education']= health_data['Education'].replace({1:'NeverAttended',2:'Elementary',3:'SomeHighSchool',4:'GED',5:"SomeCollege",6:"College"})
health_data['Income'] = health_data['Income'].replace({1:'10,0000',2:'15,0000',3:'20,000',4:'25,000',5:'35,000',6:'50,000',7:'75,000',8:'75,000+'})
health_data['GenHlth'] = health_data['GenHlth'].replace({1:'Excellent',2:'VeryGood',3:'Good',4:'Fair',5:'Poor'})

#Encode the catergorical features as "one-hot" numerica feature aka dummy variables 
dummies = pd.get_dummies(health_data[['Age','Education','Income','GenHlth']],dtype=int)
#dummies.info()
#print(dummies.head())

#Create Y-lable aka outcome variable
y = health_data['Diabetes_012']

# #Drop the outcome variable and our catergorical variables 
X_numerical = health_data.drop(['Diabetes_012','Age','Education','Income','GenHlth'],axis=1).astype('float64')

#Create a list of all numerical features
# list_numerical =X_numerical.columns
#print(list_numerical)
#Create all the predictor variables/features 
X = pd.concat([X_numerical,dummies],axis=1)

#Split data into trainting and test, 70/30 split

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=.03,random_state=10)

# Standardization of  our data. We want all numerical features to be centered around zero
# Ultimately we get a z score
# The only factors that are numerical aka non-binary are BMI,Mental Health, and Physical Health 

scaler = StandardScaler().fit(X_train[['BMI','MentHlth','PhysHlth']])
X_train[['BMI','MentHlth','PhysHlth']] =scaler.transform(X_train[['BMI','MentHlth','PhysHlth']])
X_test[['BMI','MentHlth','PhysHlth']] = scaler.transform(X_test[['BMI','MentHlth','PhysHlth']])

#Implement Lasso Regression 
from sklearn.linear_model import LogisticRegression
model = LogisticRegression(penalty='l1',solver='liblinear',C=0.001)
model.fit(X_train,y_train)
print("Coefficients:", model.coef_)
#Apply an alpha of 1
# reg = Lasso(alpha=0.01)
# reg = reg.fit(X_train,y_train)
# print(reg.coef_.tolist())
# print(X_train.columns)
#First look at Model Evalution 
# print('R squared training Set', round(reg.score(X_train,y_train)*100,2))
# print('R squated test set',round(reg.score(X_test,y_test)*100,2))
#To be removed 
# #MSE for training data and test data 
# pred_train=reg.predict(X_train)
# mse_train = mean_squared_error(y_train,pred_train)
# print('MSE training set',round(mse_train,2))
#  #test data
# pred = reg.predict(X_test)
# mse_test = mean_squared_error(y_test,pred)
# print("MSE test set",round(mse_test,2))

#Examin different alphas 
# alphas = []
# for i in range(500):
#     alphas.append(.001 + (i*.00015))
# lasso = Lasso(max_iter=10000)
# coefs = []
# for a in alphas:
#     lasso.set_params(alpha=a)
#     lasso.fit(X_train,y_train)
#     coefs.append(lasso.coef_)

# ax = plt.gca()

# ax.plot(alphas,coefs)
# ax.set_xscale('log')
# plt.axis('tight')
# plt.xlabel('alpha')
# plt.ylabel('Standardized Coefficients')
# plt.title('Lasso Coeffecients as a function of alpha')
# plt.show()

# #Choose the best alpha 
# #Lasso with 5 fold cross-validation 
# model = LassoCV(cv = 5, random_state=0,max_iter=10000)

# #fit model
# model.fit(X_train,y_train)
# print(model.alpha_)

# #Use the best value for final model:
# lasso_best = Lasso(alpha=model.alpha_)
# lasso_best.fit(X_train,y_train)
# #Show model coeffiecients
# print(list(zip(lasso_best.coef_,X)))

#Based on alpha = .01 we naively selected HighBP, HighChol, BMI, HeartDiease, MenHleath, PhysHealth, GenHealth

#Initial Test is a GLM with a binomial distribution
#Second we witll create a logistic regression 
#Compare the results

#Now that Lasso regression is finished 
#Continue onto the creation of the logist regression and binomial GLM

#Fist step is to divide the data into training and testing data 

# health_training, health_test = train_test_split(health_data,test_size=.03,random_state=10)

# # health_glm = smf.glm('Diabetes_012 ~HighBP + HighChol + BMI + HeartDiseaseorAttack + MentHlth + PhysHlth + GenHlth',
# #                       data=health_training, family = sm.families.Binomial()).fit()

# health_log = smf.logit('Diabetes_012 ~HighBP + HighChol + BMI + HeartDiseaseorAttack + MentHlth + PhysHlth + GenHlth',
#                        data=health_training).fit()

# # print(health_glm.summary())
# print(health_log.summary())

#Create a random forest with all facotrs
# X= health_data[['HighBP','HighChol','BMI','HeartDiseaseorAttack','MentHlth','PhysHlth','GenHlth']]
# X = health_data.drop('Diabetes_012',axis=1)
# y = health_data['Diabetes_012']
# X_train,X_test,y_train,y_test = train_test_split(X,y,test_size =.03)

# #Create an instance of the RandomForest 
# rf = RandomForestClassifier()
# rf.fit(X_train,y_train)

# y_pred = rf.predict(X_test)
# accuracy = accuracy_score(y_test,y_pred)
# print("Accuracy:", accuracy)

# #Show Single Decision Tree 
# for i in range (3):
#     tree = rf.estimators_[i]
#     dot_data = export_graphviz(tree,
#                                feature_names=X_train.columns,
#                                filled = True, 
#                                max_depth = 2, 
#                                impurity = False,
#                                proportion = True)
#     graph = graphviz.Source(dot_data)
#     display.graph()


#Return for later 
#Hyperparameter Tuning
#set the number of decisions trees and select the max depth

# param_dist = {'n_estimators': randint(50,500),
#               'max_depth' : randint(1,20)}

# rf = RandomForestClassifier()

# rand_search = RandomizedSearchCV(rf,
#                                  param_distributions= param_dist,
#                                  n_iter=5,
#                                  cv=5)

# #Fit the random search object to the data 
# rand_search.fit(X_train,y_train)

# #Finding the best model
# best_rf = rand_search.best_estimator_
# print('Best hyperparameters:', rand_search.best_params_)

# #K-nearest neighbor model 
# y_pred = best_rf.predict(X_test)
# cm = confusion_matrix(y_test,y_pred)
# ConfusionMatrixDisplay(confusion_matrix=cm).plot()
# plt.show()
# #More Metrics
# knn = KNeighborsClassifier(n_neighbors=5)
# knn.fit(X_train,y_train)
# y_pred = knn.predict(X_test)
# accuracy = accuracy_score(y_test,y_pred)
# precision = precision_score(y_test,y_pred)
# recall = recall_score(y_test,y_pred)

# print('Accuracy:', accuracy)
# print('Precision:', precision)
# print('Recall:', recall)

# # #Best Features
# feature_importance = pd.Series(best_rf.feature_importances_, index = X_train.columns).sort_values(ascending =False)

# #Plot a simple bar chart
# feature_importance.plot.bar()
# plt.show()


# #Next Model is Xg boost
# X = health_data.drop('Diabetes_012',axis=1)
# y = health_data['Diabetes_012']
# X_train,X_test,y_train,y_test = train_test_split(X,y,test_size =.03)

# #
# #Create the regression matrices 
# model = XGBClassifier(object = 'binary:logistic', random_state = 42, learning_state=32)
# model.fit(X_train,y_train)
# #Xgb will automatical assigned 1 to number greater the 0.5 
# prediction_train = model.predict(X_train)
# prediction_test = model.predict(X_test)

# #Extracting actual probabilities
# prob_prediction_train = model.predict_proba(X_train)
# prob_prediction_test = model.predict_proba(X_test)

# #Evaluating Model Performance
# #Calulate the accuracy
# accuracy = model.score(X_test,y_test)
# print("Accuracy: %f%%" % (accuracy * 100))

# #Calculate log-loss
# print(log_loss(y_test,prob_prediction_test))

# plot_tree(model)
# plt.show()

