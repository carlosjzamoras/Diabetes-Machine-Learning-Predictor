# Diabetes Machine Learning Predictor
Diabetes is defined asa chronic health condition that impacts how the body processes blood sugar. The most accurate way of detecting diabetes is doing a blood test that measures blood glucose levels. While millions of Americans may be diabetic or prediabetic, we want to provide a certain level of assurance before having patients pay for costly blood test.  

## Problem Statement
This study attempts to predict a patient's liklihood of having diabetes based on responses provided by the Behavioral Risk Factor Surveillance System. We apply known machine learning techniques to predict an individual's diabetes risk and tune our models accordingly. By the end of the study we should have a model that predicts diabetes within a reasonable amount of accuracy and precision. In addition, we will provide the key factors that are the best indicators of diabetes. 

## Data Processing 
The first step in our predictive processing is importating the data and loading it into Python. We will also import the necessary libraries at for this specific stage. Some of the libraries shown will also be used for the Exploratory Data Analysis (EDA) portion of the model creation. The data set is preprocessed and contains 22 variables. The variable of interest is the indicatory or binary variables _Diabtes_012_, stating $1$ is the individual has diabeates or not/ 

```python
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
import statsmodels.formula.api as smf
#First step is to create EDA of variables 
#The goal is to predict Diabetes

#import data
health_data = pd.read_csv('/Users/carloszamora/Desktop/diabetes_012_health_indicators_BRFSS2015.csv')
```
## Exploratory Data Analysis
An exploratory Data Analysis is a preliminary step in summarizing the data set. We first exam A historgram to view the distribution of each variable and some summary statistics of each variables. 
```python
print(health_data.describe())
#Creating a histogram of all the variables 
health_data.hist(alpha=0.5, figsize=(20, 10))
for ax in plt.gcf().axes:
    ax.set_title(ax.get_title().replace('Histogram of',''))
    ax.set_xlabel('Values')
    ax.set_ylabel('Frequency')
plt.tight_layout()
plt.show()
```
>output is shown below.
```
  Diabetes_012         HighBP       HighChol      CholCheck            BMI         Smoker  ...       PhysHlth       DiffWalk            Sex            Age      Education         Income
count  253680.000000  253680.000000  253680.000000  253680.000000  253680.000000  253680.000000  ...  253680.000000  253680.000000  253680.000000  253680.000000  253680.000000  253680.000000
mean        0.296921       0.429001       0.424121       0.962670      28.382364       0.443169  ...       4.242081       0.168224       0.440342       8.032119       5.050434       6.053875
std         0.698160       0.494934       0.494210       0.189571       6.608694       0.496761  ...       8.717951       0.374066       0.496429       3.054220       0.985774       2.071148
min         0.000000       0.000000       0.000000       0.000000      12.000000       0.000000  ...       0.000000       0.000000       0.000000       1.000000       1.000000       1.000000
25%         0.000000       0.000000       0.000000       1.000000      24.000000       0.000000  ...       0.000000       0.000000       0.000000       6.000000       4.000000       5.000000
50%         0.000000       0.000000       0.000000       1.000000      27.000000       0.000000  ...       0.000000       0.000000       0.000000       8.000000       5.000000       7.000000
75%         0.000000       1.000000       1.000000       1.000000      31.000000       1.000000  ...       3.000000       0.000000       1.000000      10.000000       6.000000       8.000000
max         2.000000       1.000000       1.000000       1.000000      98.000000       1.000000  ...      30.000000       1.000000       1.000000      13.000000       6.000000       8.000000
```
![Histogram of diabetes variables](https://github.com/user-attachments/assets/464fdf27-8c1c-46a9-b5d5-afebdac8b4d8)

We can immediately tell that there are many more patients without diabetes than with diabetes. Some of the predictor variables are sort of normally distributed while other have uneven date or are skewed. We continue forward and conduct a correlation matrix of the variables. This will provide us some insight into what variables are key predictors for diabetes and potentially include them in our models.

### Correlation Matrix 
A correlation matrix tells use the strength of the relationship between two variables. In our case, we are looking to see what variables have a strong correlation with Diabetes. While this is not the only metric we use in determining what predictor variables go into our model, it is a good starting point. 

```python
#Calculate Pearson Correlation Coeffcient 
corr_matrix = health_data.corr()

#plot the correlation coefficient
plt.figure(figsize=(10,8))
plt.matshow(corr_matrix,fignum = 1, cmap="seismic",vmin=-1,vmax=1)

#add correlation coefficient numnber into cell 
for i in range(len(corr_matrix.columns)):
    for j in range(len(corr_matrix.columns)):
        plt.text(j,i, f"{corr_matrix.iloc[i,j]:.2f}",
                 ha='center',va='center',color='black',fontsize=8)
                   
#Set tick positions and lables 
tick_marks = np.arange(0,len(health_data.columns),1)
x_lables= health_data.columns
# x_lables = ['\n'.join(wrap(l,10)) for l in x_lables]
plt.xticks(tick_marks,x_lables,rotation=90,ha='center')
plt.yticks(tick_marks,health_data.columns)

#add color bar gradient to represent Correlation Coefficient 
color_bar = plt.colorbar()
color_bar.set_label('Correlation Coefficient')

#Add the title 
plt.title("Correlation Matrix")
plt.subplots_adjust(left=.5,top=.8,bottom=.3,right=.9)
plt.show()
```
>output is shown below
![correlation matrix](https://github.com/user-attachments/assets/7b45a521-018e-4613-b5e7-97c7bbce9143)

Based on this matrix, we can see that the variables with the strongest correlections are:  
 - High Blood Pressure (HighBP)
 - High Cholesterol (HighChol)
 - BMI
 - Heart Diesease or Heart Attack (HeartDiseaseorAttack)
 - General Health
 - Physcial Health
 - Difficulty Walking (DiffWalk)
 - Age
 - Income

However some of these variables are correlated amongst themselves. In order to address overfitting, we may want to remove one or two variables that are highly correlated amongst themselves. 
### Addressing Multicollinearity 
Multicollinearity occurs when the predictor (independent) variables in our regressive model are correlated amongst themselves. As stated above we can simply drop variable that is highly correlated with another predictor variable, or we can take a more rigerous approach--we can conduct a lasso regression
#### Lasso Regression
Lasso regression with binary outcomes is a form of logistic regression tha includes L1 regulatization, or rather a penalty factor. The penalty factors tends to drive the coefficients to zero, effectively removing those variables from our model. We use lasso regression as a method of feature selection. One of the key differences here is that instead of an alpha with use $C$, the inverse of the regularization strength, the smaller the values the stronger the regularization. We will use this in conjunction with our coefficient matrix to create two logit models with different and compare. 
```python
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
```
>output is shown below
```
[[ 0.74583782  0.51861609  0.          0.31591258  0.          0.
   0.25887611 -0.11442267  0.         -0.06095793  0.          0.
   0.          0.          0.07223784  0.16713334  0.          0.
   0.          0.          0.          0.          0.          0.
   0.          0.          0.04017044  0.03958824  0.          0.
  -0.07425358  0.          0.          0.          0.          0.
   0.          0.          0.          0.          0.          0.
   0.         -0.238827   -0.87515712  0.13996607  0.          0.
  -0.59171136]]
```
From here we can tell when a higher regulization is applied that the key factors are _HighBP_, _HighChol_, _BMI_, _General Health_, _Heart Disease or Attack_, and _Physical Activity_. 

##### Choosing the best C (inverse of reguluzation strength)
We will perform a 5 fold cross validation technique in order to find out the best C value for out logistic lasso regression. 
```python
# #Choose the best C value
#Logistic Lasso with % fold cross-validation
clf = LogisticRegressionCV(
    Cs=10,
    cv =5,
    penalty='l1',
    solver='liblinear',
    scoring='f1',
    max_iter=1000,
    refit=True
)
clf.fit(X_test,y_test)
print("Best C:", clf.C_[0])
```
>output is shown below
```
Coefficients: [[ 0.67478267  0.55338964  1.15706162  0.38766644 -0.04102682  0.13277449
   0.23165537 -0.05224019 -0.02281784 -0.03454998 -0.67805137  0.03992066
   0.07212196 -0.01030522 -0.02885197  0.12546634  0.24695673 -2.38799138
  -2.13121892 -1.93928154 -1.4802937  -1.29990454 -1.07270692 -0.8712403
  -0.78795748 -0.56905507 -0.40881534 -0.33862375 -0.41643371 -0.58363294
  -0.83219381 -0.54971581 -0.77859807 -0.64649551 -0.73810157 -0.68652758
  -0.81720298 -0.852915   -0.89526448 -0.92305391 -0.98792574 -1.06405307
  -1.09870292 -1.25491822 -1.84298443 -0.0842998  -0.51336566  0.06035295
  -1.17006263]]
Best C: 21.54434690031882
```
From these results the best inverse penalizing factor to use is 21.5 Next we will draw a comparison between the lasso logistic regression and logistic regression. However, the best penalizing factor did not make all the factors zero.  While Lasso regression serves as a powerful tool, it does not always work as intended. It is unable to differentiate betweena _true_ casual variable and a variable that has little relationship with our outcome variable. In general we would like to reduce the model complexity. 

## Generalized Linear Model (Logit)
### Logistic Lasso Regression v. Logistic Regression
We will make a direct comparision between the logistic lasso regression where the penalizing factor has been optimized and logistic regression where the predictor variables have been naively selected based on the correlation matrix (listed aboved). 
```python

#Now that Lasso regression is finished 
#Continue comparing the naive logistic  regression (GLM w/ binomial) with the optimized 


# health_training, health_test = train_test_split(health_data,test_size=.03,random_state=10)
health_training, health_test = train_test_split(health_data_clean,test_size=.03,random_state=10)
health_glm = smf.glm('Diabetes_012 ~ HighBP + HighChol + BMI + HeartDiseaseorAttack + MentHlth + PhysHlth + GenHlth',
                      data=health_training, family = sm.families.Binomial()).fit()
log_prob_y=health_glm.predict(health_test)
log_pred_y=(log_prob_y >=0.5).astype(int)
```
Now we compare the results from the Lasso model and the Logistic regression model. 
```python
#Comparing Models 
print("Accuracy Score Lasso: ",accuracy_score(y_test,lasso_y_pred))
print("Acurracy Score Logistic: ",accuracy_score(health_test['Diabetes_012'],log_pred_y))
print("Precision Score Lasso: ",precision_score(y_test,lasso_y_pred))
print("Precision Score Logistic: ",precision_score(health_test['Diabetes_012'],log_pred_y))
print("Recall Score Lasso: ",recall_score(y_test,lasso_y_pred))
print("Recall Score Logistic: ",recall_score(health_test['Diabetes_012'],log_pred_y))
print("F1_score Lasso: ",f1_score(y_test,lasso_y_pred))
print("F1 Score Logistic: ",f1_score(health_test['Diabetes_012'],log_pred_y) )
```
>output is shown below
```
Accuracy Score Lasso:  0.8477204046774406
Acurracy Score Logistic:  0.8462751281040599
Precision Score Lasso:  0.5803814713896458
Precision Score Logistic:  0.573170731707317
Recall Score Lasso:  0.1748768472906404
Recall Score Logistic:  0.15435139573070608
F1_score Lasso:  0.26876971608832806
F1 Score Logistic:  0.24320827943078913
```
### ROC and AUC
The ROC is a popular tool used to evaluated bininary classifications.  We are plotting the true positive rate against the false positive rate at different threshholds. 
```python
#Compute ROC for both models
fpr1, tpr1, _ = roc_curve(y_test,lasso_y_pred)
fpr2, tpr2, _ = roc_curve(health_test['Diabetes_012'],log_pred_y)
#Compute the AUC for both models
auc_model1= roc_auc_score(y_test,lasso_y_pred)
auc_model2= roc_auc_score(health_test['Diabetes_012'],log_pred_y)
#Plot the figure
fig, ax = plt.subplots(1,2, figsize=(14,6))

# Plot ROC curve for Model 1
ax[0].plot(fpr1, tpr1, color='blue', label=f'Model 1 (AUC = {auc_model1:.2f})')
ax[0].plot([0, 1], [0, 1], linestyle='--', color='gray')  # Random chance line
ax[0].set_xlabel('False Positive Rate')
ax[0].set_ylabel('True Positive Rate')
ax[0].set_title('Model 1: ROC Curve')
ax[0].legend()

# Plot ROC curve for Model 2
ax[1].plot(fpr2, tpr2, color='red', label=f'Model 2 (AUC = {auc_model2:.2f})')
ax[1].plot([0, 1], [0, 1], linestyle='--', color='gray')  # Random chance line
ax[1].set_xlabel('False Positive Rate')
ax[1].set_ylabel('True Positive Rate')
ax[1].set_title('Model 2: ROC Curve')
ax[1].legend()

# Show the plot
plt.tight_layout()
plt.show()
```
>output is shown below


![ROC curve](https://github.com/user-attachments/assets/86784a2c-44f5-45c4-9ca9-fb2b737d1561)

#### Author's comments 
So our logistic model while doing well in the accuracy score , performs poorly in the precision score. We look into the AUC score and while the logistic lasso regression performs marginally  better , both models are not ideal. These models barely perform better than a random guess. An important takeaway are that some predictor variables are easily accessible without any intrusive medical testing. Items like BMI and Blood Pressure are able to act as preliminary indictors in our simple models. 

A key distinction is that our naive logistic regression does not inherently remove multicollinearity. The comparison is useful however is only including factors that may not be as personal. 
## Random Forest
Ww move forward to a more complex model called a random forest. A random forest is an extension of a decision tree. In practice 
```python
#Create a random forest with all factors
X = health_data.drop('Diabetes_012',axis=1)
y = health_data['Diabetes_012']
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size =.03)

#Create an instance of the RandomForest 
rf = RandomForestClassifier(class_weight='balanced')
rf.fit(X_train,y_train)

y_pred = rf.predict(X_test)
accuracy = accuracy_score(y_test,y_pred)
precision = precision_score(y_test,y_pred)
f1Score = f1_score(y_test,y_pred)
AUC_score = roc_auc_score(y_test,y_pred)
print("Accuracy:", accuracy)
print("Precision:",precision)
print('F1 Score: ', f1Score)
print('AUC Score: ',AUC_score )
```
>outcome is shown below

```
Accuracy: 0.8406254105899357
Precision: 0.4980694980694981
F1 Score:  0.2984384037015616
AUC Score:  0.586211034269199
```
While the accuracy score is slightly better the AUC score is slightly better, the model still performs poorly in distingushing between true positives and false positives. 
## K-Nearest Neighbor
Next we perform a K-nearest neighbor algorithm to classify a patient as diabetic or not. 
```python
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train,y_train)
y_pred = knn.predict(X_test)
accuracy = accuracy_score(y_test,y_pred)
precision = precision_score(y_test,y_pred)
recall = recall_score(y_test,y_pred)
AUC_score = roc_auc_score(y_test,y_pred)
print('Accuracy:', accuracy)
print('Precision:', precision)
print('Recall:', recall)
print('AUC_Score:', AUC_score)
```
## XGBoost 
We perform a gradient boosting algorithm and compare its results (return to a later date for hyperparemeter tuning) 
```python
#
#Create the regression matrices 
model = XGBClassifier(object = 'binary:logistic', random_state = 42, learning_state=32)
model.fit(X_train,y_train)
#Xgb will automatical assigned 1 to number greater the 0.5 
prediction_train = model.predict(X_train)
prediction_test = model.predict(X_test)

#Extracting actual probabilities
prob_prediction_train = model.predict_proba(X_train)
prob_prediction_test = model.predict_proba(X_test)

#Evaluating Model Performance
#Calulate the accuracy
accuracy = model.score(X_test,y_test)
print("Accuracy: %f%%" % (accuracy * 100))

#Calculate log-loss
print(log_loss(y_test,prob_prediction_test))

plot_tree(model)
plt.show()

```

## Verdict 
XGBoost performed the best due to the underlying nature of the algorithm being performed. Unlike logistic regression which assumes a linear relationship between the outcome variable and the predictor variables, XGBoost is an ensemble of decision trees that can properly model nonlinear interactions between the predictor and outcome variables. KNN performed decently well but also struggles with sparse data and has no built in feature selection. While our non gradient boosting models performed decently well, it is likely that there is a nonlinear relationship between some of the predictor variables and the binary classification of having diabetes or not. 

