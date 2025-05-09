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
From these results the best inverse penalizing factor to use is 21.5 Next we will draw a comparison between the lasso logistic regression. However, the best penalizing factor did not make all the factors zero.  While Lasso regression serves as a powerful tool, it does not always work as intended. It is unable to differentiate betweena _true_ casual variable and a variable that has little relationship with our outcome variable. In general we would like to reduce the model complexity. 

##### Logistic Lasso Regression v. Logistic Regression
We will make a direct comparision between the logistic lasso regression where the penalizing factor has been optimized and logistic regression where the predictor variables have been naively selected based on the correlation matrix. 

## Generalized Linear Model (Logit)

## Random Forest

## K-Nearest Nieghbor

## XGBoost 

## Verdict 

## Concluding Remarks
