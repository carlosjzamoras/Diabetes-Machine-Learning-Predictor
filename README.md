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


## Generalized Linear Model (Logit)

## Random Forest

## K-Nearest Nieghbor

## XGBoost 

## Verdict 

## Concluding Remarks
