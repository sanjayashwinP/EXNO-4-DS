# EXNO:4-DS
### NAME:SANJAY ASHWIN P 
### REG NO:212223040181
# AIM:
To read the given data and perform Feature Scaling and Feature Selection process and save the
data to a file.

# ALGORITHM:
 STEP 1:Read the given Data.
 STEP 2:Clean the Data Set using Data Cleaning Process.
 STEP 3:Apply Feature Scaling for the feature in the data set.
 STEP 4:Apply Feature Selection for the feature in the data set.
 STEP 5:Save the data to the file.

# FEATURE SCALING:
1. Standard Scaler: It is also called Z-score normalization. It calculates the z-score of each value and replaces the value with the calculated Z-score. The features are then rescaled with x̄ =0 and σ=1
2. MinMaxScaler: It is also referred to as Normalization. The features are scaled between 0 and 1. Here, the mean value remains same as in Standardization, that is,0.
3. Maximum absolute scaling: Maximum absolute scaling scales the data to its maximum value; that is,it divides every observation by the maximum value of the variable.The result of the preceding transformation is a distribution in which the values vary approximately within the range of -1 to 1.
4. RobustScaler: RobustScaler transforms the feature vector by subtracting the median and then dividing by the interquartile range (75% value — 25% value).

# FEATURE SELECTION:
Feature selection is to find the best set of features that allows one to build useful models. Selecting the best features helps the model to perform well.
The feature selection techniques used are:
1.Filter Method
2.Wrapper Method
3.Embedded Method

# CODING AND OUTPUT:


```
import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
data=pd.read_csv("/content/income(1) (1).csv",na_values=[ " ?"])
data
```
![image](https://github.com/user-attachments/assets/ec9326a6-c886-45f7-86e1-c427bf8e028e)

```
data.isnull().sum()
```
![image](https://github.com/user-attachments/assets/08022717-def9-4fe4-a98f-86501d57e1ec)

```
missing=data[data.isnull().any(axis=1)]
missing
```
![image](https://github.com/user-attachments/assets/84b89e40-d58f-49d9-86e2-5ac1cf2e8cbd)

```
data2=data.dropna(axis=0)
data2
```
![image](https://github.com/user-attachments/assets/5042ce46-0ce0-4703-a028-a8d44820f424)

```
sal=data["SalStat"]
data2["SalStat"]=data["SalStat"].map({' less than or equal to 50,000':0,' greater than 50,000':1})
print(data2['SalStat'])
```
![image](https://github.com/user-attachments/assets/5953257c-ba17-45a9-b771-2551d72181ea)

```
sal2=data2['SalStat']
dfs=pd.concat([sal,sal2],axis=1)
dfs
```
![image](https://github.com/user-attachments/assets/ab34822b-c465-4653-aab3-84ce7921147f)

```
data2
```
![image](https://github.com/user-attachments/assets/ada46d64-980e-4d40-87c2-4be3b4590f49)

```
new_data=pd.get_dummies(data2, drop_first=True)
new_data
```
![image](https://github.com/user-attachments/assets/66cef839-f2d6-44d3-ad1c-f3dac392780c)

```
columns_list=list(new_data.columns)
print(columns_list)
```
![image](https://github.com/user-attachments/assets/f14e1f76-34bf-4025-b121-9f62a28fd9b3)

```
features=list(set(columns_list)-set(['SalStat']))
print(features)
```
![image](https://github.com/user-attachments/assets/619915f9-2700-40fa-a7a0-0ff3972b7c42)

```
y=new_data['SalStat'].values
print(y)
```
![image](https://github.com/user-attachments/assets/0432c067-e3a5-4643-bfea-adb415592001)

```
x=new_data[features].values
print(x)
```
![image](https://github.com/user-attachments/assets/7351dd8b-362d-4018-b9ee-3082b63c3e09)

```
train_x,test_x,train_y,test_y=train_test_split(x,y,test_size=0.3,random_state=0)
KNN_classifier=KNeighborsClassifier(n_neighbors = 5)
KNN_classifier.fit(train_x,train_y)
```
![image](https://github.com/user-attachments/assets/1158d06a-4a95-4e0a-8c42-020f2571e05b)

```
prediction=KNN_classifier.predict(test_x)
confusionMatrix=confusion_matrix(test_y, prediction)
print(confusionMatrix)
```
![image](https://github.com/user-attachments/assets/8bfa081b-7f34-4aef-ab21-7835e1f36aaf)

```
accuracy_score=accuracy_score(test_y,prediction)
print(accuracy_score)
```
![image](https://github.com/user-attachments/assets/078f429d-e72f-4e79-8945-701bd084a8fa)

```
print("Misclassified Samples : %d" % (test_y !=prediction).sum())
```
![image](https://github.com/user-attachments/assets/99c66fe6-e5f1-4dd5-848d-d9da3205fd9a)


```
data.shape
```
![image](https://github.com/user-attachments/assets/5721701e-ce9c-44e7-ad1b-16f8c35b601e)

```
import pandas as pd
from sklearn.feature_selection import SelectKBest, mutual_info_classif, f_classif
data={
    'Feature1': [1,2,3,4,5],
    'Feature2': ['A','B','C','A','B'],
    'Feature3': [0,1,1,0,1],
    'Target'  : [0,1,1,0,1]
}

df=pd.DataFrame(data)
x=df[['Feature1','Feature3']]
y=df[['Target']]
selector=SelectKBest(score_func=mutual_info_classif,k=1)
x_new=selector.fit_transform(x,y)
selected_feature_indices=selector.get_support(indices=True)
selected_features=x.columns[selected_feature_indices]
print("Selected Features:")
print(selected_features)
```
![image](https://github.com/user-attachments/assets/73624e45-890f-4611-990a-432348ae3dce)

```
import pandas as pd
import numpy as np
from scipy.stats import chi2_contingency
import seaborn as sns
tips=sns.load_dataset('tips')
tips.head()
```
![image](https://github.com/user-attachments/assets/5ba049cd-059f-440f-8639-475661838ec8)

```
tips.time.unique()
```
![image](https://github.com/user-attachments/assets/77665fa2-f8db-4bc3-8820-9004076247f6)

```
contingency_table=pd.crosstab(tips['sex'],tips['time'])
print(contingency_table)
```
![image](https://github.com/user-attachments/assets/beca5671-f3b0-4cd0-b575-e416867c8edc)

```
chi2,p,_,_=chi2_contingency(contingency_table)
print(f"Chi-Square Statistics: {chi2}")
print(f"P-Value: {p}")
```
![image](https://github.com/user-attachments/assets/e2b2dd13-e643-4ef9-a9a3-bf364a550dba)



# RESULT:
Thus, Feature selection and Feature scaling has been used on thegiven dataset.
