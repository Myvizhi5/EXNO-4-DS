# EXNO:4-DS
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
<img width="1677" height="473" alt="322338017-b544c435-1cc1-4bc6-83c9-de2945348808" src="https://github.com/user-attachments/assets/175af992-d970-4b05-9f3c-552ba6ea3da6" />

```
data.isnull().sum()
```
<img width="220" height="317" alt="322338037-40b1ab98-5a1a-41a1-b943-102b7c4cabed" src="https://github.com/user-attachments/assets/be1c5e6b-c1d7-456a-94c8-c129cd702a0e" />

```
missing=data[data.isnull().any(axis=1)]
missing
```
<img width="1647" height="500" alt="322338066-a5fe88ab-c993-4c97-b249-cffea5a21a54" src="https://github.com/user-attachments/assets/ae6d403b-babd-4cb8-976d-a2ec31d8615d" />

```
data2=data.dropna(axis=0)
data2
```
<img width="1677" height="480" alt="322338086-40a10680-63a6-4f18-87ae-517ceda76ca9" src="https://github.com/user-attachments/assets/e8fb8c8f-db2a-4bd9-ab1f-99e028f15197" />

```
sal=data["SalStat"]
data2["SalStat"]=data["SalStat"].map({' less than or equal to 50,000':0,' greater than 50,000':1})
print(data2['SalStat'])
```
<img width="653" height="337" alt="322338114-e59ce957-1bdc-4455-97a5-15d66108b864" src="https://github.com/user-attachments/assets/804a95a9-9d32-4b2a-8a5c-a7c8636cb835" />

```
sal2=data2['SalStat']
dfs=pd.concat([sal,sal2],axis=1)
dfs
```
<img width="388" height="482" alt="322338170-f8435063-835b-4eba-af2e-c46c67ea55e9" src="https://github.com/user-attachments/assets/709ad550-7086-4d7f-8249-c417b78d8961" />


```
data2
```
<img width="1540" height="486" alt="322338187-c034e83a-8e21-400e-bc40-103e3da86d0e" src="https://github.com/user-attachments/assets/14f7e9f5-588b-42f8-b9cd-5f5c12f919bb" />

```
new_data=pd.get_dummies(data2, drop_first=True)
new_data
```
<img width="1710" height="536" alt="322338213-f21819e3-a5bd-47e6-b1b7-9bc08b64bed9" src="https://github.com/user-attachments/assets/99c24042-6071-4373-a029-b4b588242ced" />

```
columns_list=list(new_data.columns)
print(columns_list)
```
<img width="1552" height="41" alt="322338249-8af6f5ce-4d99-4ed6-9371-730aeaa5a56b" src="https://github.com/user-attachments/assets/2bd66072-e379-48b7-91fc-8585d0ebabf2" />

```
features=list(set(columns_list)-set(['SalStat']))
print(features)
```
<img width="1552" height="41" alt="322338261-5f31a677-7d30-417a-8044-d5db741cafbf" src="https://github.com/user-attachments/assets/90b99317-feec-4177-a132-28c7d828c412" />

```
y=new_data['SalStat'].values
print(y)
```
<img width="175" height="41" alt="322338286-f4c779af-4c87-449e-9daa-be5d8d275212" src="https://github.com/user-attachments/assets/1c955c5f-eb6a-410d-9844-87ba3d597dff" />

```
x=new_data[features].values
print(x)
```
<img width="408" height="173" alt="322338321-4154db03-4c87-4b98-a13b-964f19bee9b0" src="https://github.com/user-attachments/assets/cb8ca277-b287-4797-b6f5-60d0c2afa1a3" />

```
train_x,test_x,train_y,test_y=train_test_split(x,y,test_size=0.3,random_state=0)
KNN_classifier=KNeighborsClassifier(n_neighbors = 5)
KNN_classifier.fit(train_x,train_y)
```
<img width="222" height="72" alt="322338343-e5e02520-eb39-436c-ac2e-e43048c1d672" src="https://github.com/user-attachments/assets/effb022f-b640-46a5-abff-cb9160a3622a" />

```
prediction=KNN_classifier.predict(test_x)
confusionMatrix=confusion_matrix(test_y, prediction)
print(confusionMatrix)
```
<img width="152" height="67" alt="322338371-a6eedfe3-aedd-4500-958f-6faafd54f464" src="https://github.com/user-attachments/assets/c3fd1552-2109-4af7-9a67-214829ca85da" />

```
accuracy_score=accuracy_score(test_y,prediction)
print(accuracy_score)
```
<img width="200" height="45" alt="322338387-0e56ff41-2f35-4d01-b479-53547391567b" src="https://github.com/user-attachments/assets/e133f6b9-cf01-4221-a08b-09356a3d810b" />

```
print("Misclassified Samples : %d" % (test_y !=prediction).sum())
```
<img width="305" height="38" alt="322338405-4af5ed3f-362a-40c6-a438-c89f31584e51" src="https://github.com/user-attachments/assets/06731664-5707-48f0-84d7-a2b81f9ca46a" />

```
data.shape
```
<img width="113" height="41" alt="322338420-1986f990-26e6-4b42-acfc-b2a6e52f8042" src="https://github.com/user-attachments/assets/18bb3bb3-d842-4797-9e6c-a967ce1f8bbc" />

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
<img width="350" height="67" alt="322338454-20777b0d-3cdb-4ae9-80e4-1f76ed093191" src="https://github.com/user-attachments/assets/65cf61a8-c2d0-449e-a60a-4a5df19e1ad7" />

```
import pandas as pd
import numpy as np
from scipy.stats import chi2_contingency
import seaborn as sns
tips=sns.load_dataset('tips')
tips.head()
```
<img width="547" height="258" alt="322338479-6d6f7ff2-b1da-4568-9cd1-cb6fa9553cd6" src="https://github.com/user-attachments/assets/a31797fc-23cf-4a5d-b2b5-8c74d724f8d3" />

```
tips.time.unique()
```
<img width="422" height="62" alt="322338497-f77bc757-8a31-4a5d-be15-5a447e6549c6" src="https://github.com/user-attachments/assets/5a2d2686-8333-4348-a43c-23ab61a98de1" />

```
contingency_table=pd.crosstab(tips['sex'],tips['time'])
print(contingency_table)
```
<img width="237" height="108" alt="322338518-06365e9f-f51b-4cf6-ab04-8a136726a025" src="https://github.com/user-attachments/assets/f706dad2-424f-402a-ba59-75ff440c9bcb" />

```
chi2,p,_,_=chi2_contingency(contingency_table)
print(f"Chi-Square Statistics: {chi2}")
print(f"P-Value: {p}")
```
<img width="403" height="67" alt="322338543-6adc4da7-421c-458f-9ec6-f6158aa6f731" src="https://github.com/user-attachments/assets/70328267-8824-40d9-a1fe-927b8abbaecf" />



# RESULT:
Thus, to read the given data and perform Feature Scaling and Feature Selection process and save the
data to a file is excuted successfully.
