# %% [markdown]
# # What is brain tumor?
# A brain tumor is a growth of abnormal cells in the brain. The anatomy of the brain is very 
# complex, with different parts responsible for different nervous system. Brain Tumor can 
# develop in any part of the brain or skull, including its protective lining, the underside of the brain(skull base), the brainstem, the sinuses and the nasal cavity, and many other areas.
# 
# Brain tumors can be cancerous(malignant) or noncancerous(benign).When benign or malignant 
# tumors grow, they can cause the pressure inside skull to increase. This can cause brain damage, and it can be life-threating.
# 
# However, Early detection and classification of brain tumors is an important research domain in the field of medical imaging and accordingly helps in selecting the most convenient treatment method to save patient life.
# 
# ![800wm.jpg](attachment:3e14e405-9a2e-48f4-94bf-11a876ec8a40.jpg)

# %% [markdown]
# # Import Libraries

# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# %% [markdown]
# # Load Dataset

# %%
df= pd.read_csv('./input/Brain Tumor.csv')

# %%
df.head()

# %%
df.tail()

# %%
df.info()

# %%
df.isnull().sum()

# %% [markdown]
# # Drop Unnamed Column

# %%
df.drop(columns=['Unnamed: 0'],axis=1,inplace=True)

# %%
df.columns

# %% [markdown]
# # Check Unique Values in data

# %%
df['y'].unique()

# %% [markdown]
# # Statistical Information of data

# %%
df.describe()

# %%
X=df.iloc[:,:-1].values
y=df.iloc[:,-1].values

# %% [markdown]
# # Label Encoding 

# %%
from sklearn import preprocessing
label_encoder = preprocessing.LabelEncoder()
df['y']= label_encoder.fit_transform(df['y'])

# %%
df['y'].unique()

# %% [markdown]
# # Split datset into training and testing data

# %%
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=.2,random_state=42)

# %% [markdown]
# # Training data

# %%
X_train

# %%
y_train

# %% [markdown]
# # Testing data

# %%
X_test

# %%
y_test

# %% [markdown]
# # Transformation of data

# %%
from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
X_train=sc.fit_transform(X_train)
X_test=sc.transform(X_test)

# %% [markdown]
# # Shape of X_train and y_train

# %%
print(X_train.shape)
print(y_train.shape)

# %% [markdown]
# # Machine Learning(Classification)
# 
# - Logistic Regression
# - KNN(KNearest Neighbors)
# - Random Forest Classifier
# - Decision Tree Classifier

# %% [markdown]
# # Logistic Regression

# %%
# build the model
from sklearn.linear_model import LogisticRegression
# Fit LogisticRegression
log_reg = LogisticRegression()
log_reg.fit(X_train,y_train)

# %%
#Prediction
y_predict = log_reg.predict(X_test)
y_predict

# %%
## Accuracy
score = log_reg.score(X_test, y_test)
print(score)

# %%
from sklearn import metrics
cnf_matrix = metrics.confusion_matrix(y_test, y_predict)
cnf_matrix

# %%
from sklearn.metrics import classification_report
print(classification_report(y_test, y_predict))

# %%
# Plotting of Confusion Matrics of Logistic Regression
from sklearn.metrics import confusion_matrix
pred_list = [log_reg]

for i in pred_list:
    print("Score : ",i.score(X_test,y_test))
    y_pred = i.predict(X_test)
    sns.heatmap(confusion_matrix(y_test,y_pred),annot = True)
    plt.xlabel("Y_pred")
    plt.ylabel("Y_test")
    plt.title(i)
    plt.show()

# %% [markdown]
# # KNearest Neighbors

# %%
# Build the model
from sklearn.neighbors import KNeighborsClassifier
# Fit KNN Classifier
KNN = KNeighborsClassifier(n_neighbors=3, metric='minkowski', p=2)
KNN.fit(X_train,y_train)

# %%
# Prediction
y_predict = KNN.predict(X_test)
y_predict

# %%
# Accuracy Score
score = KNN.score(X_test, y_test)
print(score)

# %%
# Confusion Metrix
from sklearn import metrics
cnf_matrix = metrics.confusion_matrix(y_test, y_predict)
cnf_matrix

# %%
# Classification Report
from sklearn.metrics import classification_report
print(classification_report(y_test, y_predict))

# %%
# Plotting of Confusion Matrix of KNN
from sklearn.metrics import confusion_matrix
pred_list = [KNN]

for i in pred_list:
    print("Score : ",i.score(X_test,y_test))
    y_pred = i.predict(X_test)
    sns.heatmap(confusion_matrix(y_test,y_pred),annot = True)
    plt.xlabel("Y_pred")
    plt.ylabel("Y_test")
    plt.title(i)
    plt.show()

# %% [markdown]
# # RandomForestClassifier

# %%
# Build Model
from sklearn.ensemble import RandomForestClassifier
# Fit RandomForest Clsassifier
RFC = RandomForestClassifier(n_estimators=20, random_state=0)
RFC.fit(X_train,y_train)

# %%
# Prediction
y_predict = RFC.predict(X_test)
y_predict

# %%
# Accuracy Score
score = RFC.score(X_test, y_test)
print(score)

# %%
# Plotting of confusion matrix of Random Forest Classifier
from sklearn.metrics import confusion_matrix
pred_list = [RFC]

for i in pred_list:
    print("Score : ",i.score(X_test,y_test))
    y_pred = i.predict(X_test)
    sns.heatmap(confusion_matrix(y_test,y_pred),annot = True)
    plt.xlabel("Y_pred")
    plt.ylabel("Y_test")
    plt.title(i)
    plt.show()

# %% [markdown]
# # DecisionTreeClassifier

# %%
# Build the model
from sklearn.tree import DecisionTreeClassifier
# Fit DecisionTree Classifier
DTC = DecisionTreeClassifier()
DTC.fit(X_train,y_train)

# %%
#prediction
y_predict = DTC.predict(X_test)
y_predict

# %%
# Accuracy Score
score = DTC.score(X_test, y_test)
print(score)

# %%
# Plotting of Confusion Matrix of Decision Tree Classifier
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
pred_list = [DTC]

for i in pred_list:
    print("Score : ",i.score(X_test,y_test))
    y_pred = i.predict(X_test)
    sns.heatmap(confusion_matrix(y_test,y_pred),annot = True)
    plt.xlabel("Y_pred")
    plt.ylabel("Y_test")
    plt.title(i)
    plt.show()

# %% [markdown]
# **If you liked this Notebook, Please do upvote.**
# 
# **If you have any suggestion or questions, feel free to comment!**
# 
# **Best Wishes!**


