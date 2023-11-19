#!/usr/bin/env python
# coding: utf-8

# ## Can we predict Stroke?
# ### DSC-680
# ### Debasish Panda 
# 

# ##### Step1 - Importing model data
# 
# ##### Step2 - I have cleaned my data set and dummified categorical variables.
# 
# ##### Step3 - Modeling - As my target variable is a binary values (either stroke even it true or not), I am going to perform classification based model. I will be using knn model for finding the outcome.
# 
# #### Steps to be performed Part of Modeling:
# 
# ###### 1. split the data into train/test (75/25) with stroke as the target variable. I will be dropping ID varoable from consideration as the feature has no bearing
# ###### 2. applying knn classification model to predict the outcome, as the target variable is a classification
# ###### 3. scale the data using standard scalar, create a pipe with knn and then apply grid search using n_neighbors
# ###### 4. calculate the accuracy/precision/recall and f1 score along with a confusion matrix of result set.
# 
# ##### The reslt is that there is a 90%+ accuracy, but model could not detect true positives due to train set imbalance.
# 
# ##### Step4 - Corrected the imbalances in train by using SMOTE To rebalance underbalanced class
# 
# ##### Step5 - Retrained the model using balanced data set, this resulted in a little reduction in Accuracy, but improved identification of True positives.

# In[1]:


#Import libraries
get_ipython().run_line_magic('matplotlib', 'inline')
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


#Import data set
stroke = pd.read_csv("healthcare-dataset-stroke-data.csv")
stroke


# In[3]:


stroke.info()


# In[4]:


#Checking for Missing values
stroke.isna().sum()


# In[5]:


#setting  missing values in bmi column to median value
stroke['bmi'] = stroke['bmi'].fillna(stroke['bmi'].median())
stroke.isna().sum()


# In[6]:


#basic infromation of all the data
stroke.describe()


# In[7]:


#basic infromation of the observaions which have stroke
stroke[stroke['stroke']==1].describe()


# In[8]:


#basic infromation of the observaions which have stroke
stroke[stroke['stroke']==0].describe()


# Important information from the tables above:
# 4.87% of the observation in this dataset had stroke.
# There is a big difference between those who have stroke and those who don't have stroke! In those who have a stroke - the average age and average glucose level is significantly higher, the number of people with heart disease and hypertension is significantly higher.

# In[9]:


#numeric attributes histograms
atttibutes_hist = stroke[["age", "avg_glucose_level", "bmi"]].hist(bins=20, figsize=(20,15))
atttibutes_hist


# In[10]:


#categorical attributes histograms (as pie charts)
fig, ax = plt.subplots(4,2, figsize = (12,12))
((ax1, ax2), (ax3, ax4), (ax5, ax6), (ax7, ax8)) = ax

labels = stroke['gender'].value_counts().index.tolist()[:2]
values = stroke['gender'].value_counts().tolist()[:2]
ax1.pie(x=values, labels=labels, autopct="%1.2f%%", shadow=True, explode=[0, 0.05])
ax1.set_title("Gender Distribution Pie Chart", fontdict={'fontsize': 14})

labels = ["History of No hypertension", "History of hypertension"]
values = stroke['hypertension'].value_counts().tolist()
ax2.pie(x=values, labels=labels, autopct="%1.2f%%", shadow=True, explode=[0, 0.2])
ax2.set_title("Hypertension Distribution Pie Chart", fontdict={'fontsize': 14})

labels = ["History of No heart disease", "History of heart disease"]
values = stroke['heart_disease'].value_counts().tolist()
ax3.pie(x=values, labels=labels, autopct="%1.2f%%", shadow=True, explode=[0, 0.2])
ax3.set_title("Heart disease Distribution Pie Chart", fontdict={'fontsize': 14})

labels = ["married", "never married"]
values = stroke['ever_married'].value_counts().tolist()
ax4.pie(x=values, labels=labels, autopct="%1.2f%%", shadow=True, explode=[0, 0.05])
ax4.set_title("Marriage Distribution Pie Chart", fontdict={'fontsize': 14})

labels = ["Private Job", "Self-employed", "Children", "Government Job", "Never Worked Before"]
values = stroke['work_type'].value_counts().tolist()
ax5.pie(x=values, labels=labels, autopct="%1.2f%%", shadow=True, explode=[0.1, 0.1, 0.1, 0.1, 0.2])
ax5.set_title("Work Type Pie Chart", fontdict={'fontsize': 14})

labels = ["Urban Residence", "Rural Residence"]
values = stroke['Residence_type'].value_counts().tolist()
ax6.pie(x=values, labels=labels, autopct="%1.2f%%", shadow=True, explode=[0, 0.05])
ax6.set_title("Residence Type Pie Chart", fontdict={'fontsize': 14})

labels = ["Never Smoked Before", "Unknown", "Smoked in the past", "Currently Smokes"]
values = stroke['smoking_status'].value_counts().tolist()
ax7.pie(x=values, labels=labels, autopct="%1.2f%%", shadow=True, explode=[0.03, 0.03, 0.03, 0.03])
ax7.set_title("Smoking Status Pie Chart", fontdict={'fontsize': 14})

labels = ["Didn't have Stroke", "Had Stroke"]
values = stroke['stroke'].value_counts().tolist()
ax8.pie(x=values, labels=labels, autopct="%1.2f%%", shadow=True, explode=[0, 0.2])
ax8.set_title("Stroke Pie Chart", fontdict={'fontsize': 14})

plt.tight_layout()
plt.show()


# In[11]:


#validating type values and counts for gender attribute
print(stroke['gender'].value_counts())


# Quick review shows there is an outlier - we will remove this outlier

# In[12]:


stroke = stroke[stroke['gender'] != "Other"]
stroke['gender'].value_counts()


# In[13]:


#Creating list of columns that have categorical data
catCols = [col for col in stroke.columns if stroke[col].dtype=="O"]
catCols


# In[14]:


#creating dummy variables for all categorical columns
from sklearn.preprocessing import LabelEncoder # loading library
label_encoder = LabelEncoder() # setting encoder function
class MultiColumnLabelEncoder:
    def __init__(self,columns = None):
        self.columns = columns # array of column names to encode

    def fit(self,X,y=None):
        return self # not relevant here

    def transform(self,X):
        '''
        Transforms columns of X specified in self.columns using
        LabelEncoder(). If no columns specified, transforms all
        columns in X.
        '''
        output = X.copy()
        if self.columns is not None:
            for col in self.columns:
                output[col] = LabelEncoder().fit_transform(output[col])
        else:
            for colname,col in output.iteritems():
                output[colname] = LabelEncoder().fit_transform(col)
        return output
    def fit_transform(self,X,y=None):
        return self.fit(X,y).transform(X)
strokeCat = MultiColumnLabelEncoder(columns = catCols).fit_transform(stroke)


# In[15]:


#valdidating categorical value conversion
strokeCat.head


# In[16]:


#Correlation Matrix
strokeCat.corr()


# In[17]:


#Correlation against stroke outcome
corr_matrix = strokeCat.corr()
corr_matrix["stroke"].sort_values(ascending = False)


# In[18]:


#Correlation heat map against numeric attributes
fig, ax = plt.subplots(figsize=(10,8))
sns.heatmap(strokeCat[['stroke', 'age', 'avg_glucose_level', 'bmi']].corr(),annot=True)


# In[19]:


#Correlation heat map against categorical attributes
fig, ax = plt.subplots(2,2, figsize = (12,12))
((ax1, ax2), (ax3, ax4)) = ax

# the "no_" attributes is the opposite to the "yes_" attributes so the correlation to stroke will be the same but negative.
sns.heatmap(strokeCat[['stroke', 'hypertension', 'heart_disease']].corr(),annot=True, ax=ax1)
sns.heatmap(strokeCat[['stroke', 'gender', 'ever_married', 'Residence_type']].corr(),annot=True, ax=ax2)
sns.heatmap(strokeCat[['stroke', 'work_type']].corr(),annot=True, ax=ax3)
sns.heatmap(strokeCat[['stroke', 'smoking_status']].corr(),annot=True, ax=ax4)

plt.tight_layout()
plt.show()


# In[20]:


#Stroke plot against age and avg glucose level
strokeClean = strokeCat

fig, ax = plt.subplots(1,2, figsize = (14,5), )
((ax1, ax2)) = ax

strokeClean.plot(ax=ax1, kind='scatter', x='age', y='stroke', alpha = 0.2)
strokeClean.plot(ax=ax2, kind='scatter', x='avg_glucose_level', y='stroke', alpha = 0.2)

plt.tight_layout()
plt.show()


# In[21]:


#Stroke correlation with hyptertention
fig, ax = plt.subplots(1,2, figsize = (12,12))
((ax1, ax2)) = ax

labels = ["deosn't have stroke", "have stroke"]
values = strokeClean[strokeClean['hypertension']==1]['stroke'].value_counts().tolist()
ax1.pie(x=values, labels=labels, autopct="%1.2f%%", shadow=True)
ax1.set_title("stroke ratio - there is hypertention", fontdict={'fontsize': 14})

labels = ["deosn't have stroke", "have stroke"]
values = strokeClean[strokeClean['hypertension']==0]['stroke'].value_counts().tolist()
ax2.pie(x=values, labels=labels, autopct="%1.2f%%", shadow=True)
ax2.set_title("stroke ratio - there isn't hypertention", fontdict={'fontsize': 14})

plt.tight_layout()
plt.show()


# In[22]:


#Stroke Correlation with Heart disease
fig, ax = plt.subplots(1,2, figsize = (12,12))
((ax1, ax2)) = ax

labels = ["deosn't have stroke", "have stroke"]
values = strokeClean[strokeClean['heart_disease']==1]['stroke'].value_counts().tolist()
ax1.pie(x=values, labels=labels, autopct="%1.2f%%", shadow=True)
ax1.set_title("stroke ratio - there is heart disease", fontdict={'fontsize': 14})

labels = ["deosn't have stroke", "have stroke"]
values = strokeClean[strokeClean['heart_disease']==0]['stroke'].value_counts().tolist()
ax2.pie(x=values, labels=labels, autopct="%1.2f%%", shadow=True)
ax2.set_title("stroke ratio - there isn't heart disease", fontdict={'fontsize': 14})

plt.tight_layout()
plt.show()


# In[23]:


#Stroke Correlation with Marriage
fig, ax = plt.subplots(1,2, figsize = (12,12))
((ax1, ax2)) = ax

labels = ["deosn't have stroke", "have stroke"]
values = strokeClean[strokeClean['ever_married']==1]['stroke'].value_counts().tolist()
ax1.pie(x=values, labels=labels, autopct="%1.2f%%", shadow=True)
ax1.set_title("stroke ratio - married", fontdict={'fontsize': 14})

labels = ["deosn't have stroke", "have stroke"]
values = strokeClean[strokeClean['ever_married']==0]['stroke'].value_counts().tolist()
ax2.pie(x=values, labels=labels, autopct="%1.2f%%", shadow=True)
ax2.set_title("stroke ratio - not married", fontdict={'fontsize': 14})

plt.tight_layout()
plt.show()


# In[24]:


#BMI Correalation
corr_matrix = strokeClean.corr()
bmi_corr = corr_matrix["bmi"].sort_values(ascending = False).drop('bmi')
print(bmi_corr[bmi_corr>0.15])
print(bmi_corr[bmi_corr<-0.15])


# In[25]:


#BMI vs Stroke scatter plot
strokeClean.plot.scatter( x='bmi', y='stroke', alpha = 0.05, title="stroke by bmi")


# In[26]:


# check that the pattern above is realy exist and not because the plot density:

values_30plusminusBMI = strokeClean[(strokeClean['bmi']>27) & (strokeClean['bmi']<33)]['stroke'].value_counts().tolist()
values_stroke = strokeClean['stroke'].value_counts().tolist()

print("-+30bmi without stroke cases : all wothiut stroke cases (ratio) = " + str(values_30plusminusBMI[0]/values_stroke[0]))
print("-+30bmi : all observations (ratio) = " + str(sum(values_30plusminusBMI)/sum(values_stroke)))
print("-+30bmi with stroke cases : all stroke cases (ratio) = " + str(values_30plusminusBMI[1]/values_stroke[1]))
print("as we can see, among 1/2 of the stroke cases the bmi is around 30. In contrast to cases where there is no stroke where the ratio is significantly lower,only 1/3")


# In[27]:


#Scatter plot of bmi vs avg_glucose_level and bmi vs age
fig, ax = plt.subplots(1,2, figsize = (14,5))
((ax1, ax2)) = ax
#stroke by combination of bmi and avg_glucose_level
strokeClean[strokeClean['stroke'] ==0].plot.scatter(ax=ax1, x='bmi', y='avg_glucose_level', alpha = 0.2, c='gray', label='no stroke')
strokeClean[strokeClean['stroke'] ==1].plot.scatter(ax=ax1, x='bmi', y='avg_glucose_level', alpha = 0.8, c='orange', label='stroke')
ax1.legend()
ax1.set_title('stroke by combination of bmi and avg_glucose_level')
#stroke by combination of bmi and age
strokeClean[strokeClean['stroke'] ==0].plot.scatter(ax=ax2, x='bmi', y='age', alpha = 0.3, c='gray', label='no stroke')
strokeClean[strokeClean['stroke'] ==1].plot.scatter(ax=ax2, x='bmi', y='age', alpha = 0.6, c='orange', label='stroke')
ax2.legend()
ax2.set_title('stroke by combination of bmi and age')

plt.tight_layout()
plt.show()


# #### Preparing Data set for Modeling.

# I am going to Split my data set into train and test data sets(75/25) and apply knn modelling with standard Scalar and hyper parameters for grid search

# In[28]:


strokeClean.info()


# In[29]:


#import Necessary libraries
from sklearn.model_selection import train_test_split


# In[30]:


#stroke Target value is taken as a numpy array
y = strokeClean["stroke"].values
#All the features are separated from our target value or label and stored in x
X = strokeClean.drop(["stroke","id"],axis=1)


# In[31]:


#Split data into training and testing sets - 75/25 Train and test size
X_train, X_test, y_train, y_test = train_test_split(X,y ,random_state=142, train_size=0.75, test_size=0.25)


# In[32]:


#printing Size and shape of the target and features for test and train
print(X_train.shape)
print(y_train.size)
print(X_test.shape)
print(y_test.size)


# In[33]:


train_df = pd.DataFrame(y_train,columns=['Stroke'])

fig, ax = plt.subplots(1,1, figsize = (12,12))
labels = ["deosn't have stroke", "have stroke"]
values_train = train_df.value_counts().tolist()

ax.pie(x=values_train, labels=labels, autopct="%1.2f%%", shadow=True)
ax.set_title("stroke ratio in train data:", fontdict={'fontsize': 15})
plt.show()
print("Values for No and Yes Stroke: " +str(values_train))


# In[34]:


test_df = pd.DataFrame(y_test,columns=['Stroke'])

fig, ax = plt.subplots(1,1, figsize = (12,12))
labels = ["deosn't have stroke", "have stroke"]
values_test = test_df.value_counts().tolist()

ax.pie(x=values_test, labels=labels, autopct="%1.2f%%", shadow=True)
ax.set_title("stroke ratio in test data:", fontdict={'fontsize': 15})
plt.show()
print("Values for No and Yes Stroke: " +str(values_test))


# KNN modeling with Original and training data

# In[35]:


# Load libraries
from sklearn.neighbors import KNeighborsClassifier 
from sklearn import datasets
from sklearn.preprocessing import StandardScaler 
from sklearn.pipeline import Pipeline, FeatureUnion 
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler, PolynomialFeatures, OneHotEncoder,StandardScaler,PowerTransformer, MinMaxScaler, RobustScaler


# In[36]:


#MinMax Scalacr
scaler = StandardScaler()
# Create a KNN classifier
knn = KNeighborsClassifier(n_neighbors=5, n_jobs=-1) 
# Create a pipeline
pipe = Pipeline([("scaler", scaler),("knn", knn)])


# In[37]:


#Scaling Test and Train Data feature set using MinMax Scaler
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# In[38]:


# Create space of candidate values
search_space = [{"knn__n_neighbors": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]}]


# In[39]:


# Create grid search with fitting the train data
grid = GridSearchCV(pipe, search_space, cv=5, verbose=0).fit(X_train_scaled, y_train)


# In[40]:


# Best neighborhood size (k)
print("Best K value: %.0f"  % (grid.best_estimator_.get_params()["knn__n_neighbors"]))


# In[41]:


#Grid Search prediction on test set
grid_pred = grid.predict(X_test_scaled)


# In[42]:


#Checking Accuracy Score
import math
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay


# In[43]:


#Calculating metrics
accuracy=accuracy_score(y_test, grid_pred)
precision = precision_score(y_test, grid_pred, average='weighted')
recall=recall_score(y_test, grid_pred, average='weighted')
f1 = f1_score(y_test, grid_pred, average='weighted')
print("Accuracy: %.4f"  % (accuracy))
print("precision: %.4f"  % (precision))
print("recall: %.4f"  % (recall))
print("f1 Score: %.4f"  % (f1))
print("Confusion Matrix for Prediction:")
cm=confusion_matrix(y_test, grid_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)

disp.plot(cmap=plt.cm.Blues)
plt.show()


# #### Observations

# My target variable for the modelling is a classification, which is essentially to predict based on the feature values, if the patient is going to have a stroke event or not.
# 
# Part of my modelling, I have imputed missing values and cleaned a outlier with gender variable after that I have created a knn model with standard Scalar and hyper parameter searc using 10 n_neighbors.
# 
# 
# I see that knn model has resulted in very high accuracy/precision/recall and f1 scores - all of which are in 90%s.
# 
# 
# This high accuracy could be a result of imbalanced dataset (95% negative outcomes, and 5% positive outcomes of stroke).

# As a pathforward I am going to use SMOTE to oversample my unbalanced postive outcome

# In[46]:


get_ipython().system('pip install imblearn')
from imblearn.over_sampling import SMOTE # loading Library
#Over Sampling data using SMOTE
oversample = SMOTE()
XUp, yUp = oversample.fit_resample(X_train_scaled, y_train)
upsampled_df = pd.DataFrame(yUp,columns=['Stroke'])

fig, ax = plt.subplots(1,1, figsize = (12,12))
labels = ["deosn't have stroke", "have stroke"]
values_upsample = upsampled_df.value_counts().tolist()

ax.pie(x=values_upsample, labels=labels, autopct="%1.2f%%", shadow=True)
ax.set_title("stroke ratio in train data - after over sampeling:", fontdict={'fontsize': 15})
plt.show()
print("there are now equal number of cases with stroke and without: " +str(values_upsample))


# ##### Performing few Models to see individual accuracies

# In[47]:


from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import BernoulliNB
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier


# In[48]:


from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score, ConfusionMatrixDisplay, precision_score, recall_score, f1_score, classification_report, roc_curve, plot_roc_curve, auc, precision_recall_curve, plot_precision_recall_curve, average_precision_score
from sklearn.model_selection import cross_val_score


# In[49]:


models = []
models.append(['Logistic Regreesion', LogisticRegression(random_state=0)])
models.append(['SVM', SVC(random_state=0)])
models.append(['KNeighbors', KNeighborsClassifier()])
models.append(['GaussianNB', GaussianNB()])
models.append(['BernoulliNB', BernoulliNB()])
models.append(['Decision Tree', DecisionTreeClassifier(random_state=0)])
models.append(['Random Forest', RandomForestClassifier(random_state=0)])
x_train_res=XUp
y_train_res=yUp
x_test=X_test_scaled


lst_1= []

for m in range(len(models)):
    lst_2= []
    model = models[m][1]
    model.fit(x_train_res, y_train_res)
    y_pred = model.predict(x_test)
    cm = confusion_matrix(y_test, y_pred)  #Confusion Matrix
    accuracies = cross_val_score(estimator = model, X = x_train_res, y = y_train_res, cv = 10)   #K-Fold Validation
    roc = roc_auc_score(y_test, y_pred)  #ROC AUC Score
    precision = precision_score(y_test, y_pred,average='weighted')  #Precision Score
    recall = recall_score(y_test, y_pred,average='weighted')  #Recall Score
    f1 = f1_score(y_test, y_pred,average='weighted')  #F1 Score
    print(models[m][0],':')
    print(cm)
    print('Accuracy Score: ',accuracy_score(y_test, y_pred))
    print('')
    print("K-Fold Validation Mean Accuracy: {:.2f} %".format(accuracies.mean()*100))
    print('')
    print("Standard Deviation: {:.2f} %".format(accuracies.std()*100))
    print('')
    print('ROC AUC Score: {:.2f}'.format(roc))
    print('')
    print('Precision: {:.2f}'.format(precision))
    print('')
    print('Recall: {:.2f}'.format(recall))
    print('')
    print('F1: {:.2f}'.format(f1))
    print('-----------------------------------')
    print('')
    lst_2.append(models[m][0])
    lst_2.append((accuracy_score(y_test, y_pred))*100) 
    lst_2.append(accuracies.mean()*100)
    lst_2.append(accuracies.std()*100)
    lst_2.append(roc)
    lst_2.append(precision)
    lst_2.append(recall)
    lst_2.append(f1)
    lst_1.append(lst_2)


# Rerunning same KNN Grid Searh Pipe with Train data augmented to balance classes using SMOTE

# In[50]:


# Create grid search with fitting the train data
gridUp = GridSearchCV(pipe, search_space, cv=5, verbose=0).fit(XUp, yUp)


# In[51]:


# Best neighborhood size (k)
print("Best K value: %.0f"  % (gridUp.best_estimator_.get_params()["knn__n_neighbors"]))


# In[52]:


#Grid Search prediction on test set
grid_predUp = gridUp.predict(X_test_scaled)


# In[53]:


#Calculating metrics
accuracy=accuracy_score(y_test, grid_predUp)
precision = precision_score(y_test, grid_predUp, average='weighted')
recall=recall_score(y_test, grid_predUp, average='weighted')
f1 = f1_score(y_test, grid_predUp, average='weighted')
print("Accuracy: %.4f"  % (accuracy))
print("precision: %.4f"  % (precision))
print("recall: %.4f"  % (recall))
print("f1 Score: %.4f"  % (f1))
print("Confusion Matrix for Prediction:")
cm1=confusion_matrix(y_test, grid_predUp)
disp1 = ConfusionMatrixDisplay(confusion_matrix=cm1)

disp1.plot(cmap=plt.cm.Blues)
plt.show()


# #### Observation

# After realizing my model is not identifying any positive outcome, I have used SMOTE to balance my training set to include more values for positive outcome.
# 
# 
# Several of the individual models like Logistics regression to Random Forest classifier had lesser precision than original KNN Model.
# 
# 
# This retraining with updated train data resulted in a slight reduction of accuracy and other measures compared to original data based KNN modelling. But have successfully classified the outcomes that are postive.
# 
# 
# I see that knn model has resulted in very high accuracy/precision/recall and f1 scores - all of which are in high 80s
# This is a slight drop in the metrics from previous iteration, however this is a better model as the data properly identifies all target classes.
# 

# In[ ]:




