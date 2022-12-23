"""MedNoShow_Improve_Cycle4.ipynb
"""

import pandas as pd

df=pd.read_csv('drive/MyDrive/2021_DS_Projects/DS_Project/Medical_No_Shows.csv')

"""### Time Series"""

#convert time into timestamp
df.ScheduledDay=pd.to_datetime(df.ScheduledDay)
df.AppointmentDay=pd.to_datetime(df.AppointmentDay)

def Noshowind(x):
    if x=='No': return 0
    else: return 1

Appt_No_show=df['No-show'].apply(lambda x:Noshowind(x))
Appt_No_show.index = df['AppointmentDay']

# create feature of week day
final=pd.DataFrame()
Appt_weekday = df['AppointmentDay'].dt.day_name()
Scheduled_weekday=df['ScheduledDay'].dt.day_name()

#prepare countplot
final = pd.DataFrame(list(zip(Appt_weekday, Scheduled_weekday,Appt_No_show)), columns =['appt_wd', 'schedule_wd','Noshow'])
final['no_show']=df[['No-show']]

# difference of the appointment-schedule day
ScheduledDay=pd.to_datetime(df.ScheduledDay)
AppointmentDay=pd.to_datetime(df.AppointmentDay)
df['time_diff_days']=abs(AppointmentDay-ScheduledDay).dt.days


"""### EDA Analysis"""

final_dataset=df[['PatientID','AppointmentID', 'Gender', 'ScheduledDay',
       'AppointmentDay', 'Age', 'LocationID', 'MedicaidIND', 'Hypertension',
       'Diabetes', 'Alcoholism', 'Disability', 'SMS_received', 'No-show',
       'time_diff_days']]

"""### EDA and Feature Engineering"""

def ConvNoshow(x):
    if x=='Yes': return 1
    else: return 0
final_dataset['NoShow']=final_dataset['No-show'].apply(ConvNoshow)
final_dataset.drop(['No-show'],axis=1,inplace=True)

def ConvGender(x):
    if x=='M': return 1
    else: return 0
final_dataset['GenderC']=final_dataset['Gender'].apply(ConvGender)
final_dataset.drop(['Gender'],axis=1,inplace=True)

final_dataset['appt_wd']= df['AppointmentDay'].dt.weekday
final_dataset['schedule_wd']=df['ScheduledDay'].dt.weekday
final_dataset['PatientIDLength']=df['PatientID'].apply(lambda x:len(x))

final=final_dataset.select_dtypes('int')

"""prepare for model training 
### Split Training and Testing sets"""
X=final.iloc[:,1:]
y=final.iloc[:,:1]

y_train.value_counts()

"""### Feature Importance Evaluation and Selection"""
### Logistic Regression

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

from sklearn.feature_selection import mutual_info_classif

# determine the mutual information
mutual_info = mutual_info_classif(X_train, y_train.values.ravel())

mutual_info = pd.Series(mutual_info)
mutual_info.index = X_train.columns
mutual_info.sort_values(ascending=False)

from sklearn.feature_selection import SelectKBest

sel_five_cols = SelectKBest(mutual_info_classif, k=7)
sel_five_cols.fit(X_train, y_train.values.ravel())
X_train.columns[sel_five_cols.get_support()]

"""#### Conclustion: Comparing with Cycle 1; the new features could improve the prediction. It looks promising... """

from sklearn.ensemble import RandomForestClassifier

rfc = RandomForestClassifier()
rfc.fit(X_train,y_train.values.ravel())
# predictions
rfc_predict = rfc.predict(X_test)

from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report, confusion_matrix

rfc_cv_score = cross_val_score(rfc, X_train,y_train.values.ravel(), cv=10, scoring='roc_auc')


print("=== Confusion Matrix ===")
print(confusion_matrix(y_test, rfc_predict))
print('\n')
print("=== Classification Report ===")
print(classification_report(y_test, rfc_predict))
print('\n')
print("=== All AUC Scores ===")
print(rfc_cv_score)
print('\n')
print("=== Mean AUC Score ===")
print("Mean AUC Score - Random Forest: ", rfc_cv_score.mean())

"""### Tuning HyperParameters for RandomForest"""

from IPython.display import Javascript
display(Javascript('IPython.notebook.execute_cells_below()'))

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier

model_params = {
    'random_forest': {
        'model': RandomForestClassifier(),
        'params' : {
        'n_estimators': [100], #
        'max_features' : ['auto'],
        'min_samples_leaf' : [2]

        }
    },
    'logistic_regression' : {
        'model': LogisticRegression(solver='liblinear',multi_class='auto'),
        'params': {
            'C': [1,5,10]
        }
    },
    'naive_bayes_gaussian': {
        'model': GaussianNB(),
        'params': {}
    },
    'decision_tree': {
        'model': DecisionTreeClassifier(),
        'params': {
            'criterion': ['gini','entropy'],
            
        }
    }     
}

from sklearn.model_selection import GridSearchCV
import pandas as pd
scores = []

for model_name, mp in model_params.items():
    clf =  GridSearchCV(mp['model'], mp['params'], cv=3, return_train_score=True)
    clf.fit(X_train, y_train.values.ravel())
    scores.append({
        'model': model_name,
        'best_score': clf.best_score_,
        'best_params': clf.best_params_
    })
    
df = pd.DataFrame(scores,columns=['model','best_score','best_params'])

rfcmodel = RandomForestClassifier(n_estimators=100, criterion='gini', min_samples_split=5, 
                                  min_samples_leaf=2, max_features='sqrt', bootstrap=True, n_jobs=-1, random_state=42)
rfcmodel.fit(X_train,y_train.values.ravel())
y_pred_test = rfcmodel.predict(X_test)

from sklearn.metrics import confusion_matrix
print(confusion_matrix(y_test, y_pred_test))
from sklearn import metrics
print(metrics.accuracy_score(y_test, y_pred_test))

rfc_cv_score = cross_val_score(rfcmodel, X_train,y_train.values.ravel(), cv=5, scoring='roc_auc')
print("=== Confusion Matrix ===")
print(confusion_matrix(y_test, y_pred_test))
print('\n')
print("=== Classification Report ===")
print(classification_report(y_test, y_pred_test))
print('\n')
print("=== All AUC Scores ===")
print(rfc_cv_score)
print('\n')
print("=== Mean AUC Score ===")
print("Mean AUC Score - Random Forest: ", rfc_cv_score.mean())

"""### prepare pickle dump file for app deployment"""

import pickle
# open a file, where you ant to store the data
file = open('random_forest_classification_model.pkl', 'wb')

# dump information to that file
pickle.dump(rfcmodel, file)



