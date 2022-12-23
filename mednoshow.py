"""MedNoShow_Improve_Cycle4.ipynb
"""
import pandas as pd
from sklearn.model_selection import train_test_split,cross_val_score, GridSearchCV

from sklearn.metrics import classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier

import pickle
from feature_selection import multual_info

input_df=pd.read_csv('Medical_No_Shows.csv')

def print_test_results(y_test, rfc_predict, rfc_cv_score):
    '''
        print out test results
        :param y_test: test truth
        :param rfc_predict: test prediction
        :param rfc_cv_score: random forest cv score
        :return: None
    '''

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

"""### Time Series"""

def convert_time_data(input_df):
    '''
        convert time into timestamp
        :param input_df: 
        :return: 
    '''
    input_df.ScheduledDay=pd.to_datetime(input_df.ScheduledDay)
    input_df.AppointmentDay=pd.to_datetime(input_df.AppointmentDay)
    return input_df

convert_time_data(input_df)

def NoShow_indicator(x):
    if x=='No': return 0
    else: return 1

def ConvGender(x):
    if x=='M': return 1
    else: return 0

def ConvNoshow(x):
    if x=='Yes': return 1
    else: return 0

#feature engineering
def feature_enrichment(input_df):
    Appt_No_show=input_df['No-show'].apply(lambda x:NoShow_indicator(x))
    Appt_No_show.index = input_df['AppointmentDay']

    # create feature of week day
    # final=pd.DataFrame()
    input_df['Appt_weekday'] = input_df['AppointmentDay'].dt.day_name()
    input_df['Scheduled_weekday']=input_df['ScheduledDay'].dt.day_name()


    # difference of the appointment-schedule day
    input_df['ScheduledDay']=pd.to_datetime(input_df.ScheduledDay)
    input_df['AppointmentDay']=pd.to_datetime(input_df.AppointmentDay)
    input_df['time_diff_days']=abs(input_df['AppointmentDay']-input_df['ScheduledDay']).dt.days

    input_df['GenderC'] = input_df['Gender'].apply(ConvGender)
    input_df.drop(['Gender'], axis=1, inplace=True)

    input_df['appt_wd'] = input_df['AppointmentDay'].dt.weekday
    input_df['schedule_wd'] = input_df['ScheduledDay'].dt.weekday

    return input_df

feature_enrichment(input_df)




def noshow_to_indicator(input_df):
    input_df['NoShow']=input_df['No-show'].apply(ConvNoshow)
    input_df.drop(['No-show'], axis=1, inplace=True)
    return input_df

noshow_to_indicator(input_df)


### model training

model_input_features=['Age', 'MedicaidIND', 'Hypertension',
       'Diabetes', 'Alcoholism', 'Disability', 'SMS_received',
       'time_diff_days', 'GenderC', 'appt_wd', 'schedule_wd' ]
target=['NoShow']

input_df=input_df[model_input_features + target]

"""prepare for model training 
### Split Training and Testing sets"""
X=input_df[model_input_features]
y=input_df[target]

"""### Feature Importance Evaluation and Selection"""

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

#feature selection decision in feature_selection.py
# model_input_features= multual_info(X_train, y_train)

rfc = RandomForestClassifier()
rfc.fit(X_train,y_train.values.ravel())
# predictions
rfc_predict = rfc.predict(X_test)

rfc_cv_score = cross_val_score(rfc, X_train,y_train.values.ravel(), cv=10, scoring='roc_auc')






model_params = {
    'random_forest': {
        'model': RandomForestClassifier(),
        'params' : {
        'n_estimators': [100],
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


scores = []

for model_name, mp in model_params.items():
    clf =  GridSearchCV(mp['model'], mp['params'], cv=3, return_train_score=True)
    clf.fit(X_train, y_train.values.ravel())
    scores.append({
        'model': model_name,
        'best_score': clf.best_score_,
        'best_params': clf.best_params_
    })
    
input_df = pd.DataFrame(scores, columns=['model', 'best_score', 'best_params'])

rfcmodel = RandomForestClassifier(n_estimators=100, criterion='gini', min_samples_split=5, 
                                  min_samples_leaf=2, max_features='sqrt', bootstrap=True, n_jobs=-1, random_state=42)
rfcmodel.fit(X_train,y_train.values.ravel())
y_pred_test = rfcmodel.predict(X_test)

rfc_cv_score = cross_val_score(rfcmodel, X_train,y_train.values.ravel(), cv=5, scoring='roc_auc')

print_test_results(y_test, y_pred_test, rfc_cv_score)


"""### prepare pickle dump file for app deployment"""


# open a file, where you ant to store the data
file = open('random_forest_classification_model.pkl', 'wb')

# dump information to that file
pickle.dump(rfcmodel, file)



