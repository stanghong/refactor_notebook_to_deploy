
import pandas as pd
from sklearn.feature_selection import mutual_info_classif, SelectKBest
from sklearn.metrics import classification_report, confusion_matrix

# determine the mutual information
def multual_info(X_train, y_train):
    mutual_info = mutual_info_classif(X_train, y_train.values.ravel())

    mutual_info = pd.Series(mutual_info)
    mutual_info.index = X_train.columns
    mutual_info.sort_values(ascending=False)

    sel_seven_cols = SelectKBest(mutual_info_classif, k=7)
    res=sel_seven_cols.fit(X_train, y_train.values.ravel())

    return res.columns


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

def NoShow_indicator(x):
    if x=='No': return 0
    else: return 1

def ConvGender(x):
    if x=='M': return 1
    else: return 0

def ConvNoshow(x):
    if x=='Yes': return 1
    else: return 0

    # feature engineering
def feature_enrichment(input_df):
    Appt_No_show = input_df['No-show'].apply(lambda x: NoShow_indicator(x))
    Appt_No_show.index = input_df['AppointmentDay']

    # create feature of week day
    # final=pd.DataFrame()
    input_df['Appt_weekday'] = input_df['AppointmentDay'].dt.day_name()
    input_df['Scheduled_weekday'] = input_df['ScheduledDay'].dt.day_name()

    # difference of the appointment-schedule day
    input_df['ScheduledDay'] = pd.to_datetime(input_df.ScheduledDay)
    input_df['AppointmentDay'] = pd.to_datetime(input_df.AppointmentDay)
    input_df['time_diff_days'] = abs(input_df['AppointmentDay'] - input_df['ScheduledDay']).dt.days

    input_df['GenderC'] = input_df['Gender'].apply(ConvGender)
    input_df.drop(['Gender'], axis=1, inplace=True)

    input_df['appt_wd'] = input_df['AppointmentDay'].dt.weekday
    input_df['schedule_wd'] = input_df['ScheduledDay'].dt.weekday

    return input_df

def convert_time_data(input_df):
    '''
        convert time into timestamp
        :param input_df:
        :return:
    '''
    input_df.ScheduledDay = pd.to_datetime(input_df.ScheduledDay)
    input_df.AppointmentDay = pd.to_datetime(input_df.AppointmentDay)
    return input_df

def noshow_to_indicator(input_df):
    input_df['NoShow']=input_df['No-show'].apply(ConvNoshow)
    input_df.drop(['No-show'], axis=1, inplace=True)
    return input_df