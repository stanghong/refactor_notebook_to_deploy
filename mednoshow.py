"""MedNoShow_Improve_Cycle4.ipynb
"""
import pandas as pd
from sklearn.model_selection import train_test_split,cross_val_score, GridSearchCV

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
import os
import json
import pickle
from feature_engineering import *

# import configuration parameters
module_path = os.path.dirname(os.path.realpath(__file__))
config_path = os.path.join(module_path, "config.json")
with open(config_path) as f:
    file_contents = f.read()
    # print(file_contents)
    model_config = json.loads(file_contents)

training_data_s3_path = model_config["training_data_path"]
model_input_features = model_config["model_input_features"]

model_params = {
    'random_forest': {
        'model': RandomForestClassifier(),
        'params': {
            'n_estimators': [100],
            'max_features': ['auto'],
            'min_samples_leaf': [2]

        }
    },
    'logistic_regression': {
        'model': LogisticRegression(solver='liblinear', multi_class='auto'),
        'params': {
            'C': [1, 5, 10]
        }
    },
    'naive_bayes_gaussian': {
        'model': GaussianNB(),
        'params': {}
    },
    'decision_tree': {
        'model': DecisionTreeClassifier(),
        'params': {
            'criterion': ['gini', 'entropy'],

        }
    }
}

target=['NoShow']

if __name__ == "__main__":
    input_df=pd.read_csv(training_data_s3_path)

    convert_time_data(input_df)

    feature_enrichment(input_df)

    noshow_to_indicator(input_df)

    ### model training
    input_df=input_df[model_input_features + target]
    X=input_df[model_input_features]
    y= input_df[target]
    ### Split Training and Testing sets"""
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

    """### Feature Importance Evaluation and Selection"""

    #feature selection decision in feature_engineering.py
    # model_input_features= multual_info(X_train, y_train)

    rfc = RandomForestClassifier()
    rfc.fit(X_train,y_train.values.ravel())
    # predictions
    rfc_predict = rfc.predict(X_test)
    #cross validation scores
    rfc_cv_score = cross_val_score(rfc, X_train,y_train.values.ravel(), cv=10, scoring='roc_auc')
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

    #final select RF model
    rfcmodel = RandomForestClassifier(n_estimators=100, criterion='gini', min_samples_split=5,
                                      min_samples_leaf=2, max_features='sqrt', bootstrap=True, n_jobs=-1, random_state=42)
    rfcmodel.fit(X_train,y_train.values.ravel())
    y_pred_test = rfcmodel.predict(X_test)

    rfc_cv_score = cross_val_score(rfcmodel, X_train,y_train.values.ravel(), cv=5, scoring='roc_auc')

    print_test_results(y_test, y_pred_test, rfc_cv_score)


    """### prepare pickle dump file for app deployment"""
    file = open('random_forest_classification_model.pkl', 'wb')
    # dump information to that file
    pickle.dump(rfcmodel, file)



