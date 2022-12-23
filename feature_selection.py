
import pandas as pd
from sklearn.feature_selection import mutual_info_classif, SelectKBest

# determine the mutual information
def multual_info(X_train, y_train):
    mutual_info = mutual_info_classif(X_train, y_train.values.ravel())

    mutual_info = pd.Series(mutual_info)
    mutual_info.index = X_train.columns
    mutual_info.sort_values(ascending=False)

    sel_seven_cols = SelectKBest(mutual_info_classif, k=7)
    res=sel_seven_cols.fit(X_train, y_train.values.ravel())

    return res.columns


"""#### Conclustion: Comparing with Cycle 1; the new features could improve the prediction. It looks promising... """