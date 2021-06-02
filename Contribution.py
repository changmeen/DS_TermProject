import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
pd.set_option('display.max_columns', None)
import warnings
warnings.filterwarnings('ignore')

from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

app = pd.read_csv("application_record.csv")
crecord = pd.read_csv("credit_record.csv")

app.drop('FLAG_MOBIL', axis=1, inplace=True)
app = app.drop_duplicates('ID', keep='last')
app.drop('OCCUPATION_TYPE', axis=1, inplace=True)

q_hi = app['CNT_CHILDREN'].quantile(0.999)
q_low = app['CNT_CHILDREN'].quantile(0.001)
app = app[(app['CNT_CHILDREN'] > q_low) & (app['CNT_CHILDREN'] < q_hi)]

q_hi = app['AMT_INCOME_TOTAL'].quantile(0.999)
q_low = app['AMT_INCOME_TOTAL'].quantile(0.001)
app = app[(app['AMT_INCOME_TOTAL'] > q_low) & (app['AMT_INCOME_TOTAL'] < q_hi)]

q_hi = app['CNT_FAM_MEMBERS'].quantile(0.999)
q_low = app['CNT_FAM_MEMBERS'].quantile(0.001)
app = app[(app['CNT_FAM_MEMBERS'] > q_low) & (app['CNT_FAM_MEMBERS'] < q_hi)]

crecord['Months from today'] = crecord['MONTHS_BALANCE']*-1
crecord = crecord.sort_values(['ID', 'Months from today'], ascending=True)
crecord['STATUS'].replace({'C': 0, 'X': 0}, inplace=True)
crecord['STATUS'] = crecord['STATUS'].astype('int')
crecord['STATUS'] = crecord['STATUS'].apply(lambda x: 1 if x >= 2 else 0)

crecordgb = crecord.groupby('ID').agg(max).reset_index()

df = app.join(crecordgb.set_index('ID'), on='ID', how='inner')
df['DAYS_BIRTH'] = df['DAYS_BIRTH']*-1
df['DAYS_EMPLOYED'] = df['DAYS_EMPLOYED']*-1
df.drop(['ID', 'Months from today', 'MONTHS_BALANCE'], axis=1, inplace=True)

df.reset_index(drop=True, inplace=True)


# This function is for Encoding data with parameter method(Encoding method)
def Encode(data, method):
    if method == 'OrdinalEncoding':
        encoder = preprocessing.OrdinalEncoder()
        encoded_data = encoder.fit_transform(data)
        encoded_data = pd.DataFrame(columns=data.columns, data=encoded_data)
    elif method == 'LabelEncoding':
        for x in data:
            if data[x].dtypes == 'object':
                data[x] = LabelEncoder().fit_transform(data[x])
        encoded_data = data

    return encoded_data


# This function is for Scaling data with parameter method(Scaling method)
def Scale(data, method):
    if method == 'MaxAbsScaling':
        scaler = preprocessing.MaxAbsScaler()
        scaled_data = scaler.fit_transform(data)
    elif method == 'MinMaxScaling':
        scaler = preprocessing.MinMaxScaler()
        scaled_data = scaler.fit_transform(data)
    elif method == 'RobustScaling':
        scaler = preprocessing.RobustScaler()
        scaled_data = scaler.fit_transform(data)
    elif method == 'StandardScaling':
        scaler = preprocessing.StandardScaler()
        scaled_data = scaler.fit_transform(data)

    df_scaled_data = pd.DataFrame(columns=data.columns, data=scaled_data)
    return df_scaled_data


def contribution(df, target):
    # Split df to features X and Target y
    X = df.drop(target, axis=1)
    y = df[target]

    # Divide X to Categorical X_cate and Numeric X_nume
    X_cate = X.select_dtypes(include='object')
    X_nume = X.select_dtypes(exclude='object')

    # In this function Standard, Robust, MinMax, MaxAbs Scaling will done
    scalingList = ['StandardScaling', 'RobustScaling', 'MinMaxScaling', 'MaxAbsScaling']
    # In this function Ordinal, Label Encoding will done
    encodingList = ['OrdinalEncoding', 'LabelEncoding']

    # After Ordinal Encoded data will at OE, Label Encoded data will at LE
    dataEncodeList = ['OE', 'LE']
    # After Scaling for each OE and LE they will be get 4 Scaling method
    # 1. Standard, 2. Robust, 3. MinMax, 4. MaxAbs so total 8 dataframe will came out
    dataframeList = ['df_stoe', 'df_rboe', 'df_mmoe', 'df_maoe',
                     'df_stla', 'df_rbla', 'df_mmla', 'df_mala']

    # Encode the Categorical dates
    for i in range(len(encodingList)):
        df_cate = Encode(X_cate, encodingList[i])
        dataEncodeList[i] = pd.concat([df_cate, X_nume], axis=1)

    # Scale the data that is Encoded
    for i in range(len(dataEncodeList)):
        for j in range(len(scalingList)):
            if i == 0:
                dataframeList[j] = Scale(dataEncodeList[i], scalingList[j])
            else:
                dataframeList[i + j + 3] = Scale(dataEncodeList[i], scalingList[j])

    # i is for checking which Encoded and Scaled data we are using
    i = 1;
    # Now we have Encoded and Scaled Dataset
    # Split dataset into train and test
    for x in dataframeList:
        X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)
        
        # solve unbalanced data
        oversample = SMOTE()
        X_train_balanced, y_train_balanced = oversample.fit_resample(X_train, y_train)
        X_test_balanced, y_test_balanced = oversample.fit_resample(X_test, y_test)

        # Set grid_params for KNN and DecisionTree for each
        grid_params_knn = {
            'n_neighbors': np.arange(3, 30),
            'weights': ['uniform', 'distance'],
            'metric': ['euclidean', 'manhattan']
        }
        grid_params_dt = {
            'min_samples_split': [2, 3, 4],
            'max_features': [3, 5, 7],
            'max_depth': [3, 5, 7],
            'max_leaf_nodes': list(range(7, 100))
        }

        # Make KNN and DecisionTree model with GreedSearchCV
        gs_knn = GridSearchCV(KNeighborsClassifier(), grid_params_knn, verbose=1, cv=5, n_jobs=-1)
        gs_knn.fit(X_train_balanced, y_train_balanced)
        gs_dt = GridSearchCV(DecisionTreeClassifier(), grid_params_dt, verbose=1, cv=3, n_jobs=-1)
        gs_dt.fit(X_train_balanced, y_train_balanced)

        if i == 1:
            print("Standard Scaling, Ordinal Encoding")
        if i == 2:
            print("Robust Scaling, Ordinal Encoding")
        if i == 3:
            print("MinMax Scaling, Ordinal Encoding")
        if i == 4:
            print("MaxAbs Scaling, Ordinal Encoding")
        if i == 5:
            print("Standard Scaling, Label Encoding")
        if i == 6:
            print("Robust Scaling, Label Encoding")
        if i == 7:
            print("MinMax Scaling, Ordinal Encoding")
        if i == 8:
            print("MaxAbs Scaling, Ordinal Encoding")

        i += 1

        # KNN Classifier part
        print("KNN Classifier")
        print("best_parameter: ", gs_knn.best_params_)
        print("best_train_score: %.2f" % gs_knn.best_score_)
        knn_score = gs_knn.score(X_test_balanced, y_test_balanced)
        print("test_score: %.2f" % knn_score)
        print()

        # DecisionTree Classifier part
        print("DecisionTree Classifier")
        print("best_parameter: ", gs_dt.best_params_)
        print("best_train_score: %.2f" % gs_dt.best_score_)
        dt_score = gs_dt.score(X_test_balanced, y_test_balanced)
        print("test_score: %.2f" % dt_score)
        print()


"""
Here calling contribution function
It will automatically Encode dataframe's Categorical columns
And Scale the whole data
By using KNN Classifier and DecisionTree Classifier With GreedSearchCV
Best parameters and Best score and test score will printed.
"""

# This function acts when you type contribution(dataframe df, target value name)
contribution(df, 'STATUS')
