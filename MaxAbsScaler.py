import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
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

for x in app:
    if app[x].dtypes == 'object':
        app[x] = LabelEncoder().fit_transform(app[x])

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

X = df[['NAME_FAMILY_STATUS', 'FLAG_OWN_REALTY', 'FLAG_WORK_PHONE', 'NAME_EDUCATION_TYPE',
        'FLAG_PHONE', 'CODE_GENDER', 'AMT_INCOME_TOTAL']]
y = df['STATUS']

# Upside is same from Main.py only change showing part
# ---------------------------------------------------------------------------------------------------------------------
# At Downside we do scaling, using KNN, Decision Tree Classifiers
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# MaxAbs Scaling
ma = preprocessing.MaxAbsScaler()
X_train_scale = ma.fit_transform(X_train)
X_test_scale = ma.transform(X_test)

# Because label has too small number of 0 we have to do something
# So we use SMOTE to copy data randomly who's value is 1
oversample = SMOTE()
X_train_balanced, y_train_balanced = oversample.fit_resample(X_train_scale, y_train)
X_test_balanced, y_test_balanced = oversample.fit_resample(X_test_scale, y_test)

# Set HyperParameters of KNeighborsClassifier
grid_params_knn = {
    'n_neighbors': np.arange(3, 30),
    'weights': ['uniform', 'distance'],
    'metric': ['euclidean', 'manhattan']
}

# Make GridSearchCV with KNeighborsClassifier
# make model by fit train dataset
gs_knn = GridSearchCV(KNeighborsClassifier(), grid_params_knn, verbose=1, cv=5, n_jobs=-1)
gs_knn.fit(X_train_balanced, y_train_balanced)

# Show the model performance of train set
print("MaxAbs Scaler, KNN Classifier")
print("best_parameter: ", gs_knn.best_params_)
print("best_train_score: %.2f" % gs_knn.best_score_)

# Show the score of model from test set
knn_score = gs_knn.score(X_test_balanced, y_test_balanced)
print("score: %.2f" % knn_score)
print()

# ------------------------------------------DecisionTree---------------------------------------------------
# Set HyperParameters of DecisionTreeClassifier
grid_params_dt = {
    'min_samples_split': [2, 3, 4],
    'max_features': [3, 5, 7],
    'max_depth': [3, 5, 7],
    'max_leaf_nodes': list(range(7, 100))
}

# Make GridSearchCV with DecisionTreeClassifier
# make model by fit train dataset
gs_dt = GridSearchCV(DecisionTreeClassifier(), grid_params_dt, verbose=1, cv=3, n_jobs=-1)
gs_dt.fit(X_train_balanced, y_train_balanced)

# Show the model performance of train set
print("MaxAbs Scaler, Decision Tree Classifier")
print("best_parameter: ", gs_dt.best_params_)
print("best_train_score: %.2f" % gs_dt.best_score_)

# Show the score of model from test set
dt_score = gs_dt.score(X_test_balanced, y_test_balanced)
print("score: %.2f" % dt_score)
print()

# -----------------------------------------Show the result----------------------------------------------------
# Using bar plot, show the scores per Classifiers
Classifiers = ['KNN Classifier', 'DecisionTree Classifier']
Scores = [knn_score * 100, dt_score * 100]
plt.bar(Classifiers, Scores, width=0.5)
plt.title('MaxAbs Scaling')
plt.xlabel('Classifiers')
plt.ylabel('Scores (%)')
plt.show()

# -----------------------------------------------------------------------------------------