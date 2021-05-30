import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

from nltk import DecisionTreeClassifier

warnings.filterwarnings('ignore')

from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif

app = pd.read_csv("application_record.csv")
crecord = pd.read_csv("credit_record.csv")

# Check values of FLAG_MOBILE column
# The result shows there are only value 1, it means we don't need FLAG_MOBIL column
# So drop FLAG_MOBIL column

app.drop('FLAG_MOBIL', axis=1, inplace=True)

# In the app dataset, ID feature has several duplicate values.
app = app.drop_duplicates('ID', keep='last')

# Make heatmap and check is there any null in application record
# The result says there are many null values at OCCUPATION_TYPE
# 여기 밑에 주석은 실행해야 하는 것
# sns.heatmap(app.isnull())
# plt.show()

# OCCUPATION_TYPE feature have a lot of NAN value
# So drop OCCUPATION_TYPE column from application_record
app.drop('OCCUPATION_TYPE', axis=1, inplace=True)

# Find columns that have non numeric values
# To check if they are useful or not
ot = pd.DataFrame(app.dtypes == 'object').reset_index()
object_type = ot[ot[0] == True]['index']

# Find column that have numeric values
# To check if they are useful or not
num_type = pd.DataFrame(app.dtypes != 'object').reset_index().rename(
    columns={0: 'yes/no'})
num_type = num_type[num_type['yes/no'] == True]['index']

# Encode object(Non numeric) columns into 0, 1
for x in app:
    if app[x].dtypes == 'object':
        app[x] = LabelEncoder().fit_transform(app[x])

# Clear outliers of 3 Columns
# FOR CNT_CHILDREN COLUMN
q_hi = app['CNT_CHILDREN'].quantile(0.999)
q_low = app['CNT_CHILDREN'].quantile(0.001)
app = app[(app['CNT_CHILDREN'] > q_low) & (app['CNT_CHILDREN'] < q_hi)]

# FOR AMT_INCOME_TOTAL COLUMN
q_hi = app['AMT_INCOME_TOTAL'].quantile(0.999)
q_low = app['AMT_INCOME_TOTAL'].quantile(0.001)
app = app[(app['AMT_INCOME_TOTAL'] > q_low) & (app['AMT_INCOME_TOTAL'] < q_hi)]

# FOR CNT_FAM_MEMBERS COLUMN
q_hi = app['CNT_FAM_MEMBERS'].quantile(0.999)
q_low = app['CNT_FAM_MEMBERS'].quantile(0.001)
app = app[(app['CNT_FAM_MEMBERS'] > q_low) & (app['CNT_FAM_MEMBERS'] < q_hi)]

crecord['Months from today'] = crecord['MONTHS_BALANCE'] * -1
crecord = crecord.sort_values(['ID', 'Months from today'], ascending=True)

# 0: 1-29 days past due
# 1: 30-59 days past due
# 2: 60-89 days overdue
# 3: 90-119 days overdue
# 4: 120-149 days overdue
# 5: Overdue or bad debts, write-offs for more than 150 days
# C: paid off that month
# X: No loan for the month

# https://www.thebalance.com/when-does-a-late-payment-go-on-my-credit-report-960434
# According to this post, Some creditors or lenders cannot report overdue payments to credit investigators until 60 days later.
# So we changed less than 60 days, no payments, no loans to zero (good credit) and more than 60 days of arrears to 1 (bad credit).

crecord['STATUS'].replace({'C': 0, 'X': 0}, inplace=True)
crecord['STATUS'] = crecord['STATUS'].astype('int')
crecord['STATUS'] = crecord['STATUS'].apply(lambda x: 1 if x >= 2 else 0)

# Because the ID was duplicated, it was reduced to the record that was used the longest time ago.
crecordgb = crecord.groupby('ID').agg(max).reset_index()

# Merge application_record and credit_record by ID
# After Merge drop useless columns
df = app.join(crecordgb.set_index('ID'), on='ID', how='inner')
df['DAYS_BIRTH'] = df['DAYS_BIRTH'] * -1
df['DAYS_EMPLOYED'] = df['DAYS_EMPLOYED'] * -1
df.drop(['ID', 'Months from today', 'MONTHS_BALANCE'], axis=1, inplace=True)

# The label of df is STATUS but there are too many 0 and small number of 1


df.reset_index(drop=True, inplace=True)

X = df.drop(['STATUS'], axis=1)
y = df['STATUS']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# 여기서부터 스케일링 방식에 따라 달라짐
# -----------------------------------------------------------------------------------------
# Fourth -> MaxAbsScaler

ma = preprocessing.MaxAbsScaler()
X_train_scale = ma.fit_transform(X_train)
X_test_scale = ma.transform(X_test)

bestfeatures = SelectKBest(score_func=f_classif, k=10)
fit = bestfeatures.fit(X_train, y_train)

dfcolumns = pd.DataFrame(X_train.columns)
dfscores = pd.DataFrame(fit.scores_)

featureScores = pd.concat([dfcolumns, dfscores], axis=1)
featureScores.columns = ['Specs', 'Score']
print("featureScores")
print(featureScores)

# Because label has too small number of 0 we have to do something
# So we use SMOTE to copy data randomly who's value is 1
oversample = SMOTE()
X_train_balanced, y_train_balanced = oversample.fit_resample(X_train_scale, y_train)
X_test_balanced, y_test_balanced = oversample.fit_resample(X_test_scale, y_test)

print("Print the y_train (before oversampling)")
print(y_train.value_counts())
print()

print("Print the y_train_balanced (after oversampling y_train)")
print(y_train_balanced.value_counts())
print()

print("Print the y_test (before oversampling)")
print(y_test.value_counts())
print()

print("Print the y_test_balanced (after oversampling y_test)")
print(y_test_balanced.value_counts())
print()

grid_params_knn = {
    'n_neighbors': np.arange(3, 7),
    'weights': ['uniform', 'distance'],
    'metric': ['euclidean', 'manhattan']
}

gs_knn = GridSearchCV(KNeighborsClassifier(), grid_params_knn, verbose=1, cv=3, n_jobs=-1)
gs_knn.fit(X_train_balanced, y_train_balanced)

print("MaxAbs Scaler, KNN Classifier")
print("best_params_: ", gs_knn.best_params_)
print("best_score_: ", gs_knn.best_score_)

prediction = gs_knn.predict(X_test_balanced)
score = gs_knn.score(X_test_balanced, y_test_balanced)
print("score: %.2f" % score)
print()

# -----------------------------------------------------------------------------------------

# Decision Tree with Entropy
# param 정리

grid_params_dt = {
    'max_leaf_nodes': list(range(2, 100)),
    'min_samples_split': [2, 3, 4]
}

gs_dt = GridSearchCV(DecisionTreeClassifier(), grid_params_dt, verbose=1, cv=3, n_jobs=-1)
gs_dt.fit(X_train_balanced, y_train_balanced)


# Show the results...
print("MaxAbs Scaler, Decision Tree Classifier")
print("best_params_: ", gs_dt.best_params_)
print("best_score_: ", gs_dt.best_score_)

prediction = gs_dt.predict(X_test_balanced)
score = gs_dt.score(X_test_balanced, y_test_balanced)
print("score: %.2f" % score)
print()

# -----------------------------------------------------------------------------------------

