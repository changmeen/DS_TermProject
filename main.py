import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
import warnings
from sklearn import preprocessing
warnings.filterwarnings('ignore')

app = pd.read_csv("application_record.csv")
crecord = pd.read_csv("credit_record.csv")

app['FLAG_MOBIL'].value_counts()
app.drop('FLAG_MOBIL', axis=1)

# In the app dataset, ID feature has several duplicate values.
app = app.drop_duplicates('ID', keep='last')
# OCCUPATION_TYPE feature have a lot of NAN value
app.drop('OCCUPATION_TYPE', axis=1, inplace=True)

for x in app:
    if app[x].dtypes=='object':
        app[x] = LabelEncoder().fit_transform(app[x])
print(app.head(10))

# eliminate outlier
# FOR CNT_CHILDREN COLUMN
q_hi = app['CNT_CHILDREN'].quantile(0.999)
q_low = app['CNT_CHILDREN'].quantile(0.001)
app = app[(app['CNT_CHILDREN']>q_low) & (app['CNT_CHILDREN']<q_hi)]

# FOR AMT_INCOME_TOTAL COLUMN
q_hi = app['AMT_INCOME_TOTAL'].quantile(0.999)
q_low = app['AMT_INCOME_TOTAL'].quantile(0.001)
app= app[(app['AMT_INCOME_TOTAL']>q_low) & (app['AMT_INCOME_TOTAL']<q_hi)]

#FOR CNT_FAM_MEMBERS COLUMN
q_hi = app['CNT_FAM_MEMBERS'].quantile(0.999)
q_low = app['CNT_FAM_MEMBERS'].quantile(0.001)
app= app[(app['CNT_FAM_MEMBERS']>q_low) & (app['CNT_FAM_MEMBERS']<q_hi)]

crecord['Months from today'] = crecord['MONTHS_BALANCE']*-1
crecord = crecord.sort_values(['ID','Months from today'], ascending=True)

# 0: 1-29 days past due 
# 1: 30-59 days past due 
# 2: 60-89 days overdue 
# 3: 90-119 days overdue 
# 4: 120-149 days overdue 
# 5: Overdue or bad debts, write-offs for more than 150 days 
# C: paid off that month 
# X: No loan for the month

print(crecord['STATUS'].value_counts())

crecord['STATUS'].replace({'C': 0, 'X' : 0}, inplace=True)
crecord['STATUS'] = crecord['STATUS'].astype('int')
crecord['STATUS'] = crecord['STATUS'].apply(lambda x:1 if x >= 2 else 0)

crecord['STATUS'].value_counts(normalize=True)

crecordgb = crecord.groupby('ID').agg(max).reset_index()

df = app.join(crecordgb.set_index('ID'), on='ID', how='inner')
df.drop(['Months from today', 'MONTHS_BALANCE'], axis=1, inplace=True)

print(df['STATUS'].value_counts(normalize=True))