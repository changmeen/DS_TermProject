import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

from sklearn.preprocessing import LabelEncoder
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif

app = pd.read_csv("application_record.csv")
crecord = pd.read_csv("credit_record.csv")

# Print application_record Dataset
# There are 438,557 records at application_record Dataset
print("Print application_record Dataset")
print(app)
print()

# Print credit_record Dataset
# There are 1,048,575 records at credit_record Dataset
print("Print credit_record Dataset")
print(crecord)
print()

# Check values of FLAG_MOBILE column
# The result shows there are only value 1, it means we don't need FLAG_MOBIL column
# So drop FLAG_MOBIL column
print("Check values of FLAG_MOBILE column")
print(app['FLAG_MOBIL'].value_counts())
print()
app.drop('FLAG_MOBIL', axis=1, inplace=True)

# Check Number of unique ID from application_record
# Result is 438,510 it is smaller than 438,557
# It means there are duplicated records in application_record
print("Check Number of unique ID from application_record")
print(app['ID'].nunique())
print()

# Check Number of unique ID from credit_record
# Result is 45,985 it is much smaller than 1,048,575
# It means there are so many duplicated records in credit_record
print("Check Number of unique ID from credit_record")
print(crecord['ID'].nunique())
print()

# In the app dataset, ID feature has several duplicate values.
app = app.drop_duplicates('ID', keep='last')

# Make heatmap and check is there any null in application record
# The result says there are many null values at OCCUPATION_TYPE
sns.heatmap(app.isnull())
plt.show()

# OCCUPATION_TYPE feature have a lot of NAN value
# So drop OCCUPATION_TYPE column from application_record
app.drop('OCCUPATION_TYPE', axis=1, inplace=True)

# Find columns that have non numeric values
# To check if they are useful or not
# Print object columns names
ot = pd.DataFrame(app.dtypes == 'object').reset_index()
object_type = ot[ot[0] == True]['index']
print("columns that have non-numeric values from application_record")
print(object_type)
print()

# Find column that have numeric values
# To check if they are useful or not
num_type = pd.DataFrame(app.dtypes != 'object').reset_index().rename(
    columns={0: 'yes/no'})
num_type = num_type[num_type['yes/no'] == True]['index']

# Print numeric columns number per values
a = app[object_type]['CODE_GENDER'].value_counts()
b = app[object_type]['FLAG_OWN_CAR'].value_counts()
c = app[object_type]['FLAG_OWN_REALTY'].value_counts()
d = app[object_type]['NAME_INCOME_TYPE'].value_counts()
e = app[object_type]['NAME_EDUCATION_TYPE'].value_counts()
f = app[object_type]['NAME_FAMILY_STATUS'].value_counts()
g = app[object_type]['NAME_HOUSING_TYPE'].value_counts()

print("Print Numeric Columns number per values from application_record")
print(a, "\n")
print(b, "\n")
print(c, "\n")
print(d, "\n")
print(e, "\n")
print(f, "\n")
print(g, "\n")

# Encode object(Non numeric) columns into 0, 1
for x in app:
    if app[x].dtypes == 'object':
        app[x] = LabelEncoder().fit_transform(app[x])
print("Encode object(Non numeric) columns into 0, 1 from application_record")
print(app.head())
print()

print("Print application Dataset of Numeric columns")
print(app[num_type].head())
print()

# Show Numeric values as scatter plot
fig, ax = plt.subplots(nrows=3, ncols=3, figsize=(14, 6))
sns.scatterplot(x='ID', y='CNT_CHILDREN', data=app, ax=ax[0][0], color='orange')
sns.scatterplot(x='ID', y='AMT_INCOME_TOTAL', data=app, ax=ax[0][1], color='orange')
sns.scatterplot(x='ID', y='DAYS_BIRTH', data=app, ax=ax[0][2])
sns.scatterplot(x='ID', y='DAYS_EMPLOYED', data=app, ax=ax[1][0])
sns.scatterplot(x='ID', y='FLAG_WORK_PHONE', data=app, ax=ax[1][2])
sns.scatterplot(x='ID', y='FLAG_PHONE', data=app, ax=ax[2][0])
sns.scatterplot(x='ID', y='FLAG_EMAIL', data=app, ax=ax[2][1])
sns.scatterplot(x='ID', y='CNT_FAM_MEMBERS', data=app, ax=ax[2][2], color='orange')

# There are outliers in 3 columns: CNT_CHILDREN, AMT_INCOME_TOTAL, CNT_FAM_MEMBERS
plt.show()

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

# See result of cleaning outliers
fig, ax = plt.subplots(nrows=3, ncols=3, figsize=(14, 6))
sns.scatterplot(x='ID', y='CNT_CHILDREN', data=app, ax=ax[0][0], color='orange')
sns.scatterplot(x='ID', y='AMT_INCOME_TOTAL', data=app, ax=ax[0][1], color='orange')
sns.scatterplot(x='ID', y='DAYS_BIRTH', data=app, ax=ax[0][2])
sns.scatterplot(x='ID', y='DAYS_EMPLOYED', data=app, ax=ax[1][0])
sns.scatterplot(x='ID', y='FLAG_WORK_PHONE', data=app, ax=ax[1][2])
sns.scatterplot(x='ID', y='FLAG_PHONE', data=app, ax=ax[2][0])
sns.scatterplot(x='ID', y='FLAG_EMAIL', data=app, ax=ax[2][1])
sns.scatterplot(x='ID', y='CNT_FAM_MEMBERS', data=app, ax=ax[2][2], color='orange')
plt.show()

crecord['Months from today'] = crecord['MONTHS_BALANCE']*-1
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
print("Print the STATUS values from credit_record")
print(crecord['STATUS'].value_counts())
print()

crecord['STATUS'].replace({'C': 0, 'X': 0}, inplace=True)
crecord['STATUS'] = crecord['STATUS'].astype('int')
crecord['STATUS'] = crecord['STATUS'].apply(lambda x: 1 if x >= 2 else 0)

print("Print the STATUS values changed by percentage from credit_record")
print(crecord['STATUS'].value_counts(normalize=True))
print()

# Because the ID was duplicated, it was reduced to the record that was used the longest time ago.
crecordgb = crecord.groupby('ID').agg(max).reset_index()

# Merge application_record and credit_record by ID
# After Merge drop useless columns
df = app.join(crecordgb.set_index('ID'), on='ID', how='inner')
df['DAYS_BIRTH'] = df['DAYS_BIRTH']*-1
df['DAYS_EMPLOYED'] = df['DAYS_EMPLOYED']*-1
df.drop(['ID', 'Months from today', 'MONTHS_BALANCE'], axis=1, inplace=True)

# The label of df is STATUS but there are too many 0 and small number of 1
print("Print the STATUS values changed by percentage from df")
print(df['STATUS'].value_counts(normalize=True))
print()

df.reset_index(drop=True, inplace=True)
print("Print df after drop ID, Months from today, MONTHS_BALANCE")
print(df)
print()

X = df.drop(['STATUS'], axis=1)
y = df['STATUS']

bestfeatures = SelectKBest(score_func=f_classif, k=10)
fit = bestfeatures.fit(X, y)

dfcolumns = pd.DataFrame(X.columns)
dfscores = pd.DataFrame(fit.scores_)

featureScores = pd.concat([dfcolumns, dfscores], axis=1)
featureScores.columns = ['Specs', 'Score']

# The Specs that it's score is higher than 0 is:
# NAME_FAMILY_STATUS, FLAG_OWN_REALITY, FLAG_WORK_PHONE,
# NAME_EDUCATION_TYPE, FLAG_PHONE, CODE_GENDER, AMT_INCOME_TOTAL
# Total 7 features are chosen to consider
print("Find out Important Features by using SelectKBest")
print(featureScores.nlargest(10, 'Score'))
