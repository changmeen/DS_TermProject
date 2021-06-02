import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

from sklearn import preprocessing
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE
from sklearn.neighbors import KNeighborsClassifier

app = pd.read_csv("application_record.csv")
crecord = pd.read_csv("credit_record.csv")

app.drop('FLAG_MOBIL', axis=1, inplace=True)
app = app.drop_duplicates('ID', keep='last')
app.drop('OCCUPATION_TYPE', axis=1, inplace=True)

# Encoding non-numeric columns
le1 = LabelEncoder()
le1.fit(app['CODE_GENDER'])
app['CODE_GENDER'] = le1.transform(app['CODE_GENDER'])
le2 = LabelEncoder()
le2.fit(app['FLAG_OWN_CAR'])
app['FLAG_OWN_CAR'] = le2.transform(app['FLAG_OWN_CAR'])
le3 = LabelEncoder()
le3.fit(app['FLAG_OWN_REALTY'])
app['FLAG_OWN_REALTY'] = le3.transform(app['FLAG_OWN_REALTY'])
le4 = LabelEncoder()
le4.fit(app['NAME_INCOME_TYPE'])
app['NAME_INCOME_TYPE'] = le4.transform(app['NAME_INCOME_TYPE'])
le5 = LabelEncoder()
le5.fit(app['NAME_EDUCATION_TYPE'])
app['NAME_EDUCATION_TYPE'] = le5.transform(app['NAME_EDUCATION_TYPE'])
le6 = LabelEncoder()
le6.fit(app['NAME_FAMILY_STATUS'])
app['NAME_FAMILY_STATUS'] = le6.transform(app['NAME_FAMILY_STATUS'])
le7 = LabelEncoder()
le7.fit(app['NAME_FAMILY_STATUS'])
app['NAME_FAMILY_STATUS'] = le7.transform(app['NAME_FAMILY_STATUS'])

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

ma = preprocessing.MaxAbsScaler()
X_scale = ma.fit_transform(X)

over_sample = SMOTE()
X_balanced, y_balanced = over_sample.fit_resample(X_scale, y)

def prediction():
    print("()안의 내용들은 ,를 기준으로 입력 가능한 정보들이 적혀있는겁니다.")
    print("마지막 AMT_INCOME_TOTAL은 0이상의 실수(달러기준)를 입력해 주시면 됩니다.")
    print("0혹은 1을 입력하는 경우 0은 No 1은 Yes를 의미합니다.")
    NAME_FAMILY_STATUS = input("NAME_FAMILY_STATUS(Married,"
                               " Single / not married, Civil marriage, Separated, Widow)\n=>")
    FLAG_OWN_REALTY = input("FLAG_OWN_REALTY(Y, N)\n=>")
    FLAG_WORK_PHONE = input("FLAG_WORK_PHONE(0, 1)\n=>")
    NAME_EDUCATION_TYPE = input("NAME_EDUCATION_TYPE\n"
                                "(Secondary / secondary special, Higher education,\n"
                                "Incomplete higher, Lower secondary, Academic degree)\n=>")
    FLAG_PHONE = input("FLAG_PHONE(0, 1)\n=>")
    CODE_GENDER = input("CODE_GENDER(F, M)\n=>")
    AMT_INCOME_TOTAL = float(input("AMT_INCOME_TOTAL(Numeric input)\n=>"))

    KNN = KNeighborsClassifier(metric='euclidean', n_neighbors=9, weights='distance')
    KNN.fit(X_balanced, y_balanced)

    list = [[NAME_FAMILY_STATUS], [FLAG_OWN_REALTY], [FLAG_WORK_PHONE],
            [NAME_EDUCATION_TYPE], [FLAG_PHONE], [CODE_GENDER], [AMT_INCOME_TOTAL]]
    userDF = pd.DataFrame(list).T
    userDF.columns = ['NAME_FAMILY_STATUS', 'FLAG_OWN_REALTY',
                      'FLAG_WORK_PHONE', 'NAME_EDUCATION_TYPE',
                      'FLAG_PHONE', 'CODE_GENDER', 'AMT_INCOME_TOTAL']

    userDF['NAME_FAMILY_STATUS'] = le6.transform(userDF['NAME_FAMILY_STATUS'])
    userDF['FLAG_OWN_REALTY'] = le3.transform(userDF['FLAG_OWN_REALTY'])
    userDF['NAME_EDUCATION_TYPE'] = le5.transform(userDF['NAME_EDUCATION_TYPE'])
    userDF['CODE_GENDER'] = le1.transform(userDF['CODE_GENDER'])

    userDF_scale = ma.transform(userDF)
    result = int(KNN.predict(userDF_scale))

    if result == 0:
        print("Denied")
    else:
        print("Approved")


prediction()