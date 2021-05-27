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