import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt
import statsmodels.api as sm
%matplotlib inline
#We are setting the seed to assure you get the same answers on quizzes as we set up
random.seed(42)

# ======== data cleansing ===========
df = pd.read_csv('ab_data.csv')
df.head()

df2 = df.query('group=="treatment" and landing_page=="new_page"').append(df.query('group=="control" and landing_page=="old_page"'))
df2.describe()

# Double Check all of the correct rows were removed - this should be 0
df2[((df2['group'] == 'treatment') == (df2['landing_page'] == 'new_page')) == False].shape[0]

df2.query('user_id==773192')
df2 = df2.drop(1899, axis=0)

# ======== Using proportions_ztest ===========
convert_old = df2.query('converted == 1 and landing_page == "old_page"').user_id.count()
convert_new = df2.query('converted == 1 and landing_page == "new_page"').user_id.count()
n_old = df2.query('landing_page == "old_page"').user_id.count()
n_new = df2.query('landing_page == "new_page"').user_id.count()

stats, pval = sm.stats.proportions_ztest([convert_old, convert_new], [n_old, n_new], alternative='smaller') 

# ======== Logistic Regression ===========
df_copy = df2.copy()
df_copy['intercept'] = 1
df_copy[['new_page','old_page']] = pd.get_dummies(df_copy['landing_page'])
df_copy['ab_page'] = df['group'].apply(lambda x: 1 if x=='treatment' else 0)
df_copy.head()

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, precision_score, recall_score, accuracy_score
from sklearn.model_selection import train_test_split
np.random.seed(42)

model = sm.Logit(df_copy['converted'],df_copy[['intercept','ab_page']])
result = model.fit()
result.summary()

print(np.exp(-.015))

y = df_copy['converted']
X = df_copy[['ab_page']]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
lr = LogisticRegression()
lr.fit(X_train, y_train)
y_pred = lr.predict(X_test)
print(precision_score(y_test, y_pred))
print(recall_score(y_test, y_pred))
print(accuracy_score(y_test, y_pred))
confusion_matrix(y_test, y_pred)

