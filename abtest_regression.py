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

# ======== A/B test===========

diff = []
size_df2 = df2.shape[0]
for _ in range(10000):
    sample = df2.sample(size_df2, replace=True)
    nm = sample.query('landing_page == "new_page"')['converted'].mean()
    om = sample.query('landing_page == "old_page"')['converted'].mean()
    diff.append(nm-om)

plt.hist(diff);
observed_diff = df2.query('landing_page=="new_page"').converted.mean()-df2.query('landing_page=="old_page"').converted.mean()
diff = np.array(diff)
sim_null = np.random.normal(0, diff.std(), diff.size)
plt.hist(sim_null);
plt.axvline(x=observed_diff,color='red')


new_page_converted = np.random.choice([0, 1], size=nnew, p=[1-df2.converted.mean(), df2.converted.mean()])
new_page_converted.mean()

old_page_converted = np.random.choice([0, 1], size=nnew, p=[1-df2.converted.mean(), df2.converted.mean()])
old_page_converted.mean()

new_page_converted.mean()-old_page_converted.mean()

p_diffs = []

bias_rate = df2.converted.mean()

for _ in range(10000):
    new_page_sim = np.random.choice([0, 1], size=nnew, p=[1-bias_rate, bias_rate])
    old_page_sim = np.random.choice([0, 1], size=nnew, p=[1-bias_rate, bias_rate])
    p_diffs.append(new_page_sim.mean()-old_page_sim.mean())

p_diffs = np.array(p_diffs)
plt.hist(p_diffs);
plt.axvline(x=observed_diff,color='red')

(p_diffs>observed_diff).mean()

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

