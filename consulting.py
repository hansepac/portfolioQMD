#%%
import pandas as pd
import altair as alt
import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics

#%%
dat = pd.read_sas('data\gss2021.sas7bdat')

dat.dropna(subset=['HAPPY'], inplace=True)
dat.dropna(axis=1, how='all', inplace=True)
new_dat = dat.filter(['HAPPY', 'HAPMAR', 'LIFE', 'SATSOC', 'HLTHMNTL', 'ACTSSOC', 'HEALTH', 'EMOPROBS', 'ATTEND', 'EDUC', 'RELACTIV', 'HLTHPHYS'], axis=1)
new_dat.fillna(new_dat.mean(), inplace=True)

def happy_filter(df):
    if df >= 2:
        return 1
    if df == 1:
        return 0

fix_happy = new_dat.HAPPY.astype(int).apply(happy_filter)
x = new_dat.drop('HAPPY', axis=1)


#x_filt = x_filt.fillna(x_filt.mean())
"""
SATSOC - In general, how would you rate your satisfaction with your social activities and relationships?
HLTHMNTL - In general, how would you rate your mental health, including your mood and your ability to think?
ACTSSOC - In general, please rate how well you carry out your usual social activities and roles. (This includes activities at home, at work and in your community, and responsibilities as a parent, child, spouse, employee, friend, etc.)
HLTHPHYS - In general, how would you rate your physical health?
HEALTH: 1 - 4 (no very good) - Would you say your own health, in general, is excellent, good, fair, or  poor?
1 - Excellent
2 - Very Good
3 - Good
4 - Fair
5 - Poor
EMOPROBS - In the past seven days, how often have you been bothered by emotional problems such as feeling anxious, depressed or irritable?
1 - Never
2 - Rarely
3 - Sometimes
4 - Often
5 - Always
LIFE - In general, do you find life exciting, pretty routine, or dull?
1 - Exciting
2 - Routine
3 - Dull
ATTEND - How often do you attend religious services?
0 - Never
1 - Less than once a year
2 - About once or twice a year
3 - Several times a year
4 - About once a month
5 - 2-3 times a month
6 - Nearly every week
7 - Every week
8 - Several times a week
RELACTIV - How often do you take part in the activities and organizations of a church or place of worship other than attending services?
1 - Never
2 - Less than once a year
3 - About once or twice a year
4 - Several times a year
5 - About once a month
6 - 2-3 times a month
7 - Nearly every week
8 - Every week
9 - Several times a week
10 - Once a day
11 - Several times a day
EDUC - Highest year of education?
0 - 20 Years
HAPMAR - Would you say that your marriage is very happy, pretty happy, or not too happy?
1 - Very Happy
2 - Pretty Happy
3 - Not too Happy
"""

x_all = dat.drop(['HAPPY'], axis=1)
x_all = x_all.fillna(x_all.mean())
y = fix_happy
x_train, x_test, y_train, y_test = train_test_split(x_all, y, test_size= .2, random_state= 1234)

clf_rf = RandomForestClassifier().fit(x_train, y_train)

y_pred_rf = clf_rf.predict(x_test)

accuracy_rf = metrics.accuracy_score(y_pred_rf, y_test)
print(f"RandomForest: {accuracy_rf}")
# %%

feature_df = pd.DataFrame({'features':x_all.columns, 'importance':clf_rf.feature_importances_})
feature_df

chart = alt.Chart(feature_df).encode(
    x = alt.X('importance'),
    y = alt.Y('features', sort='-x')
).mark_bar()
chart
# %%
