# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
from IPython import get_ipython

# %%
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
#get_ipython().run_line_magic('matplotlib', 'inline')
#get_ipython().run_line_magic('config', "InlineBackend.figure_format = 'retina'")


# %%
print(f'pandas  version: {pd.__version__}')
print(f'numpy   version: {np.__version__}')
print(f'seaborn version: {sns.__version__}')


# %%
#url='https://github.com/prasertcbs/basic-dataset/raw/master/diabetes.csv'
url='C:\AppServ\www\ML\m_elderly_person_area1.csv'
df=pd.read_csv(url)
df.head()


# %%
df.info()


# %%
#sns.pairplot(df,
#             kind='reg', 
#             plot_kws={'scatter_kws': {'alpha': 0.4}, 
#                       'line_kws': {'color': 'orange'}},
#             diag_kws={'color': 'green', 'alpha':.2});


# %%
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import Perceptron
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn import metrics


# %%
print(f'sklearn version: {sklearn.__version__}')


# %%
df.columns


# %%
#model= DecisionTreeClassifier(random_state=7)
model= RandomForestClassifier(n_estimators=250, random_state=7)
# model= ExtraTreesClassifier(n_estimators=250, random_state=7)
#X=df[[ 'weight', 'height', 'waistline']]
#X=df[[ 'weight', 'height', 'waistline','income_total','expend_avg','expend_health_year','work_week_hr','gender']]		
X=df[[ 'weight', 'height', 'waistline','income_total','expend_avg','expend_health_year','work_week_hr','gender']]		
y=df['sectio4_score']
model.fit(X,y)


# %%
model.feature_importances_


# %%
fs=pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=True)
fs


# %%
fs.sum()


# %%
fs.plot(kind='barh')


# %%
fs[fs > .1]


# %%
fs.nlargest(4) #.index


# %%
fs[fs > .1].index


# %%
X=df[fs[fs > .1].index]
X.head()


# %%
X=df[['weight', 'height', 'waistline','income_total','expend_avg','expend_health_year','work_week_hr','gender']]
#X=df[['income_total','weight','expend_avg','height']]
#X=df[['height','waistline']]
# X=df[fs[fs > .1].index]
y=df['sectio4_score']


# %%
# use stratify to split train/test
test_size=.2
X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                    test_size=test_size, 
                                                   # stratify=y,
                                                    random_state=7)


# %%
algo=[
    #[KNeighborsClassifier(n_neighbors=5), 'KNeighborsClassifier'], 
    #[LogisticRegression(solver='lbfgs'), 'LogisticRegression'], 
    #[Perceptron(), 'Perceptron'],
    [DecisionTreeClassifier(min_samples_split=10), 'DecisionTreeClassifier'],
    #[GradientBoostingClassifier(), 'GradientBoostingClassifier'],
    [RandomForestClassifier(), 'RandomForestClassifier']
    #[BaggingClassifier(), 'BaggingClassifier'],
    #[AdaBoostClassifier(), 'AdaBoostClassifier'],
    #[GaussianNB(), 'GaussianNB'],
    #[MLPClassifier(), 'MLPClassifier'],
    #[SVC(kernel='linear'), 'SVC_linear']
    #[GaussianProcessClassifier(), 'GaussianProcessClassifier']
]
model_scores=[]
for a in algo:
    model = a[0]
    model.fit(X_train, y_train)
    score=model.score(X_test, y_test)
    model_scores.append([score, a[1]])
    y_pred=model.predict(X_test)
    print(f'{a[1]:20} score: {score:.04f}')
    print(metrics.confusion_matrix(y_test, y_pred))
    print(metrics.classification_report(y_test, y_pred))
    print('-' * 100)

print(model_scores)
print(f'best score = {max(model_scores)}')    


# %%
model_scores


# %%
dscore=pd.DataFrame(model_scores, columns=['score', 'classifier'])
dscore.sort_values('score', ascending=False)


# %%

#X=df[['height','weight','waistline']]
X=df[['height','waistline']]
# X=df[fs[fs > .1].index]
y=df['gender']#get correlations of each features in dataset
corrmat = df.corr()
top_corr_features = corrmat.index
plt.figure(figsize=(10,10))
#plot heat map
g=sns.heatmap(df[top_corr_features].corr(),annot=True,cmap="RdYlGn")


# %%



