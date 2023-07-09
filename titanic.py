import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt

# 載入資料
titanic = sns.load_dataset('titanic')

#繪製生死統計圖
fig = plt.figure(figsize=(16,8),dpi=1600)
ax1 = plt.subplot2grid(((1,2)),(0,0))
titanic['survived'].value_counts().plot(kind='bar',alpha=0.5)
ax1.set_xlim(-1,2)
plt.title('distribution of survival (1 = survived)')

#生死年齡分佈 顏色越深越多人
ax2=plt.subplot2grid((1,2),(0,1))
plt.scatter(titanic['survived'],titanic['age'],alpha=0.2)
plt.ylabel('age')
ax2.set_xlim(-1,2)
plt.title('survival by age 1=survived')




#titanic.info()

#計算空值
countisunll = titanic.isnull().sum()

#統計survived人數
survival = titanic['survived'].value_counts()


titanic = titanic.drop(['embark_town','deck','who','adult_male'],axis=1)

#補充空值
titanic['embarked']=titanic['embarked'].fillna(method='ffill')
titanic['age']=titanic['age'].fillna(method='ffill')


#資料預處理
lb = LabelEncoder()
titanic['alive']=lb.fit_transform(titanic['alive'])
titanic['sex']=lb.fit_transform(titanic['sex'])
titanic['embarked']=lb.fit_transform(titanic['embarked'])
titanic['class']=lb.fit_transform(titanic['class'])

#決策樹
clf = DecisionTreeClassifier()
x=titanic.iloc[:,1:]
y=titanic.iloc[:,0]
clf.fit(x,y)

#資料預測
pred_x=titanic.iloc[:20,1:]
print(clf.predict_proba(pred_x))
print(clf.predict(pred_x))
lables = ['dide','survived']
print([lables[i] for i in clf.predict(pred_x)])
pred_y = clf.predict(pred_x)
pred_y.reshape(20,1)
#準確度
print(clf.score(pred_x, pred_y))
