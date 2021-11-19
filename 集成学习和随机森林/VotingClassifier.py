'''
有许多类别分类器，假设每个分类器准确率有50%，
如果有1000个分类器
假设你创建了一个包含1000个分类器的集成，每个分类器都只有51%的概率是
正确的（几乎不比随机猜测强多少）。如果你以大多数投票的类别作为预测结果，可以期
待的准确率高达75%。但是，这基于的前提是所有的分类器都是完全独立的，彼此的错误
毫不相关。显然这是不可能的，因为它们都是在相同的数据上训练的，很可能会犯相同的
错误，所以也会有很多次大多数投给了错误的类别，导致集成的准确率有所降低
总之利用多个分类器，再从分类器的各个预测结果情况，预测出最佳预测
'''

from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_moons
import matplotlib.pyplot as plt

#数据准备
X, y = make_moons(n_samples=500, noise=0.30, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
plt.plot(X,y,"b.")
plt.show()

log_clf=LogisticRegression(solver="lbfgs", random_state=42)
rnd_clf=RandomForestClassifier(n_estimators=100, random_state=42)
svm_clf=SVC(gamma="scale", random_state=42)

#训练模型
voting_clf = VotingClassifier(
    estimators=[('lr', log_clf), ('rf', rnd_clf), ('svc', svm_clf)],
    voting='hard')
voting_clf.fit(X_train, y_train)

# 使用投票器分类模型进行预测
from sklearn.metrics import accuracy_score

print("硬投票: ")
for clf in (log_clf, rnd_clf, svm_clf, voting_clf):
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    #y_probability=clf.predict_proba(X_test)
    print(clf.__class__.__name__, accuracy_score(y_test, y_pred))
    #print("概率 : ",y_probability)
'''
与可见投票器分类器 更加准确较高
LogisticRegression 0.864
RandomForestClassifier 0.896
SVC 0.896
VotingClassifier 0.912
'''



# 软投票
'''
将概率在所有单个分类器上平均，然后让Scikit-Learn给出平均概率最高的类别作为预
测。这被称为软投票法。
'''
'''
做的就是用voting="soft"代替voting="hard"，并确保所
有分类器都可以估算出概率。默认情况下，SVC类是不行的，所以你需要将其超参数
probability设置为True（这会导致SVC使用交叉验证来估算类别概率，减慢训练速度，并
会添加predict_proba（）方法）
'''
log_clf = LogisticRegression(solver="lbfgs", random_state=42)
rnd_clf = RandomForestClassifier(n_estimators=100, random_state=42)
svm_clf = SVC(gamma="scale", probability=True, random_state=42)
#训练软投票分类器
voting_clf = VotingClassifier(
    estimators=[('lr', log_clf), ('rf', rnd_clf), ('svc', svm_clf)],
    voting='soft')
voting_clf.fit(X_train, y_train)

#使用测试集预测
print("软投票： ");
for clf in (log_clf, rnd_clf, svm_clf, voting_clf):
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    print(clf.__class__.__name__, accuracy_score(y_test, y_pred))

