# 聚类用于数据预处理
# 数据准备
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_digits
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans


X_digits, y_digits = load_digits(return_X_y=True)
print(len(X_digits))  # 1797张手写数字数据集
print(y_digits)
X_train, X_test, y_train, y_test = train_test_split(
    X_digits, y_digits, random_state=42)

# 逻辑回归
log_reg = LogisticRegression(
    multi_class="ovr", solver="lbfgs", max_iter=5000, random_state=42)
log_reg.fit(X_train, y_train)
print(log_reg.score(X_test, y_test))  # 精准率可以达到0.968+

#利用KMeans可以使得其效果更好
pipeline = Pipeline([
    ("kmeans", KMeans(n_clusters=50, random_state=42)),#聚类50个簇
    ("log_reg", LogisticRegression(multi_class="ovr",#在进行逻辑回归
                                   solver="lbfgs", max_iter=5000, random_state=42)),
])
pipeline.fit(X_train, y_train)
print(pipeline.score(X_test,y_test))#精准率达到了0.97777+ 

# 上面是随意指定的K值，那么选一个较优的K值呢
# 网格搜索
param_grid = dict(kmeans__n_clusters=range(2, 100))
grid_clf = GridSearchCV(pipeline, param_grid, cv=3, verbose=2)
grid_clf.fit(X_train, y_train)
print(grid_clf.best_params_)
print(grid_clf.score(y_train,y_test))#效果会比原来更好一点  

