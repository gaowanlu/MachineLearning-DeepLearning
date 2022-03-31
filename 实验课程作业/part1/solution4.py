# 训练与可视化决策树
'''
决策树可视化软件
http://wwwgraphviz.org/ 
用Graphviz软件包中的dot命令行工具将此.dot文件转换为多种格式，
例如PDF或PNG[1]。此命令行将.dot文件转换为.png图像文件：  
dot -Tpng iris_tree.dot -o iris_tree.png
'''

from copyreg import constructor
from graphviz import Source
from sklearn.tree import export_graphviz
import os
from sklearn.tree import DecisionTreeClassifier
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split


MYLABEL = ['age of the patient', 'spectacle prescription',
           'astigmatic', 'tear production rate']
MYDATA = [
    ['young',  'myope',	'no', 'reduced',    'no lenses'],
    ['young',  'myope',	'no', 'normal',    'soft'],
    ['young',  'myope',	'yes', 'reduced', 'no lenses'],
    ['young',  'myope',	'yes', 'normal',    'hard'],
    ['young',  'hyper',	'no', 'reduced',    'no lenses'],
    ['young',  'hyper',	'no', 'normal',    'soft'],
    ['young',  'hyper',	'yes', 'reduced', 'no lenses'],
    ['young',  'hyper',	'yes', 'normal',    'hard'],
    ['pre',  'myope',	'no', 'reduced',    'no lenses'],
    ['pre',  'myope',	'no', 'normal',    'soft'],
    ['pre',  'myope',	'yes', 'reduced', 'no lenses'],
    ['pre',  'myope',	'yes', 'normal',    'hard'],
    ['pre',  'hyper',	'no', 'reduced',    'no lenses'],
    ['pre',  'hyper',	'no', 'normal',    'soft'],
    ['pre',  'hyper',	'yes', 'reduced', 'no lenses'],
    ['pre',  'hyper',	'yes', 'normal',    'no lenses'],
    ['presbyopic',	'myope'	,    'no', 'reduced', 'no lenses'],
    ['presbyopic',	'myope'	,    'no', 'normal',    'no lenses'],
    ['presbyopic',	'myope'	,    'yes', 'reduced',    'no lenses'],
    ['presbyopic',	'myope'	,    'yes', 'normal',    'hard'],
    ['presbyopic',	'hyper'	,    'no', 'reduced', 'no lenses'],
    ['presbyopic',	'hyper'	,    'no', 'normal',    'soft'],
    ['presbyopic',	'hyper'	,    'yes', 'reduced',    'no lenses'],
    ['presbyopic',	'hyper'	,    'yes', 'normal',    'no lenses']
]
# 贴标签
mapper = {
    'young': 0,
    'pre': 1,
    'presbyopic': 2,
    'myope': 3,
    'hyper': 4,
    'no': 5,
    'yes': 6,
    'reduced': 7,
    'normal': 8,
    'no lenses': 0,
    'hard': 1,
    'soft': 2
}

# 数据准备
X = []
Y = []
for i in range(len(MYDATA)):
    temp_x = []
    temp_x.append(mapper[MYDATA[i][0]])
    temp_x.append(mapper[MYDATA[i][1]])
    temp_x.append(mapper[MYDATA[i][2]])
    temp_x.append(mapper[MYDATA[i][3]])
    X.append(temp_x)
    Y.append(mapper[MYDATA[i][4]])

X_train, X_val, y_train, y_val = train_test_split(X, Y, test_size=0.2)

# # 决策树最大深度为10
tree_clf = DecisionTreeClassifier(max_depth=10, criterion='entropy')
tree_clf.fit(X_train, y_train)
print("决策树训练完成")

# # 保存决策树模型为dot文件以至于可视化

export_graphviz(
    tree_clf,
    out_file=os.path.join(".", "iris_tree.dot"),
    feature_names=MYLABEL,
    class_names=["no lenses", "hard", "soft"],
    rounded=True,
    filled=True
)
print("请在当前目录执行 dot -Tpng iris_tree.dot -o iris_tree.png")
# Source.from_file(os.path.join(".", "iris_tree.dot"))
# # dot -Tpng iris_tree.dot -o iris_tree.png

# '''
# Scikit-Learn使用的是CART算法，该算法仅生成二叉树：非叶节点永远只有两个
# 子节点（即问题答案仅有是或否）。但是，其他算法（比如ID3生成的决策树），其节点
# 可以拥有两个以上的子节点
# '''
# plt.figure(figsize=(8, 4))
# plot_decision_boundary(tree_clf, X, y)
# plt.plot([2.45, 2.45], [0, 3], "k-", linewidth=2)
# plt.plot([2.45, 7.5], [1.75, 1.75], "k--", linewidth=2)
# plt.plot([4.95, 4.95], [0, 1.75], "k:", linewidth=2)
# plt.plot([4.85, 4.85], [1.75, 3], "k:", linewidth=2)
# plt.text(1.40, 1.0, "Depth=0", fontsize=15)
# plt.text(3.2, 1.80, "Depth=1", fontsize=13)
# plt.text(4.05, 0.5, "(Depth=2)", fontsize=11)
# plt.show()


# # 使用模型进行预测
# print(tree_clf.predict_proba([[1.2, 13]]))
# print(iris.target_names[tree_clf.predict([[1.2, 13]])])
# '''
# [[1. 0. 0.]] 各个目标的概率
# ['setosa']
# '''


# # 正则化超参数
# # 可降低树的深度放置过拟合 max_depth
# # 分裂前节点必须有的最小样本数min_samples_split
# # min_samples_leaf（叶节点必须有
# # 的最小样本数量）、min_weight_fraction_leaf（与min_samples_leaf一样，但表现为加权实
# # 例总数的占比）、max_leaf_nodes（最大叶节点数量），以及max_features（分裂每个节点
# # 评估的最大特征数量）。增大超参数min_*或减小max_*将使模型正则化
# Xm, ym = make_moons(n_samples=100, noise=0.25, random_state=53)
# print("Xm:\n", Xm)
# print("ym:\n", ym)
# deep_tree_clf1 = DecisionTreeClassifier(random_state=42)
# deep_tree_clf2 = DecisionTreeClassifier(min_samples_leaf=4, random_state=42)
# deep_tree_clf1.fit(Xm, ym)
# deep_tree_clf2.fit(Xm, ym)

# fig, axes = plt.subplots(ncols=2, figsize=(10, 4), sharey=True)
# plt.sca(axes[0])
# plot_decision_boundary(deep_tree_clf1, Xm, ym,
#                        axes=[-1.5, 2.4, -1, 1.5], iris=False)
# plt.title("No restrictions", fontsize=16)
# plt.sca(axes[1])
# plot_decision_boundary(deep_tree_clf2, Xm, ym,
#                        axes=[-1.5, 2.4, -1, 1.5], iris=False)
# plt.title("min_samples_leaf = {}".format(
#     deep_tree_clf2.min_samples_leaf), fontsize=14)
# plt.ylabel("")

# plt.show()
