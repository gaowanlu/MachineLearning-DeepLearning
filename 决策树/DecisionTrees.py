#训练与可视化决策树  
'''
决策树可视化软件
http://wwwgraphviz.org/ 
用Graphviz软件包中的dot命令行工具将此.dot文件转换为多种格式，
例如PDF或PNG[1]。此命令行将.dot文件转换为.png图像文件：  
dot -Tpng iris_tree.dot -o iris_tree.png
'''

from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier

#数据准备
iris=load_iris()
X=iris.data[:,2:]#长与宽两个特征 
y=iris.target

#决策树最大深度为2
tree_clf=DecisionTreeClassifier(max_depth=2)
tree_clf.fit(X,y)

#保存决策树模型为dot文件以至于可视化
import os
from sklearn.tree import export_graphviz
from graphviz import Source

export_graphviz(
    tree_clf,
    out_file=os.path.join(".", "iris_tree.dot"),
    feature_names=iris.feature_names[2:],#"length" "width"
    class_names=iris.target_names,#['setosa' 'versicolor' 'virginica']
    rounded=True,
    filled=True
)
Source.from_file(os.path.join(".", "iris_tree.dot"))
#dot -Tpng iris_tree.dot -o iris_tree.png

'''
Scikit-Learn使用的是CART算法，该算法仅生成二叉树：非叶节点永远只有两个
子节点（即问题答案仅有是或否）。但是，其他算法（比如ID3生成的决策树），其节点
可以拥有两个以上的子节点
'''
# from matplotlib.colors import ListedColormap
# import matplotlib.pyplot as plt
# import numpy as np
# def plot_decision_boundary(clf, X, y, axes=[0, 7.5, 0, 3], iris=True, legend=False, plot_training=True):
#     x1s = np.linspace(axes[0], axes[1], 100)
#     x2s = np.linspace(axes[2], axes[3], 100)
#     x1, x2 = np.meshgrid(x1s, x2s)
#     X_new = np.c_[x1.ravel(), x2.ravel()]
#     y_pred = clf.predict(X_new).reshape(x1.shape)
#     custom_cmap = ListedColormap(['#fafab0','#9898ff','#a0faa0'])
#     plt.contourf(x1, x2, y_pred, alpha=0.3, cmap=custom_cmap)
#     if not iris:
#         custom_cmap2 = ListedColormap(['#7d7d58','#4c4c7f','#507d50'])
#         plt.contour(x1, x2, y_pred, cmap=custom_cmap2, alpha=0.8)
#     if plot_training:
#         plt.plot(X[:, 0][y==0], X[:, 1][y==0], "yo", label="Iris setosa")
#         plt.plot(X[:, 0][y==1], X[:, 1][y==1], "bs", label="Iris versicolor")
#         plt.plot(X[:, 0][y==2], X[:, 1][y==2], "g^", label="Iris virginica")
#         plt.axis(axes)
#     if iris:
#         plt.xlabel("Petal length", fontsize=14)
#         plt.ylabel("Petal width", fontsize=14)
#     else:
#         plt.xlabel(r"$x_1$", fontsize=18)
#         plt.ylabel(r"$x_2$", fontsize=18, rotation=0)
#     if legend:
#         plt.legend(loc="lower right", fontsize=14)

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
print(tree_clf.predict_proba([[1.2,13]]))
print(iris.target_names[tree_clf.predict([[1.2,13]])])
