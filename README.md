# 机器学习与深度学习 

tensorflow\scikit-learn 机器学习与深度学习 代码阅读与整理后的学习例程  

## 快速上手  

 相关依赖： python3.6+ 、sklearn、tensorflow、matplotlib、numpy、pandas以及jupyter查看工具等,请根据例程进行依赖包的安装自己所需。  
 寻找相应的知识 您可以在例程中找到有用的代码、加以修改运用到您的项目或者进行学习。
 
## 机器学习   

### `基础知识`
* [数据集与测试集](./基础知识/准备.ipynb)  
* [csv文件使用](./基础知识/准备.ipynb)

### `分类`
  * [二分类](./分类/BinaryClassifier.py) => SDG手写数字二分类器  
  * [模型评估](./分类) => PR、ROC、混淆矩阵、F1、精准率、召回率   
  * [多分类](./分类/MultiouputClassification.py)  

### `回归` 
  * `线性回归` =>LinearRegression、最小二乘法、梯度下降、SDG、多项式回归、学习曲线、岭回归、Lasso回归
  * `逻辑回归` =>单特征、多特征 、Softmax多特征多分类 

### `支持向量机(SVM)`
  * `线性SVM分类` =>SVC、特征缩放  
  * `非线性SVM分类` =>PolynomialFeatures、LinearSVC、多项式内核、高斯RBF内核、计算复杂度
  * `SVM回归` =>LinearSVR、SVR、多阶 

### `决策树` 
  * `分类` =>DecisionTreeClassifier、决策树可视化(graphviz)、超参数正则 
  * `回归` =>DecisionTreeRegressor、模型效果可视化、正则化  

### `集成学习和随机森林`   
  * `投票分类器` => 硬投票、软投票  
  * `bagging 和 pasting` => BaggingClassifier、包外评估、随机补丁和随机子空间  
  * `随机森林` => RandomForestClassifier、极端随机树(ExtraTreesClassifier)、特征重要性  
  * `提升法` => AdaBoosting、GradientBoosting、早停止、xbgboost梯度提升早停止  
  * `堆叠法`  

### `降维`   
  * `基础` => 投影法、流形学习  
  * `PCA` => 主成分、解释方差比  
  * `选择维度` => 指定解释方差比、指定维度、PCA压缩  
  * `其他PCA` => 随机PCA、增量PCA、内核PCA  
  * `LLE` => LocallyLinearEmbedding 
  * `其他降维技术` => MDS、Isomap、t-SNE  

### `无监督学习`   
  * `聚类`  => 快速使用K-Means训练模型与预测、软聚类与硬聚类、寻找最佳K值、小批量Kmeans、KMeans图像分割、使用聚类对数据预处理、聚类半监督学习、DBSCAN（与KNN分类器）、其他聚类算法（谱聚类、层次聚类）   
  * `高斯混合模型` =>  高斯混合模型基本使用、异常值检测、模型选择调优、贝叶斯高斯混合模型  


## 深度学习   
### `人工神经网络ANN`   
  * `感知机` => 单层感知机 、多层感知机图像分类器 、MLP回归 
  * `函数API构建` => 非顺序网络(复杂自定义网络) 、TensorBoard的使用  
  * `微调神经网络超参数` => 随机调整
  
### `训练深度神经网络`  
  * `神经元权重初始化` => Glorot与He初始化  
  * `非饱和激活函数` => ReLU及其变体
  * `批量归一化` => BatchNormalization层  
  * `梯度裁剪`  clipvalue、clipnorm   
  * `重置预训练器` => 网络层重用、模型克隆、权重克隆  
  * `优化器选择与优化` => momentum超参数(动量优化)、nesterov加速梯度、AdaGrad、RMSProp、adam、Adamax、Nadam  
  * `学习率调度` => 幂调度、指数调度、分段常数调度、性能调度、1周期调度  
  * `正则化预防过拟合` => ℓ1 与 ℓ2 正则化、Dropout、Alpha Dropout、MC Dropout(蒙特卡洛方法)  
  
### `tensorflow 自定义模型训练`  
  * 像Numpy一样使用tf  => 张量定义、索引操作、张量运算、张量与numpy转换、字符串、不规则张量(Ragged tensors) 、稀疏张量(Sparse tensors)、集合、变量(tf.Variable)、张量数组

### `tensorflow 加载与预处理数据`  
### `卷积神经网络 CV`
### `RNN和CNN处理序列`  
### `RNN和注意力机制NLP`  
### `自动编码器、GAN表征学习、生成学习`  
### `强化学习`  
### `大规模训练和tensorflow模型部署`  


## 有趣的  
目录位置 /project/  
1. cvPy (Python OpenCV的常用函数封装)   
2. facedp_sgd (基于SGDClassifier的人脸检测与K-Means人脸图像分割)  
3. PCBCheck (电脑主板 螺丝装配 检测)  

  
## OpenCV

学习机器学习与深度学习、我认为最好的实践是做一些计算机视觉的小玩具。 关于我 

原来在参加RoboMaster(机甲大师)时接触到的OpenCV,项目是用C++构建的，当我接 
触到Python调用OpenCV时，总有些不适应，感觉我需要记住cv2内的function名，
但我们平时用到的其实并不多。  

在本项目opencvUtils/cvPy.py中，我们对cv2的接口进行了二次封装， 
对于专业使用 OpenCV来看，它是用不到的，如果您是一位从未接触过OpenCV的新手， 我想他会对你有用。  



## 关于我们  

一个菜鸡软件工程学子  
     怎么学起了机器学习与深度学习相关知识？  平时喜欢探究流行技术、范围广泛，涵盖工程化前端(React 等)、 后端JavaEE、SpringWeb、Node也有接触，但接触了一些时间，感觉它们都是前篇一律的业务， 我们很难做到像尤大神一样写出Vue,也很难像大神一样写出css框架，后端 的增删改查、以及各种框架之间的周转。感觉自己会成为编程工具人。  
     我第一次接触于机器学习相关的知识是利用OpenCV做视觉项目、 
发现机器学习与深度学习的用途非常广泛且有很好的未来，是在大学大二时接触到了RoboMaster比赛、但在项 
目的参与中、我们深知有些问题可以用ml、dl来解决,但我们有时也无能为力。 
我们学某种技术终究是用它。

也许学习机器学习与深度学习并不会成为我的工作、但我想利用在校的时间尽自己所能扩展自己的视野，这样才能发现自己的渺小，  
我们也不能评论学校课堂的知识没用、当来到机器学习的世界，会发现微积分、  
线性代数、概率论与数理统计、信息论等学科知识是如此重要、它确实很难，我们会感觉到自己渺小，原来知识可以运用到工作之中去、并充当不可缺失的一员。  

框架总有高光与落寞的时刻，最流行的不一定是最好的。我们解决的是问题、解决了问题就是好技术。  
正年轻，吃苦趁现在，祝君好运！


## 感谢
如果这个仓库对您有所帮助、还请您帮忙点个star,您的点赞与关注是我们持续分享的动力，欢迎Fork 贡献您的代码或者笔记到此仓库。  

Thanks a million!

## 声明
如果侵犯您的权益、请第一时间通过邮箱进行联系。🦜

## 联系我们
 📮 heizuboriyo@gmail.com  
 🏫  桂林电子科技大学(花江校区)  

## 参考书籍  
 * 《机器学习实战：基于Scikit-Learn、Keras和TensorFlow：第2版》  机械工业出版社 Aurelien Geron  (机器学习部分代码的主要来源出处)   
 * 《机器学习》 清华大学出版社 周志华   
 * 《深度学习》 人民邮电出版社  
 * 《Python编程 从入门到实践 第2版》 人民邮电出版社  