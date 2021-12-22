import tensorflow as tf 
from tensorflow import keras
import matplotlib.pyplot as plt

#数据准备 
fashion_mnist=keras.datasets.fashion_mnist 
(X_train_full,y_train_full),(X_test,y_test)=fashion_mnist.load_data() 

#训练集与验证集
X_valid, X_train = X_train_full[:5000] / 255.0, X_train_full[5000:] / 255.0  # 归一化
y_valid, y_train = y_train_full[:5000], y_train_full[5000:] 

class_names = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",\
"Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]  


print (class_names[y_train[0]])
plt.imshow(X_train[0], cmap="binary")
plt.axis('off')
plt.show()


n_rows = 4
n_cols = 10
plt.figure(figsize=(n_cols * 1.2, n_rows * 1.2))
for row in range(n_rows):
    for col in range(n_cols):
        index = n_cols * row + col
        plt.subplot(n_rows, n_cols, index + 1)
        plt.imshow(X_train[index], cmap="binary", interpolation="nearest")
        plt.axis('off')
        plt.title(class_names[y_train[index]], fontsize=12)
plt.subplots_adjust(wspace=0.2, hspace=0.5)
plt.show()