import pandas as pd
LINE_EXT_PATH = "./line-ext.csv"


def load_data(path):
    return pd.read_csv(path)


line_data = load_data(LINE_EXT_PATH)
# line_data.head()
# line_data.info()
x_train = line_data["YearsExperience"].values
y_label = line_data["Salary"].values

# 求解w


def calc_w():
    sum = 0
    for i in range(len(x_train)):
        sum += x_train[i]
    aver = sum/len(x_train)
    top_value = 0
    for i in range(len(x_train)):
        top_value += y_label[i]*(x_train[i]-aver)
    bottom_value_1 = 0
    for i in range(len(x_train)):
        bottom_value_1 += x_train[i]*x_train[i]
    bottom_value_2 = (sum*sum)/len(x_train)
    return top_value/(bottom_value_1-bottom_value_2)


w = calc_w()
print("w^", w)
# 求解b


def calc_b():
    sum = 0
    for i in range(len(x_train)):
        sum += (y_label[i]-w*x_train[i])
    return sum/len(x_train)


b = calc_b()
print("b^", b)
print("so math expression is y= {} * x +( {} )".format(w, b))


def mymodel(x):
    return w*x+b


print("now x=0.8452 predict y={}".format(mymodel(0.8452)))
