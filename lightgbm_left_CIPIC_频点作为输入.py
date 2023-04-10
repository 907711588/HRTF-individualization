import lightgbm as lgb
import warnings
warnings.filterwarnings('ignore')
import pandas as pd
import numpy as np
import time
from sklearn.model_selection import train_test_split
import joblib


# 定义谱失真函数
def spectrum_distortion(original, approx):
    sum = 0
    for t in range(len(original)):
        sum += (original[t] - approx[t]) ** 2
    sum = sum / len(original)
    sum = sum ** 0.5
    return sum


# 要调用的数据文件
SD_data = []
y_pre_data = []
y_ori_data = []
y_train_pre_data = []
y_train_ori_data = []

CIPIC_data = pd.read_csv('data/CIPIC_LeftHRTF频点.csv', header=None)
x_l = CIPIC_data.iloc[:, 1:29]
X_l = x_l.values

y_l = CIPIC_data.iloc[:, 29:30]
Y_l = y_l.values


# X_train, X_test, Y_train, Y_test = train_test_split(X_l, Y_l, test_size=0.2, random_state=2022)

X_train = X_l[:3750000, :] #30×1250*100，CIPIC数据库共37个人，取30个人作为训练集
X_test = X_l[3750000:, :]

Y_train = Y_l[:3750000, :]
Y_test = Y_l[3750000:, :]

start_time = time.time()
# 采用LGBMRegressor函数做预测
regressor = lgb.LGBMRegressor(n_estimators=10000, objective='regression', learning_rate=0.1)

# 拟合残差树
regressor.fit(X_train, Y_train, eval_set=[(X_test, Y_test)], verbose=1, eval_metric='l1', early_stopping_rounds=5000)
# 存储训练好的模型
save_model_path = 'model/测试.pkl'
joblib.dump(regressor, save_model_path)
end_time = time.time()
print('Time to load: {0} sec   '.format(int(end_time - start_time)))
