from pysofaconventions import *
import numpy as np
import pandas as pd
import os


##############################单个sofa文件数据读取######################################
def sofa_reloader(path, n, flag):
    sofa = SOFAFile(path, 'r')
    data = sofa.getDataIR()
    for i in range(1250):

        hrir = data[i, flag, :]  # hrir, 0是左耳, 1是右耳
        HRTF = np.fft.fft(hrir)
        HRTF = 20 * np.log10(np.abs(HRTF))  # 相频预测同理
        for o in range(100):     # 取正频率
            frequency = np.array(44.1 / 200 * (o + 1))   # CIPIC中HRTF采样率为44.1kHZ
            HRTF_frequency = HRTF[o]
            mixed_data = pd.DataFrame(
                np.hstack((anth[n].reshape(1, 25), angel[i, :].reshape(1, 2), frequency.reshape(1, 1),  HRTF_frequency.reshape(1, 1))))
            mixed_data.to_csv('data/CIPIC_LeftHRTF频点.csv', mode='a', header=None)


#####先存左耳数据####
left_anth_data = pd.read_csv('data/CIPIC_measure_left.csv', header=None)
left_anth = left_anth_data.iloc[1:, 1:26]
left_anth = left_anth.values

anth = left_anth

angel = np.zeros((1250, 2))
azimuth = [-80, -65, -55, -45, -40,
           -35, -30, -25, -20, -15,
           -10, -5, 0, 5, 10,
           15, 20, 25, 30, 35,
           40, 45, 55, 65, 80]
azimuth = np.array(azimuth)
elevation = [-45 + 5.625 * i for i in range(50)]
for i in range(1250):
    m = i // 50
    n = i % 50
    angel[i, 0] = azimuth[m]
    angel[i, 1] = elevation[n]

base = 'sofa'
path_list = os.listdir(base)
path_list.sort(key=lambda x: int(x.split('subject_')[1].split('.sofa')[0]))
n = 0
for f in path_list:
    fullname = os.path.join(base, f)
    sofa_reloader(fullname, n, 0)
    n = n + 1
    print('iterations number:{0}'.format(n))



print('左耳数据计算完毕')
