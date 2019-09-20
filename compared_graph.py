# import pickle
# import csv

# with open("caffe2_results.csv", mode = "w") as csvfile :
#     csv_data=csv.writer(csvfile)
#     csv_data.writerow(tmp)


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_pickle("caffe2_results.pkl")
# print (df)
tmp = []
for key, val in df.items():
    tmp.append(val) 
    for i, v in enumerate(val):
        print(v)
print(tmp[0]) 
print(tmp[1])   
 
left = np.arange(len(tmp[0]))  # numpyで横軸を設定
labels = ['vgg16 eval','vgg16 train','resnet152 eval','resnet152 train','densenet161 eval','densenet161 train']

height = 0.3
 
plt.barh(left, tmp[0], color='r', height=height, label='fp32', align='center')
plt.barh(left+height, tmp[1], color='b', height=height, label='fp16', align='center')

plt.legend(loc="best")

plt.title("This is a title")
plt.xlabel("x axis")
plt.ylabel("y axis")
plt.grid(True)

plt.yticks(left + height/2, labels)
# plt.savefig('')
plt.show()