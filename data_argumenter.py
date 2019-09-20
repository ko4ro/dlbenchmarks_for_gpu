import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import sys

def makedir(path):
    if not os.path.exists(path):
        os.mkdir(path)

def parser():
    usage = 'Usage: python {} FILENAME'\
            .format(__file__)
    arguments = sys.argv
    if len(arguments) == 1:
        return usage
    # ファイル自身を指す最初の引数を除去
    arguments.pop(0)
    # 引数として与えられたfile名
    
    df_array = []
    frame_name = []
    for arg in arguments:
        data = pd.read_pickle("{}".format(arg))

        ## PLTによるグラフ描画

        tmp = []
        for key, val in data.items():
            tmp.append(val) 
        left = np.arange(len(tmp[0]))  # numpyで横軸を設定
        labels = ['vgg16 eval','vgg16 train','resnet152 eval','resnet152 train','densenet161 eval','densenet161 train']
        height = 0.3
        
        plt.barh(left, tmp[0], color='r', height=height, label='fp32', align='center')
        plt.barh(left+height, tmp[1], color='b', height=height, label='fp16', align='center')

        plt.legend(loc="best")
        plt.title("{}".format(os.path.splitext(arg)[0]))
        plt.xlabel("imgs/sec",fontsize=18)
        

        plt.yticks(left + height/2, labels)
        makedir('./graph')
        plt.savefig('./graph/{}.png'.format(os.path.splitext(arg)[0]),dpi=200, bbox_inches="tight")
        # plt.show()   
     

        ## CSVのファイル作成

        df = pd.DataFrame(data,index=['vgg16 eval','vgg16 train','resnet152 eval','resnet152 train','densenet161 eval','densenet161 train'])
        df = df.rename(columns={'fp32':'32-bit','fp16':'16-bit'})
        df_T = df.T
        # print(df_T)
        df_array.append(df_T)
        other_ext_fname = os.path.splitext(arg)[0] + '.csv'
        frame_name.append(os.path.splitext(arg)[0])
        # print('{}'.format(os.path.splitext(arg)[0]))
        
        makedir('./csv')
        df_T.to_csv("./csv/{}".format(other_ext_fname))

        ### 小数点以下第2位までの表示方法

        #     print(df_T.round(2))
        #     other_ext_fname = os.path.splitext(fname)[0] + '.csv'
        #     df_T.round(2).to_csv("{}".format(other_ext_fname),header='{}'.format(os.path.splitext(fname)[0]))

    df_concat = pd.concat(df_array,axis=0,keys=frame_name)
    print(df_concat)
    df_concat.to_csv("./csv/all_results.csv")

        

if __name__== '__main__':
    parser()
    
