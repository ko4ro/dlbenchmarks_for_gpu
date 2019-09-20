import pandas as pd
import numpy as np
import os
import sys

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
        df = pd.DataFrame(data,index=['vgg16 eval','vgg16 train','resnet152 eval','resnet152 train','densenet161 eval','densenet161 train'])
        df = df.rename(columns={'fp32':'32-bit','fp16':'16-bit'})
        df_T = df.T
        print(df_T)
        df_array.append(df_T)
        other_ext_fname = os.path.splitext(arg)[0] + '.csv'
        frame_name.append(os.path.splitext(arg)[0])
        # print('{}'.format(os.path.splitext(arg)[0]))

        df_T.to_csv("{}".format(other_ext_fname))

    df_concat = pd.concat(df_array,axis=0,keys=frame_name)
    print(df_concat)
    df_concat.to_csv("all_results.csv")

if __name__== '__main__':
    parser()
    
