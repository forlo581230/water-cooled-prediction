import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
import moment

import os
from os import listdir

from waterChiller import WaterChiller
import csv
import math
import argparse


def main():
    '''
    將資料降維
    for 故障預測用
    '''
    parser = argparse.ArgumentParser()

    # BP parameter
    parser.add_argument('--bp', default=True,
                        help='True : BP AIR CONDITIONER , False: AIR CONDITIONER')

    args = parser.parse_args()

    transform_data(args, "datasets/hotelling")
    reduce_data("datasets/copData")

def transform_data(args, dirName):

    wc = WaterChiller()
    
    cops = []
    outputDatas = []
    PP=[]

    files = listdir(dirName)
    files = sorted(files)

    save_dir = os.path.join("datasets", "copData")
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    for fileName in files:
        print("loading...")
        df = pd.read_csv(os.path.join(dirName, fileName, fileName+'.csv'), encoding='utf-8')
        titles = np.array(df.columns)
        datas = df.iloc[:, :].to_numpy(dtype=float)
        n = datas.shape[0]
        if n == 0:
            continue

        row = 0
        while row < n:
            outputData = np.zeros((9))
            Tevwi = datas[row, 1]
            Tcdwi = datas[row, 3]
            operation_num = datas[row, -2]
            running_time = datas[row, -1]

            if args.bp:
                Pev = datas[row, 15]
                Pcd = datas[row, 18]
                Tevo = datas[row, 21]
                Thg = datas[row, 16]
                Tcdo = datas[row, 20]
                freq = datas[row, 13]
                W = datas[row, 19]
            else:
                Pev = datas[row, 6]
                Pcd = datas[row, 9]
                Tevo = datas[row, 12]
                Thg = datas[row, 7]
                Tcdo = datas[row, 11]
                freq = 60
                W = datas[row, 10]
                

            if freq == 0:
                print(moment.unix(datas[row, 0]),"=")

            h1, h2, h3, h4 = wc.getEnthalpy(Pcd, Pev, Tevo, Thg, Tcdo)

            outputData[0] = datas[row, 0]
            outputData[1] = h1/1000
            outputData[2] = h2/1000
            outputData[3] = h3/1000
            outputData[4] = (h1-h3)/(h2-h1)
            outputData[5] = Tevwi
            outputData[6] = Tcdwi
            outputData[7] = operation_num
            outputData[8] = running_time

            outputDatas.append(outputData)
            row += 1

        outputDatas = np.array(outputDatas)
        print("outputDatas : ", outputDatas.shape)

        # titles = np.array(df.columns)
        # titles[-1] = "cop"
        '''
        "Tevo"  :蒸發出口溫度
        "Tsh"   :過熱度
        "Thg"   :吐出溫度
        "Tcdo"  :冷凝出口溫度
        "Tsc"   :過冷度
        "Tca"   :冷卻水溫差
        "Tea"   :冷水溫差
        '''


        titles = np.array(["time", "h1", "h2", "h3", "C.O.P", "Tevwi", "Tcdwi", "operation_num", "running_time"])

        saveCSV(outputDatas, titles, fileName.split('.')[0])
        drawDiagram(outputDatas, titles, fileName.split('.')[0])
        
        # window_size = 0
        # numDatas = (outputDatas.shape[0])
        
        # train_size = (int)(numDatas*0.8)
        # valid_size = (int)(numDatas*0.9)
        # print("data_size = {}, train_size = {}, valid_size = {}".format(numDatas, train_size, valid_size-train_size))
        # saveCSV(outputDatas[:train_size],titles, fileName.split('.')[0]+"_v2train")
        # saveCSV(outputDatas[train_size-window_size:valid_size],titles, fileName.split('.')[0]+"_v2validation")
        # saveCSV(outputDatas[train_size-window_size:],titles, fileName.split('.')[0]+"_v2test")
        
        
    
def reduce_data(dirName):
    files = listdir(dirName)
 
    for fileName in files:
        
        print(fileName)
        df = pd.read_csv(os.path.join(dirName, fileName), encoding='utf-8')
        titles = np.array(df.columns)
        datas = df.iloc[:, :].to_numpy(dtype=float)


        outputDatas = []
        
        h1 = datas[:,1]
        h2 = datas[:,2]
        h4 = datas[:,3]
        temp = datas[:,5]


        t2 = 0
        while t2+1 < datas.shape[0]:
            timestamp1 = datas[t2, 0]
            timestamp2 = datas[t2 + 1,0]
            if timestamp2 - timestamp1 < 245:
                # avg_h1 = (h1[t2] + h1[t2+1])/2
                # avg_h2 = (h2[t2] + h2[t2+1])/2
                # avg_h4 = (h4[t2] + h4[t2+1])/2
                # cop = (avg_h1-avg_h4)/(avg_h2-avg_h1)
                # outputDatas.append([timestamp2, avg_h1, avg_h2, avg_h4, cop, temp[t2+1]])
                outputDatas.append(datas[t2])
            # else:
            #     outputDatas.append([timestamp2, h1[t2+1], h2[t2+1], h4[t2+1], datas[t2+1,4], temp[t2+1]])
            
            t2 = t2 + 1

        outputDatas = np.array(outputDatas)
        # outputDatas = datas

    

        saveCSV(outputDatas, titles,fileName.split('.')[0]+"_v2")
        drawDiagram(outputDatas, titles, fileName.split('.')[0])

        # window_size = 40
        # numDatas = (outputDatas.shape[0])
        
        # train_size = (int)(numDatas*0.8)
        # valid_size = (int)(numDatas*0.9)
        # print("data_size = {}, train_size = {}, valid_size = {}".format(numDatas, train_size, valid_size-train_size))
        # saveCSV(outputDatas[:train_size],titles, fileName.split('.')[0]+"_v2train")
        # saveCSV(outputDatas[train_size-window_size:valid_size],titles, fileName.split('.')[0]+"_v2validation")
        # saveCSV(outputDatas[train_size-window_size:],titles, fileName.split('.')[0]+"_v2test")
        


def getFiles(dirname):
    fileNames = listdir(dirname)
    fileNames = sorted(fileNames)
    return fileNames

# 1 kpa = 0.0102 kgf/cm^2


def pressureTranslate(p):
    # kgf/cm^2 to pa
    return p * (1/0.0102)*1000


def saveCSV(datas, titles, fileName):
    # 開啟輸出的 CSV 檔案
    with open('datasets/copData/'+fileName+'.csv', 'w', encoding='utf-8', newline='') as csvfile:
        # 建立 CSV 檔寫入器
        writer = csv.writer(csvfile)
        # 寫入資料
        writer.writerow(titles)
        for data in datas:
            writer.writerow(data)


def drawDiagram(outputDatas, titles, folderName):
    print(outputDatas.shape)

    fig, axs = plt.subplots(titles.shape[0]-1, 1)
    

    for i in range(titles.shape[0]-1):
        axs[i].plot(outputDatas[:, i+1], label=titles[i+1])

    # idx=0

    # for m in range(2,7):
    #     for i in range(outputDatas.shape[0]):
    #         if moment.unix(outputDatas[i, 0]).month== m:
    #             idx=i
    #             axs[-1].plot([idx, idx],[7,10],color='r')
    #             break
    

    for ax in axs:
        ax.legend()
        ax.grid()
    fig.canvas.set_window_title(folderName+'變頻')
    plt.subplots_adjust(hspace=0.5)
    plt.show()



if __name__ == "__main__":
    main()