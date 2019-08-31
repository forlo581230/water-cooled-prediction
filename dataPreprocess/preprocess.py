
import pandas as pd
import numpy as np
import math
import random
import matplotlib.pyplot as plt
import csv

def Gaussianfilter(y):
    r=2
    sigma=1
    gaussTemp = np.zeros((r*2-1, 1))
    for i in range(1,r*2):
        gaussTemp[i-1]= math.exp(-(i-r)**2/(2*sigma**2))/(sigma*math.sqrt(2*math.pi))
    
    y_filted = np.copy(y)
    for i in range(r, y.shape[1]-r):
        # print(y[:,i-r+1:i+r], gaussTemp)
        y_filted[:,i-1:i] = np.matmul(y[:,i-r+1:i+r], gaussTemp)

    return y_filted
        
def saveCSV(fileName, datas, titles):
    # 開啟輸出的 CSV 檔案
    with open('datasets/copData/'+fileName, 'w', newline='') as csvfile:
        # 建立 CSV 檔寫入器
        writer = csv.writer(csvfile)
        # 寫入資料
        writer.writerow(titles)
        for data in datas:
            writer.writerow(data)


def main():
    directory = "datasets/copData/013157800087578.csv"
    df = pd.read_csv(directory)

    all_datas = np.array(df.iloc[:, :].values)
    datas = np.array(df.iloc[:, 1:].values)
    cops = np.array(df.iloc[:, -1:].values)
    y = datas.swapaxes(0,1)

    y_filted = Gaussianfilter(y)
    # y_filted = y_filted.swapaxes(0,1)

    y_filted = y_filted.swapaxes(0,1)


    all_datas[:, 1:]= y_filted[:, :]

    plt.plot(datas[:,-1], label="ori")
    plt.plot(all_datas[:,-1],  label="gauss")
    plt.legend()
    plt.show()

    # plt.plot(datas[:,-1], label="ori")
    # plt.plot((all_datas[:,1]-all_datas[:,3])/(all_datas[:,2]-all_datas[:,1]),  label="gauss")
    # plt.legend()
    # plt.show()

    titles = np.array(df.columns)
    saveCSV("gaussdata.csv", all_datas, titles)

    splitData(all_datas, titles)

def splitData(all_datas, titles):
    train =[]
    test =[]

    percent = 0.8
    data_num = all_datas.shape[0]

    train = all_datas[:(int)(data_num*0.8),:]
    test = all_datas[(int)(data_num*0.8):,:]
    

    for i in range(0, data_num):
        for j in range(0, data_num):
            if j == i:
                continue
            if all_datas[i,0] == all_datas[j,0]:
                print(all_datas[i,0])

    train = np.array(train)
    test = np.array(test)

    print(data_num, train.shape[0]+test.shape[0])
    saveCSV("gaussdata_train.csv", train, titles)
    saveCSV("gaussdata_test.csv", test, titles)



if __name__ == "__main__":
    main()