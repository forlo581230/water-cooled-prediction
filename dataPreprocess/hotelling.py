import math
import numpy as np
import os
from os import listdir
import pandas as pd
import csv
import matplotlib.pyplot as plt
from numpy.linalg import inv
import argparse




def main():
    parser = argparse.ArgumentParser()

    # BP parameter
    parser.add_argument('--bp', default=True,
                        help='True : BP AIR CONDITIONER , False: AIR CONDITIONER')

    args = parser.parse_args()

    hotellingFilter(args,"datasets/usefulData")


    
def hotellingFilter(args,dirName):
    folders = listdir(dirName)
    
    for folderName in folders:
        dataset = []
        files = listdir(os.path.join(dirName, folderName))
        files = sorted(files)

        save_dir = os.path.join("datasets", "hotelling_before",folderName)
        save_dir2 = os.path.join("datasets", "hotelling",folderName)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        if not os.path.exists(save_dir2):
            os.makedirs(save_dir2)

        for fileName in files:
            df = pd.read_csv(os.path.join(dirName, folderName, fileName), encoding='utf-8')
            titles = np.array(df.columns)
            datas = df.iloc[:, :].to_numpy(dtype=float)

            if datas.shape[0] == 0:
                continue

            for row in range(datas.shape[0]):
                dataset.append(datas[row, :])

        dataset = np.array(dataset) 
        saveCSV(save_dir, folderName, dataset, titles)
        dataset = hotellingTsquared(args, dataset)
        saveCSV(save_dir2, folderName, dataset, titles)

def saveCSV(save_dir, fileName, datas, titles):
    # 開啟輸出的 CSV 檔案
    with open(os.path.join(save_dir, fileName+'.csv'), 'w', encoding='utf-8', newline='') as csvfile:
        # 建立 CSV 檔寫入器
        writer = csv.writer(csvfile)
        # 寫入資料
        writer.writerow(titles)
        for data in datas:
            writer.writerow(data)


def hotellingTsquared(args, dataset):
    if args.bp:
        X = dataset[:9000,[1,2,3,4,15,16,18,19,20,21]]
        X2 = dataset[:,[1,2,3,4,15,16,18,19,20,21]]
    else:
        X = dataset[:,[1,2,3,4,6,7,9,10,11,12]]
        X2 = dataset[:,[1,2,3,4,6,7,9,10,11,12]]

    X = X.T
    X2 = X2.T
    p = X.shape[0]
    n = X.shape[1]

    X_mean = np.mean(X, axis=1)
    X_mean = np.reshape(X_mean,(p,1))


    sx = np.zeros((p,p))
    for i in range(n):
        row_vector = X[:,i:i+1]-X_mean
        sx = sx + np.dot(row_vector, row_vector.T)
    sx = sx/(n-1)

    print(X_mean.shape)

    T_value = []
    for i in range(X2.shape[1]):
        row_v = X2[:,i:i+1]- X_mean
        value = np.dot(np.dot(row_v.T,inv(sx)), row_v)
        T_value.append(value[0,0])
    

    # alpha = 0.05
    # F = 1.8307
    # alpha = 0.01
    F = 2.321
    F = F*(p*(n-1))/(n-p)

    print("F : ",F)
    

    count = 0
    newDataset = []
    print(len(T_value))
    for i in range(X2.shape[1]):
        if T_value[i] > F:
            count = count + 1
        else:
            newDataset.append(dataset[i,:])
    
    newDataset = np.array(newDataset)
    print(count,"items can be deleted.")



    if args.bp:
        print("Pev:",np.mean(newDataset[:,15], axis=0))
        print("Pcd:",np.mean(newDataset[:,18], axis=0))
        print("Tevo:",np.mean(newDataset[:,21], axis=0))
        print("Thg:",np.mean(newDataset[:,16], axis=0))
        print("Tcdo:",np.mean(newDataset[:,20], axis=0))

        print("Pev:",np.std(newDataset[:,15], axis=0))
        print("Pcd:",np.std(newDataset[:,18], axis=0))
        print("Tevo:",np.std(newDataset[:,21], axis=0))
        print("Thg:",np.std(newDataset[:,16], axis=0))
        print("Tcdo:",np.std(newDataset[:,20], axis=0))
    else:
        print("Pev:",np.mean(newDataset[:,6], axis=0))
        print("Pcd:",np.mean(newDataset[:,9], axis=0))
        print("Tevo:",np.mean(newDataset[:,12], axis=0))
        print("Thg:",np.mean(newDataset[:,7], axis=0))
        print("Tcdo:",np.mean(newDataset[:,11], axis=0))
    plt.plot(T_value)
    plt.plot(list(range(n)),[F]*n)
    plt.show()

    return newDataset


if "__main__" == __name__:
    main()