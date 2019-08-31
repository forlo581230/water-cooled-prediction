import json
import os
import pandas as pd
import time
import moment

import numpy as np

def main():
    build_location_weather()
    # translate_unix()
    return


def insert_temperature(dataset):
    
    load_dir = "./Banqiao.csv"
    df = pd.read_csv(load_dir, sep=",", encoding="utf8")
    temp_data = df.iloc[:].values

    inserted_data = []

    temp_idx = 0

    print("len", dataset.shape[0])
    for i in range(dataset.shape[0]):
        timestamp_dataset = dataset[i,0]

        isSetTemp = False
        while not isSetTemp:
            timeArray = time.strptime(temp_data[temp_idx,0], "%Y-%m-%d %H:%M")
            timestamp_temp = int(time.mktime(timeArray))
            timeArray = time.strptime(temp_data[temp_idx+1,0], "%Y-%m-%d %H:%M")
            timestamp2_temp = int(time.mktime(timeArray))

            if timestamp_temp <= timestamp_dataset and timestamp_dataset<timestamp2_temp:
                inserted_data.append([dataset[i,0], dataset[i,1], dataset[i,2], dataset[i,3], dataset[i,4], temp_data[temp_idx,1]])
                isSetTemp = True
            else:
                print(timestamp_temp, timestamp_dataset, timestamp_dataset-timestamp_temp)
                temp_idx += 1
                # print("1")
        
    return np.array(inserted_data)


def translate_unix():
    dataset = "./Banqiao.csv"

    df = pd.read_csv(dataset, sep=",", encoding="utf8")
    
    data = df.iloc[:].values
    print(data[0,0])
    print((moment.utc(data[0,0])))
    timeArray = time.strptime(data[0,0], "%Y-%m-%d %H:%M")
    timeStamp = int(time.mktime(timeArray))
    print(timeStamp)

    timeArray = time.localtime(timeStamp)
    otherStyleTime = time.strftime("%Y-%m-%d %H:%M", timeArray)
    print(otherStyleTime)


def build_location_weather():
    dataset = "./C-B0024-002.json"

    time_list =[]
    temp_list = []
    m_list = []
    with open(dataset, 'r', encoding="utf-8") as f:
        json_data = json.loads(f.read())
        
        data = json_data["cwbopendata"]["dataset"]["location"][0]
        print(data["locationName"])
        timeStr = ""
        for location_data in data["weatherElement"][0]["time"]:
            
            print(location_data["obsTime"])
            try:
                timeArray = time.strptime(location_data["obsTime"], "%Y-%m-%d %H:%M")
                timestamp_temp = int(time.mktime(timeArray))
                timeStr = location_data["obsTime"]
            except:
                timestamp_temp += 3600
                timeStr = (moment.unix(timestamp_temp).format('YYYY-M-D HH:m'))
                pass


            if timestamp_temp >= 1543593600:
                time_list.append(timeStr)
                temp_list.append(location_data["weatherElement"][1]["elementValue"]["value"])
                m_list.append(location_data["weatherElement"][2]["elementValue"]["value"])
            # print(location_data["obsTime"], location_data["weatherElement"][1]["elementValue"]["value"], location_data["weatherElement"][2]["elementValue"]["value"])

        new_df = pd.DataFrame({'time':time_list, 'temp':temp_list, 'humidity':m_list})
        new_df.to_csv("Banqiao.csv",index=False)



    return


if __name__ == "__main__":
    main()