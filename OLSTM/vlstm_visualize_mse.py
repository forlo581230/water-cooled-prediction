import numpy as np
import torch
from torch.autograd import Variable

import os
import random
import matplotlib
import matplotlib.animation as animation
import itertools
# matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from random import randint
from random import choice
from textwrap import wrap
from cycler import cycler
from random import shuffle
import matplotlib as mpl
from adjustText import adjust_text
import math


import pickle
#from graphviz import Digraph
from torch.autograd import Variable
import argparse
from helper import get_all_file_names

import pandas as pd
from sklearn.preprocessing import MinMaxScaler

def main():
    parser = argparse.ArgumentParser()

    # frame rate of video
    parser.add_argument('--frame', type=int, default=1,
                        help='Frame of video created from plots')
    # gru model
    parser.add_argument('--gru', action="store_true", default=False,
                        help='Visualization of GRU model')
    # number of validation dataset
    parser.add_argument('--num_of_data', type=int, default=0,
                        help='Number of validation data will be visualized (If 0 is given, will work on test data mode)')
    # drive support
    parser.add_argument('--drive', action="store_true", default=False,
                        help='Use Google drive or not')
    # minimum lenght of trajectory
    parser.add_argument('--min_traj', type=int,  default=3,
                        help='Min. treshold of number of frame to be removed from a sequence')
    # percentage of peds will be taken for each frame
    parser.add_argument('--max_ped_ratio', type=float,  default=0.8,
                        help='Percentage of pedestrian will be illustrated in a plot for a sequence')
    # maximum ped numbers
    parser.add_argument('--max_target_ped', type=int,  default=20,
                        help='Maximum number of peds in final plot')
    # method to be visualized
    parser.add_argument('--method', type=int, default=1,
                        help='Method of lstm will be used (1 = social lstm, 2 = obstacle lstm, 3 = vanilla lstm)')
    # Parse the parameters
    args = parser.parse_args()

    prefix = ''
    f_prefix = '.'


    method_name = "VANILLALSTM"
    model_name = "LSTM"
    if args.gru:
        model_name = "GRU"

    
    plot_file_directory = 'validation'

    # Directories
    if args.num_of_data is 0:
        plot_file_directory = 'test'

    # creation of paths
    save_plot_directory = os.path.join(f_prefix, 'plot',method_name, model_name,'plots/')
    plot_directory = os.path.join(f_prefix, 'plot', method_name, model_name, plot_file_directory)
    video_directory = os.path.join(f_prefix, 'plot',method_name, model_name,'videos/')
    plot_file_name = get_all_file_names(plot_directory)

    print(plot_file_name)
    for file_index in range(len(plot_file_name)):
        file_name = plot_file_name[file_index]
        print("Now processing: ", file_name)

        file_path = os.path.join(plot_directory, file_name)

        try:
            f = open(file_path, 'rb')
        except FileNotFoundError:
            print("File not found: %s"%file_path)
            continue

        results = pickle.load(f)
        obs_length = np.array(results[0][1])
        pred_length = np.array(results[0][0])
        enthalpy_max=[408,430,251]
        enthalpy_min=[404,422.5,239]
        enthalpy_label=['evaporator outlet enthalpy', 'compressor outlet enthalpy', 'condenser outlet enthalpy']


        '''
        draw enthalpy prediction diagram
        '''
        expected_enthalpy_arr = np.array(results[0][2])
        predicted_enthalpy_arr = np.array(results[0][3])
        error_enthalpy = np.zeros((expected_enthalpy_arr.shape[0],3))
        plt.figure(figsize=(10,5))

        for i in range(expected_enthalpy_arr.shape[0]):
            for idx in range(3):
                expected_enthalpy = expected_enthalpy_arr[i,:,idx:idx+1]
                predicted_enthalpy = predicted_enthalpy_arr[i,:,idx:idx+1]
                # predicted_enthalpy[obs_length:] = filter(predicted_enthalpy[obs_length:])
                mean_obs_enthalpy = np.mean(expected_enthalpy[:obs_length])
                mean_expected_enthalpy = np.mean(expected_enthalpy[obs_length:])
                mean_predicted_enthalpy = np.mean(predicted_enthalpy[obs_length:])

                # draw plot
                plt.ylim(enthalpy_min[idx],enthalpy_max[idx])
                plt.plot(np.arange(0,obs_length*2,2), expected_enthalpy[:obs_length], color='green', label="Observation data", linestyle='-', marker='.')
                plt.plot(np.arange(obs_length*2, expected_enthalpy.shape[0]*2,2), expected_enthalpy[obs_length:], label="Expected "+enthalpy_label[idx], linestyle='-', marker='.')
                plt.plot(np.arange(obs_length*2, expected_enthalpy.shape[0]*2,2), predicted_enthalpy[obs_length:], label="Predictive "+enthalpy_label[idx], linestyle='-', marker='+')
                plt.plot([0,obs_length*2], [mean_obs_enthalpy,mean_obs_enthalpy], color='black', label="Observation length mean "+enthalpy_label[idx])
                plt.plot([obs_length*2,expected_enthalpy.shape[0]*2], [mean_expected_enthalpy,mean_expected_enthalpy], color='gray', label="Expected mean "+enthalpy_label[idx])
                plt.plot([obs_length*2,expected_enthalpy.shape[0]*2], [mean_predicted_enthalpy,mean_predicted_enthalpy], color='red', label="Predictive mean "+enthalpy_label[idx])
                
                plt.xlabel('Time(minutes)', fontsize=12)
                plt.ylabel('Enthalpy(kJ/kg)', fontsize=12)
                plt.xticks(fontsize=12)
                plt.yticks(fontsize=12)
                plt.legend()
                plt.savefig(plot_directory+"/"+str(i)+"H"+str(idx+1)+".png")
                plt.clf()
                # plt.show()

                # compute error
                # sc = MinMaxScaler()
                # fit_data = np.concatenate((expected_enthalpy, predicted_enthalpy), axis=0)
                # fit_data = sc.fit_transform(fit_data)

                # expected_enthalpy = fit_data[:expected_enthalpy.shape[0]]
                # predicted_enthalpy = fit_data[expected_enthalpy.shape[0]:]
                

                # error = np.sum(np.abs(expected_enthalpy[obs_length:]-predicted_enthalpy[obs_length:]))/expected_enthalpy[obs_length:].shape[0]
                error = np.abs(mean_expected_enthalpy-mean_predicted_enthalpy)
                error = np.around(error, decimals=4)
                error_enthalpy[i,idx] = error

        new_df = pd.DataFrame({'h1 err':error_enthalpy[:,0], 'h2 err':error_enthalpy[:,1], 'h3 err':error_enthalpy[:,2]})
        new_df.to_csv(plot_directory+"/"+"enthalpy_error_minutes"+".csv",index=False)

                
            


        '''
        draw cop prediction diagram (minutes)
        '''
        error_cop = np.zeros((expected_enthalpy_arr.shape[0], 1))
        for i in range(expected_enthalpy_arr.shape[0]):
            expected_cop_arr = (expected_enthalpy_arr[i,:,0:1]-expected_enthalpy_arr[i,:,2:3])/(expected_enthalpy_arr[i,:,1:2]-expected_enthalpy_arr[i,:,0:1])
            predicted_cop_arr = (predicted_enthalpy_arr[i,:,0:1]-predicted_enthalpy_arr[i,:,2:3])/(predicted_enthalpy_arr[i,:,1:2]-predicted_enthalpy_arr[i,:,0:1])
            # predicted_cop_arr[obs_length:] = filter(predicted_cop_arr[obs_length:])
            mean_obs_cop = np.mean(expected_cop_arr[:obs_length])
            mean_expected_cop = np.mean(expected_cop_arr[obs_length:])
            mean_predicted_cop = np.mean(predicted_cop_arr[obs_length:])
            
            plt.ylim(6,10)
            plt.plot(np.arange(0,obs_length*2,2), expected_cop_arr[:obs_length], color='green', label="Observation data", linestyle='-', marker='.')
            plt.plot(np.arange(obs_length*2,expected_cop_arr.shape[0]*2,2), expected_cop_arr[obs_length:], label="Expected COP", linestyle='-', marker='.')
            plt.plot(np.arange(obs_length*2,expected_cop_arr.shape[0]*2,2), predicted_cop_arr[obs_length:], label="Predictive COP", linestyle='-', marker='+')
            plt.plot([0,obs_length*2], [mean_obs_cop,mean_obs_cop], color='black', label="Observation length mean COP")
            plt.plot([obs_length*2,expected_cop_arr.shape[0]*2], [mean_expected_cop,mean_expected_cop], color='gray', label="Expected mean COP")
            plt.plot([obs_length*2,expected_cop_arr.shape[0]*2], [mean_predicted_cop,mean_predicted_cop], color='red', label="Predictive mean COP")
            plt.xlabel('Time(minutes)', fontsize=12)
            plt.ylabel('COP', fontsize=12)
            plt.xticks(fontsize=12)
            plt.yticks(fontsize=12)
            plt.legend()

            plt.savefig(plot_directory+"/test"+str(i)+"_COP minutes.png")
            plt.clf()
            # plt.show()


            # compute error
            # sc = MinMaxScaler()
            # fit_data = np.concatenate((expected_cop_arr, predicted_cop_arr), axis=0)
            # fit_data = sc.fit_transform(fit_data)
            # expected_cop_arr = fit_data[:expected_cop_arr.shape[0]]
            # predicted_cop_arr = fit_data[expected_cop_arr.shape[0]:]
            # print(expected_cop_arr.shape, predicted_cop_arr.shape)
            # error = np.sum(np.abs(expected_cop_arr[obs_length:]-predicted_cop_arr[obs_length:]))/expected_cop_arr[obs_length:].shape[0]
            error = np.abs(mean_expected_cop-mean_predicted_cop)
            error = np.around(error, decimals=4)
            error_cop[i,0]=error

        new_df = pd.DataFrame({'minutes err':error_cop[:,0]})
        new_df.to_csv(plot_directory+"/"+"COP_error_minutes.csv",index=False)



        '''
        hours
        '''
        expected_cop_day_arr=[]
        predicted_cop_day_arr=[]
        for i in range(expected_enthalpy_arr.shape[0]):
            expected_cop_arr = (expected_enthalpy_arr[i,:,0:1]-expected_enthalpy_arr[i,:,2:3])/(expected_enthalpy_arr[i,:,1:2]-expected_enthalpy_arr[i,:,0:1])
            predicted_cop_arr = (predicted_enthalpy_arr[i,:,0:1]-predicted_enthalpy_arr[i,:,2:3])/(predicted_enthalpy_arr[i,:,1:2]-predicted_enthalpy_arr[i,:,0:1])
            # predicted_cop_arr[obs_length:] = filter(predicted_cop_arr[obs_length:])

            expected_cop_day = expected_cop_arr[obs_length:]
            predicted_cop_day = predicted_cop_arr[obs_length:]
            expected_cop_day = np.mean(expected_cop_day)
            predicted_cop_day = np.mean(predicted_cop_day)
            expected_cop_day_arr.append(expected_cop_day)
            predicted_cop_day_arr.append(predicted_cop_day)

        plt.ylim(6.5,9)
        hour = pred_length/30
        inveral = 7
        expected_cop_day_arr=np.array(expected_cop_day_arr)
        predicted_cop_day_arr=np.array(predicted_cop_day_arr)
        expected_cop_day_arr=np.reshape(expected_cop_day_arr,(expected_cop_day_arr.shape[0],1))
        predicted_cop_day_arr=np.reshape(predicted_cop_day_arr,(predicted_cop_day_arr.shape[0],1))
        plt.xticks(np.arange(0,expected_cop_day_arr.shape[0]*hour,hour))

        plt.plot(np.arange(0,expected_cop_day_arr.shape[0]*hour,hour), expected_cop_day_arr, label="Expected COP", linestyle='--', marker='+')
        plt.plot(np.arange(0,expected_cop_day_arr.shape[0]*hour,hour), predicted_cop_day_arr, label="Predictive COP", linestyle='--', marker='^')
        plt.xlabel('Time(hours)', fontsize=12)
        plt.ylabel('COP', fontsize=12)
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)
        plt.legend()
        plt.savefig(plot_directory+"/"+str(hour)+"hours_COP"+".png")
        plt.clf()
        # plt.show()

        # compute error
        # sc = MinMaxScaler()
        # fit_data = np.concatenate((expected_cop_day_arr, predicted_cop_day_arr), axis=0)
        # fit_data = sc.fit_transform(fit_data)
        # expected = fit_data[:expected_cop_day_arr.shape[0]]
        # predicted = fit_data[expected_cop_day_arr.shape[0]:]

        error = np.abs(expected_cop_day_arr-predicted_cop_day_arr)
        error = np.around(error, decimals=4)
        print(expected_cop_day_arr.shape, predicted_cop_day_arr.shape, error.shape)

        new_df = pd.DataFrame({'real COP':expected_cop_day_arr[:,0], 'predict COP':predicted_cop_day_arr[:,0], 'err':error[:,0]})
        new_df.to_csv(plot_directory+"/"+"COP_error_hours.csv",index=False)




def filter(data):
    width = 3
    middle = (int)((width + 1)/2)

    m = data.shape[0]
    newData = np.copy(data)

    for i in range(middle, m):
        newData[i] = np.average(data[i-middle:i+middle])

    return newData





if __name__ == "__main__":
    main()