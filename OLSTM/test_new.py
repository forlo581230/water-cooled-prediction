import torch
import numpy as np
from torch.autograd import Variable

import argparse
import os
import time
import pickle
import subprocess

from model import LSTM
from utils import DataLoader
# from grid import getSequenceGridMask
from helper import *

import matplotlib.pyplot as plt
from waterChiller import WaterChiller

def main():
    '''
    主要目的為 計算測試資料的 error rate
    '''
    parser = argparse.ArgumentParser()
    # Observed length of the trajectory parameter
    parser.add_argument('--obs_length', type=int, default=240,
                        help='Observed length of the trajectory')
    # Predicted length of the trajectory parameter
    # parser.add_argument('--pred_length', type=int, default=378-60-1,
    #                     help='Predicted length of the trajectory')

    parser.add_argument('--pred_length', type=int, default=240,
                        help='Predicted length of the trajectory')
    # Model to be loaded
    parser.add_argument('--epoch', type=int, default=199,
                        help='Epoch of model to be loaded')
    # cuda support
    parser.add_argument('--use_cuda', action="store_true", default=True,
                        help='Use GPU or not')

    # gru model
    parser.add_argument('--gru', action="store_true", default=False,
                        help='True : GRU cell, False: LSTM cell')
    # method selection
    parser.add_argument('--method', type=int, default=1,
                        help='Method of lstm will be used (1 = social lstm, 2 = obstacle lstm, 3 = vanilla lstm)')

    # Parse the parameters
    sample_args = parser.parse_args()

    # for drive run
    prefix = ''
    f_prefix = '.'

    method_name = "VANILLALSTM"
    model_name = "LSTM"
    save_tar_name = method_name+"_lstm_model_"
    if sample_args.gru:
        model_name = "GRU"
        save_tar_name = method_name+"_gru_model_"

    print("Selected method name: ", method_name, " model name: ", model_name)

    # Save directory
    save_directory = os.path.join(f_prefix, 'model/', method_name, model_name)
    # plot directory for plotting in the future
    plot_directory = os.path.join(f_prefix, 'plot/', method_name, model_name)

    result_directory = os.path.join(f_prefix, 'result/', method_name)
    plot_test_file_directory = 'test'

    # Define the path for the config file for saved args
    with open(os.path.join(save_directory, 'config.pkl'), 'rb') as f:
        saved_args = pickle.load(f)

    seq_lenght = sample_args.pred_length + sample_args.obs_length

    # Create the DataLoader object
    dataloader = DataLoader(f_prefix, 1, sample_args.pred_length +
                            sample_args.obs_length, forcePreProcess=True, infer=True)
    create_directories(os.path.join(result_directory, model_name),
                       dataloader.get_all_directory_namelist())
    create_directories(plot_directory, [plot_test_file_directory])
    dataloader.reset_batch_pointer(valid=False)

    dataset_pointer_ins = dataloader.dataset_pointer

    smallest_err = 100000
    smallest_err_iter_num = -1
    origin = (0, 0)
    reference_point = (0, 1)

    submission_store = []  # store submission data points (txt)
    result_store = []  # store points for plotting

    # Initialize net
    net = LSTM(saved_args, True)

    if sample_args.use_cuda:
        net = net.cuda()

    # Get the checkpoint path
    checkpoint_path = os.path.join(
        save_directory, save_tar_name+str(sample_args.epoch)+'.tar')
    if os.path.isfile(checkpoint_path):
        print('Loading checkpoint')
        checkpoint = torch.load(checkpoint_path)
        model_epoch = checkpoint['epoch']
        net.load_state_dict(checkpoint['state_dict'])
        print('Loaded checkpoint at epoch', model_epoch)


    results_it = []
    for iterator in range(50):
        x_seq_arr = []
        ret_x_seq_arr =[]
        error_arr=[]
        expected_day_arr = []
        predicted_day_arr = []
        

        total_error = 0

        for batch in range(dataloader.num_batches):
            # Get data
            x, y, d = dataloader.next_batch(randomUpdate=False)

            # Get the sequence
            x_seq, y_seq, d_seq = x[0], y[0], d[0]
            x_seq = np.array(x_seq)
            '''
            x_seq = dataloader.inverse_transform_MinMaxScaler(x_seq)
            print('{}/{}'.format(batch, dataloader.num_batches))
            x_seq[sample_args.obs_length:,-2]= 17
            x_seq[sample_args.obs_length:,-1]= 28
            x_seq = dataloader.fit_transform_MinMaxScaler(x_seq)
            '''
            x_seq = Variable(torch.from_numpy(x_seq).float())

            temp = x_seq[:,-2:]
            # x_seq = x_seq[:,:-2]



            if sample_args.use_cuda:
                x_seq = x_seq.cuda()
                temp = temp.cuda()

            
            obs_data = x_seq[:sample_args.obs_length]

            ret_x_seq = sample(sample_args, x_seq, temp, net)

            error = get_mean_error(x_seq[sample_args.obs_length:,:-2], ret_x_seq[sample_args.obs_length:,:-2], False)
            total_error += error

            
            # 顯示預測
            # x_seq = result[0]
            x_seq = x_seq.data.cpu().numpy()
            # print(x_seq.size())
            # x_seq = np.reshape(x_seq,(x_seq.shape[0], saved_args.input_size))
            x_seq = dataloader.inverse_transform_MinMaxScaler(x_seq)
            # ret_x_seq = result[1]
            ret_x_seq = ret_x_seq.data.cpu().numpy()
            # ret_x_seq = np.reshape(ret_x_seq,(ret_x_seq.shape[0], saved_args.input_size))
            ret_x_seq = dataloader.inverse_transform_MinMaxScaler(ret_x_seq)

            gt = (x_seq[:,0]-x_seq[:,2])/(x_seq[:,1]-x_seq[:,0])
            pred = (ret_x_seq[:,0]-ret_x_seq[:,2])/(ret_x_seq[:,1]-ret_x_seq[:,0])


            gt2 = gt[sample_args.obs_length:]
            pred2 = pred[sample_args.obs_length:]
            expected_day = np.mean(gt2)
            predicted_day = np.mean(pred2)
            # print(expected_day, predicted_day, expected_day-predicted_day)
            # print('Error: ',error)


            expected_day = np.mean(gt2)
            predicted_day = np.mean(pred2)

            x_seq_arr.append(x_seq)
            ret_x_seq_arr.append(ret_x_seq)
            error_arr.append(error.data.cpu().numpy())
            expected_day_arr.append(expected_day)
            predicted_day_arr.append(predicted_day)

            # fig, axs = plt.subplots(6, 1)
            # axs[0].plot(ret_x_seq[:,0], color = 'blue' , label = 'Predict h1', linestyle='--', marker='^')
            # axs[0].plot(x_seq[:,0], color = 'red', label = 'Real h1', linestyle='-', marker='.')
            # axs[1].plot(ret_x_seq[:,1], color = 'blue' , label = 'Predict h2', linestyle='--', marker='^')
            # axs[1].plot(x_seq[:,1], color = 'red', label = 'Real h2', linestyle='-', marker='.')
            # axs[2].plot(ret_x_seq[:,2], color = 'blue' , label = 'Predict h3', linestyle='--', marker='^')
            # axs[2].plot(x_seq[:,2], color = 'red', label = 'Real h3', linestyle='-', marker='.')
            # axs[3].plot(pred, color = 'blue' , label = 'Predict h3', linestyle='--', marker='^')
            # axs[3].plot(gt, color = 'red', label = 'Real h3', linestyle='-', marker='.')

            # axs[4].plot(ret_x_seq[:,-2], color = 'blue' , label = 'Predict Tevwi', linestyle='--', marker='^')
            # axs[4].plot(x_seq[:,-2], color = 'red', label = 'Real Tevwi', linestyle='-', marker='.')

            # axs[5].plot(ret_x_seq[:,-1], color = 'blue' , label = 'Predict Tcdwi', linestyle='--', marker='^')
            # axs[5].plot(x_seq[:,-1], color = 'red', label = 'Real Tcdwi', linestyle='-', marker='.')

            # for ax in axs:
            #     ax.legend()
            #     ax.grid()
            # plt.show()

        total_error = total_error/dataloader.num_batches
        if total_error<smallest_err:
            print("**********************************************************")
            print('Best iteration has been changed. Previous best iteration: ', smallest_err_iter_num, 'Error: ', smallest_err)
            print('New best iteration : ', iterator, 'Error: ',total_error)
            smallest_err_iter_num = iterator
            smallest_err = total_error
            
        results_it.append((sample_args.pred_length, sample_args.obs_length, x_seq_arr, ret_x_seq_arr, error_arr))
    
    dataloader.write_to_plot_file([results_it[smallest_err_iter_num]], os.path.join(plot_directory, plot_test_file_directory))




        

    

def sample(args, x_seq, temp, net):
    # Construct variables for hidden and cell states

    # Initialize the return data structure
    extra_pred_length = 0
    x_seq = x_seq[:,:-2]

    with torch.no_grad():
        hidden_states = Variable(torch.zeros(1, net.rnn_size))
        cell_states = Variable(torch.zeros(1, net.rnn_size))
        ret_x_seq = Variable(torch.zeros(args.obs_length+args.pred_length + extra_pred_length, 5))
        # all_outputs = Variable(torch.zeros(1, args.seq_length, net.input_size))

        # Initialize the return data structure
        if args.use_cuda:
            ret_x_seq = ret_x_seq.cuda()
            hidden_states = hidden_states.cuda()
            cell_states = cell_states.cuda()

        for tstep in range(args.obs_length-1):
            outputs, hidden_states, cell_states = net(x_seq[tstep].view(1,1, net.input_size), temp[tstep].view(1, 1, temp.size()[-1]), hidden_states, cell_states)
            # ret_x_seq[tstep+1, :, -1] = outputs.clone()[0,:,-1]

        # outputs, hidden_states, cell_states = net(x_seq[:args.obs_length-1].view(1, args.obs_length-1, net.input_size), hidden_states, cell_states)

        #copy all obs data
        # print(x_seq.size()) 
        ret_x_seq[:args.obs_length, :-2] = x_seq.clone()[:args.obs_length, :-2]
        ret_x_seq[:, -2:] = temp.clone()[:, -2:]

        # total_loss = 0
        # For the observed part of the trajectory
        for tstep in range(args.obs_length-1, extra_pred_length + args.pred_length + args.obs_length-1):
            outputs, hidden_states, cell_states = net(ret_x_seq[tstep, :-2].view(1,1, net.input_size), temp[tstep].view(1, 1, temp.size()[-1]), hidden_states, cell_states)
            ret_x_seq[tstep+1, 0] = outputs[0,0,0]
            ret_x_seq[tstep+1, 1] = outputs[0,0,1]
            ret_x_seq[tstep+1, 2] = outputs[0,0,2]
        return ret_x_seq


if __name__ == '__main__':
    main()
