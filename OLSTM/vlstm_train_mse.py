import torch
import numpy as np
from torch.autograd import Variable

import argparse
import os
import time
import pickle
import subprocess

from model import VLSTM
from utils import DataLoader
# from grid import getSequenceGridMask
from helper import *

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--input_size', type=int, default=5)
    parser.add_argument('--output_size', type=int, default=3)
    # RNN size parameter (dimension of the output/hidden state)
    parser.add_argument('--rnn_size', type=int, default=128,
                        help='size of RNN hidden state')
    # Size of each batch parameter
    parser.add_argument('--batch_size', type=int, default=256,
                        help='minibatch size')
    # Length of sequence to be considered parameter
    parser.add_argument('--seq_length', type=int, default=180,
                        help='RNN sequence length')
    parser.add_argument('--pred_length', type=int, default=240,
                        help='prediction length')
    # Number of epochs parameter
    parser.add_argument('--num_epochs', type=int, default=200,
                        help='number of epochs')
    # TODO: (resolve) Clipping gradients for now. No idea whether we should
    # Gradient value at which it should be clipped
    parser.add_argument('--grad_clip', type=float, default=10.,
                        help='clip gradients at this value')
    # Learning rate parameter
    parser.add_argument('--learning_rate', type=float, default=0.001,
                        help='learning rate')
    # Decay rate for the learning rate parameter
    parser.add_argument('--decay_rate', type=float, default=0.95,
                        help='decay rate for rmsprop')
    # Lambda regularization parameter (L2)
    parser.add_argument('--lambda_param', type=float, default=0.001,
                        help='L2 regularization parameter')
    # Cuda parameter
    parser.add_argument('--use_cuda', action="store_true", default=True,
                        help='Use GPU or not')
    # GRU parameter
    parser.add_argument('--gru', action="store_true", default=False,
                        help='True : GRU cell, False: LSTM cell')

    # number of validation will be used
    parser.add_argument('--num_validation', type=int, default=1,
                        help='Total number of validation dataset for validate accuracy')

    # frequency of validation
    parser.add_argument('--freq_validation', type=int, default=50,
                        help='Frequency number(epoch) of validation using validation data')
    
    # frequency of optimazer learning decay
    parser.add_argument('--freq_optimizer', type=int, default=80,
                        help='Frequency number(epoch) of learning decay for optimizer')

    args = parser.parse_args()
    train(args)

def train(args):
    prefix = ''
    f_prefix = '.'
    
    if not os.path.isdir("log/"):
        print("Directory creation script is running...")
        subprocess.call([f_prefix+'/make_directories.sh'])

    args.freq_validation = np.clip(args.freq_validation, 0, args.num_epochs)
    validation_epoch_list = list(range(args.freq_validation, args.num_epochs+1, args.freq_validation))
    validation_epoch_list[-1]-=1


    # Create the data loader object. This object would preprocess the data in terms of
    # batches each of size args.batch_size, of length args.seq_length
    dataloader = DataLoader(f_prefix, args.batch_size, args.seq_length, args.num_validation, forcePreProcess=True)

    method_name = "VANILLALSTM"
    model_name = "LSTM"
    save_tar_name = method_name+"_lstm_model_"
    if args.gru:
        model_name = "GRU"
        save_tar_name = method_name+"_gru_model_"

    # Log directory
    log_directory = os.path.join(prefix, 'log/')
    plot_directory = os.path.join(prefix, 'plot/', method_name, model_name)
    plot_train_file_directory = 'validation'



    # Logging files
    log_file_curve = open(os.path.join(log_directory, method_name, model_name,'log_curve.txt'), 'w+')
    log_file = open(os.path.join(log_directory, method_name, model_name, 'val.txt'), 'w+')

    # model directory
    save_directory = os.path.join(f_prefix, 'model')
    
    # Save the arguments int the config file
    with open(os.path.join(save_directory, method_name, model_name,'config.pkl'), 'wb') as f:
        pickle.dump(args, f)

    # Path to store the checkpoint file
    def checkpoint_path(x):
        return os.path.join(save_directory, method_name, model_name, save_tar_name+str(x)+'.tar')
    
    # model creation
    net = VLSTM(args)
    if args.use_cuda:
        net = net.cuda()

    # optimizer = torch.optim.Adagrad(net.parameters(), weight_decay=args.lambda_param)
    optimizer = torch.optim.RMSprop(net.parameters(), lr=args.learning_rate)
    loss_f = torch.nn.MSELoss()
    learning_rate = args.learning_rate

    best_val_loss = 100
    best_val_data_loss = 100

    smallest_err_val = 100000
    smallest_err_val_data = 100000


    best_epoch_val = 0
    best_epoch_val_data = 0

    best_err_epoch_val = 0
    best_err_epoch_val_data = 0

    all_epoch_results = []
    grids = []
    num_batch = 0

    # Training
    for epoch in range(args.num_epochs):
        print('****************Training epoch beginning******************')
        if dataloader.additional_validation and (epoch-1) in validation_epoch_list:
            dataloader.switch_to_dataset_type(True)
        dataloader.reset_batch_pointer(valid=False)
        loss_epoch = 0
        # For each batch
        # num_batches 資料可以被分多少批 要跑幾個iter
        
        for batch in range(dataloader.num_batches):
            start = time.time()

            # print(dataloader.num_batches, dataloader.batch_size)
            
            # Get batch data
            x, y, d = dataloader.next_batch(randomUpdate=False)
            
            loss_batch = 0

            # x_cat = Variable(torch.from_numpy(np.array(x[0])).float())
            x_n = np.array(x)
            y_n = np.array(y)[:,:,:3]
            x_n = Variable(torch.from_numpy(x_n).float())
            y_n = Variable(torch.from_numpy(y_n).float())
            hidden_states = Variable(torch.zeros(3, y_n.size()[0], net.embedding_size))
            cell_states = Variable(torch.zeros(3, y_n.size()[0], net.embedding_size))

            if args.use_cuda:                  
                x_n = x_n.cuda()
                y_n = y_n.cuda()
                hidden_states = hidden_states.cuda()
                cell_states = cell_states.cuda()

            # Zero out gradients
            net.zero_grad()
            optimizer.zero_grad()
            outputs, _, _ = net(x_n, hidden_states, cell_states)

            loss = loss_f(outputs, y_n)
            loss_batch = loss.detach().item()

            # Compute gradients
            loss.backward()

            # Clip gradients
            torch.nn.utils.clip_grad_norm_(net.parameters(), args.grad_clip)

            # Update parameters
            optimizer.step()

            end = time.time()
            loss_epoch += loss_batch

            print('{}/{} (epoch {}), train_loss = {:.3f}, time/batch = {:.3f}'.format((batch+1) * dataloader.batch_size,
                                                                                    dataloader.num_batches * dataloader.batch_size,
                                                                                    epoch,
                                                                                    loss_batch, end - start))
        loss_epoch /= dataloader.num_batches
        print("Training epoch: "+str(epoch)+" loss: "+str(loss_epoch))
        #Log loss values
        log_file_curve.write("Training epoch: "+str(epoch)+" loss: "+str(loss_epoch)+'\n')


        # Validation dataset
        if dataloader.additional_validation and (epoch) in validation_epoch_list:
            dataloader.switch_to_dataset_type()
            print('****************Validation with dataset epoch beginning******************')
            dataloader.reset_batch_pointer(valid=False)
            dataset_pointer_ins = dataloader.dataset_pointer
            validation_dataset_executed = True

            loss_epoch = 0
            err_epoch = 0
            num_of_batch = 0
            smallest_err = 100000

            #results of one epoch for all validation datasets
            epoch_result = []
            #results of one validation dataset
            results = []

            # For each batch
            for batch in range(dataloader.num_batches):
                # Get batch data
                x, y, d = dataloader.next_batch(randomUpdate=False)

                # Loss for this batch
                loss_batch = 0
                err_batch = 0

                # For each sequence
                for sequence in range(len(x)):
                    # Get the sequence
                    x_seq = x[sequence]
                    y_seq = y[sequence]
                    x_seq= np.array(x_seq)
                    y_seq= np.array(y_seq)[:,:3]
                    x_seq = Variable(torch.from_numpy(x_seq).float())
                    y_seq = Variable(torch.from_numpy(y_seq).float())

                    if args.use_cuda:
                        x_seq = x_seq.cuda()
                        y_seq = y_seq.cuda()

                    #will be used for error calculation
                    orig_x_seq = y_seq.clone() 

                    # print(x_seq.size(), args.seq_length)

                    with torch.no_grad():
                        hidden_states = Variable(torch.zeros(3, 1, net.embedding_size))
                        cell_states = Variable(torch.zeros(3, 1, net.embedding_size))
                        ret_x_seq = Variable(torch.zeros(args.seq_length, 3))
                        # all_outputs = Variable(torch.zeros(1, args.seq_length, net.input_size))

                        # Initialize the return data structure
                        if args.use_cuda:
                            ret_x_seq = ret_x_seq.cuda()
                            hidden_states = hidden_states.cuda()
                            cell_states = cell_states.cuda()

                        total_loss = 0
                        # For the observed part of the trajectory
                        for tstep in range(args.seq_length):
                            outputs, hidden_states, cell_states = net(x_seq[tstep].view(1, 1, net.input_size), hidden_states, cell_states)
                            ret_x_seq[tstep, 0] = outputs[0,0,0]
                            ret_x_seq[tstep, 1] = outputs[0,0,1]
                            ret_x_seq[tstep, 2] = outputs[0,0,2]
                            loss = loss_f(outputs, y_seq[tstep].view(1, 1, y_seq.size()[1]))
                            total_loss += loss

                        total_loss = total_loss / args.seq_length

                    #get mean and final error
                    # print(ret_x_seq.size(), y_seq.size())
                    err = get_mean_error(ret_x_seq.data, y_seq.data, args.use_cuda)

                    loss_batch += total_loss.item()
                    err_batch += err
                    print('Current file : ',' Batch : ', batch+1, ' Sequence: ', sequence+1, ' Sequence mean error: ', err, 'valid_loss: ',total_loss.item())
                    results.append((y_seq.data.cpu().numpy(), ret_x_seq.data.cpu().numpy()))

                loss_batch = loss_batch / dataloader.batch_size
                err_batch = err_batch / dataloader.batch_size
                num_of_batch += 1
                loss_epoch += loss_batch
                err_epoch += err_batch

            epoch_result.append(results)
            all_epoch_results.append(epoch_result)

            if dataloader.num_batches != 0:            
                loss_epoch = loss_epoch / dataloader.num_batches
                err_epoch = err_epoch / dataloader.num_batches
                # avarage_err = (err_epoch + f_err_epoch)/2

                # Update best validation loss until now
                if loss_epoch < best_val_data_loss:
                    best_val_data_loss = loss_epoch
                    best_epoch_val_data = epoch

                if err_epoch<smallest_err_val_data:
                    # Save the model after each epoch
                    print('Saving model')
                    torch.save({
                        'epoch': epoch,
                        'state_dict': net.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict()
                    }, checkpoint_path(epoch))

                    smallest_err_val_data = err_epoch
                    best_err_epoch_val_data = epoch

                print('(epoch {}), valid_loss = {:.3f}, valid_mean_err = {:.3f}'.format(epoch, loss_epoch, err_epoch))
                print('Best epoch', best_epoch_val_data, 'Best validation loss', best_val_data_loss, 'Best error epoch',best_err_epoch_val_data, 'Best error', smallest_err_val_data)
                log_file_curve.write("Validation dataset epoch: "+str(epoch)+" loss: "+str(loss_epoch)+" mean_err: "+str(err_epoch.data.cpu().numpy())+'\n')
            


        optimizer = time_lr_scheduler(optimizer, epoch, lr_decay_epoch = args.freq_optimizer)

    if dataloader.valid_num_batches != 0:        
        print('Best epoch', best_epoch_val, 'Best validation Loss', best_val_loss, 'Best error epoch',best_err_epoch_val, 'Best error', smallest_err_val)
        # Log the best epoch and best validation loss
        log_file.write('Validation Best epoch:'+str(best_epoch_val_data)+','+' Best validation Loss: '+str(best_val_data_loss))

    if dataloader.additional_validation:
        print('Best epoch acording to validation dataset', best_epoch_val_data, 'Best validation Loss', best_val_data_loss, 'Best error epoch',best_err_epoch_val_data, 'Best error', smallest_err_val_data)
        log_file.write("Validation dataset Best epoch: "+str(best_epoch_val_data)+','+' Best validation Loss: '+str(best_val_data_loss)+'Best error epoch: ',str(best_err_epoch_val_data),'\n')
        #dataloader.write_to_plot_file(all_epoch_results[best_epoch_val_data], plot_directory)

    #elif dataloader.valid_num_batches != 0:
    #    dataloader.write_to_plot_file(all_epoch_results[best_epoch_val], plot_directory)

    #else:
    if validation_dataset_executed:
        dataloader.switch_to_dataset_type(load_data=False)
        create_directories(plot_directory, [plot_train_file_directory])
        dataloader.write_to_plot_file(all_epoch_results[len(all_epoch_results)-1], os.path.join(plot_directory, plot_train_file_directory))

    # Close logging files
    log_file.close()
    log_file_curve.close()




if __name__ == '__main__':
    main()