import os
import pickle
import numpy as np
import pandas as pd
import random
import torch
import math
from torch.autograd import Variable
from helper import *
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler

class DataLoader():

    def __init__(self, f_prefix, batch_size=32, seq_length=60, num_of_validation=0, forcePreProcess=False, infer=False, generate=False):
        '''
        Initialiser function for the DataLoader class
        params:
        batch_size : Size of the mini-batch
        seq_length : Sequence length to be considered
        num_of_validation : number of validation dataset will be used
        infer : flag for test mode
        generate : flag for data generation mode
        forcePreProcess : Flag to forcefully preprocess the data again from csv files
        '''
        
        base_train_dataset = ['./data/train/A/013157800087578_v2.csv']
        base_test_dataset = ['./data/test/A/013157800087578_v2.csv']

        # dimensions of each file set
        self.dataset_dimensions = {'data': [0, 0]}

        # check infer flag, if true choose test directory as base directory
        if infer is False:
            self.base_data_dirs = base_train_dataset
        else:
            self.base_data_dirs = base_test_dataset

        # List of data directories where raw data resides
        self.base_train_path = 'data/train/'
        self.base_test_path = 'data/test/'
        self.base_validation_path = 'data/validation/'

        # get all files using python os and base directories
        # 取得指定路徑底下所有的檔案
        # self.train_dataset = base_train_dataset
        # self.test_dataset = base_test_dataset

        # check infer flag, if true choose test directory as base directory
        if infer is False:
            self.base_data_dirs = base_train_dataset
        else:
            self.base_data_dirs = base_test_dataset

        # get all files using python os and base directories
        self.train_dataset = self.get_dataset_path(self.base_train_path, f_prefix)
        self.test_dataset = self.get_dataset_path(self.base_test_path, f_prefix)
        self.validation_dataset = self.get_dataset_path(self.base_validation_path, f_prefix)



        # if generate mode, use directly train base files
        if generate:
            self.train_dataset = [os.path.join(f_prefix, dataset[1:]) for dataset in base_train_dataset]

        # request of use of validation dataset
        if num_of_validation > 0:
            self.additional_validation = True
        else:
            self.additional_validation = False

        # check validation dataset availibility and clip the reuqested number if it is bigger than available validation dataset
        if self.additional_validation:
            if len(self.validation_dataset) is 0:
                print("There is no validation dataset.Aborted.")
                self.additional_validation = False
            else:
                num_of_validation = np.clip(num_of_validation, 0, len(self.validation_dataset))
                self.validation_dataset = random.sample(self.validation_dataset, num_of_validation)

        # if not infer mode, use train dataset
        if infer is False:
            self.data_dirs = self.train_dataset
        else:
            # use validation dataset
            if self.additional_validation:
                self.data_dirs = self.validation_dataset
            # use test dataset
            else:
                self.data_dirs = self.test_dataset

        self.infer = infer
        self.generate = generate

        # Number of datasets
        self.numDatasets = len(self.data_dirs)

        # array for keepinng target ped ids for each sequence
        self.target_ids = []

        # Data directory where the pre-processed pickle file resides
        self.train_data_dir = os.path.join(f_prefix, self.base_train_path)
        self.test_data_dir = os.path.join(f_prefix, self.base_test_path)
        self.val_data_dir = os.path.join(f_prefix, self.base_validation_path)

        # Store the arguments
        self.batch_size = batch_size
        self.seq_length = seq_length
        self.orig_seq_lenght = seq_length

        # Validation arguments
        self.val_fraction = 0

        # Define the path in which the process data would be stored
        self.data_file_tr = os.path.join(self.train_data_dir, "trajectories_train.cpkl")        
        self.data_file_te = os.path.join(self.base_test_path, "trajectories_test.cpkl")
        self.data_file_vl = os.path.join(self.val_data_dir, "trajectories_val.cpkl")

        # for creating a dict key: folder names, values: files in this folder
        # self.create_folder_file_dict()

        if self.additional_validation:
            # If the file doesn't exist or forcePreProcess is true
            if not(os.path.exists(self.data_file_vl)) or forcePreProcess:
                print("Creating pre-processed validation data from raw data")
                # Preprocess the data from the csv files of the datasets
                # Note that this data is processed in frames
                self.frame_preprocess(self.validation_dataset, self.data_file_vl, self.additional_validation)

        if self.infer:
            # if infer mode, and no additional files -> test preprocessing
            if not self.additional_validation:
                if not(os.path.exists(self.data_file_te)) or forcePreProcess:
                    print("Creating pre-processed test data from raw data")
                    # Preprocess the data from the csv files of the datasets
                    # Note that this data is processed in frames
                    print("Working on directory: ", self.data_file_te)
                    self.frame_preprocess(self.data_dirs, self.data_file_te)
            # if infer mode, and there are additional validation files -> validation dataset visualization
            else:
                print("Validation visualization file will be created")

        # if not infer mode
        else:
            # If the file doesn't exist or forcePreProcess is true -> training pre-process
            if not(os.path.exists(self.data_file_tr)) or forcePreProcess:
                print("Creating pre-processed training data from raw data")
                # Preprocess the data from the csv files of the datasets
                # Note that this data is processed in frames
                self.frame_preprocess(self.data_dirs, self.data_file_tr)

        if self.infer:
            # Load the processed data from the pickle file
            if not self.additional_validation:  # test mode
                self.load_preprocessed(self.data_file_te)
            else:  # validation mode
                self.load_preprocessed(self.data_file_vl, True)

        else: # training mode
            self.load_preprocessed(self.data_file_tr)

        # Reset all the data pointers of the dataloader object
        self.reset_batch_pointer(valid=False)
        self.reset_batch_pointer(valid=True)

    def get_dataset_path(self, base_path, f_prefix):
        # get all datasets from given set of directories
        dataset = []
        dir_names = unique_list(self.get_all_directory_namelist())
        for dir_ in dir_names:
            dir_path = os.path.join(f_prefix, base_path, dir_)
            file_names = get_all_file_names(dir_path)
            [dataset.append(os.path.join(dir_path, file_name))
             for file_name in file_names]
        return dataset

    def get_all_directory_namelist(self):
        #return all directory names in this collection of dataset
        folder_list = [data_dir.split('/')[-2] for data_dir in (self.base_data_dirs)]
        return folder_list

    def frame_preprocess(self, data_dirs, data_file, validation_set=False):    
        '''
        Function that will pre-process the pixel_pos.csv files of each dataset
        into data with occupancy grid that can be used
        params:
        data_dirs : List of directories where raw data resides
        data_file : The file into which all the pre-processed data needs to be stored
        validation_set: true when a dataset is in validation set
        '''

        # containing time, h1, h2, h3, cop
        all_frame_data = []
        # Validation frame data
        valid_frame_data = []

        # Index of the current dataset
        dataset_index = 0

        for directory in data_dirs:
            # Load the data from the txt file
            print("Now processing: ", directory)
            # column_names = ['frame_num', 'ped_id', 'y', 'x']
            titles = ['h1', 'h2', 'h3', 'Tevwi', 'Tcdwi']
            df = pd.read_csv(directory)
            datas = np.array(df.loc[:, titles].values)
            datas = self.fit_transform_MinMaxScaler(datas)
            frames = np.array(df.iloc[:, 0:1].values)
            # if training mode, read train file to pandas dataframe and process
            if self.infer is False and validation_set is False:
                train_size = int(datas.shape[0]*0.8)
                datas = datas[:,:]
                frames = frames[:,:]
                print('train mode')
            else:
                # if validation mode, read validation file to pandas dataframe and process
                if self.additional_validation:
                    tmp1 = int(datas.shape[0]*0.9)
                    tmp2 = int(datas.shape[0])
                    datas = datas[tmp1:tmp2,:]
                    frames = frames[tmp1:tmp2,:]
                    print('validation mode')

                # if test mode, read test file to pandas dataframe and process
                else:
                    test_size =int(datas.shape[0]*0.8)
                    datas = datas[test_size:,:]
                    frames = frames[test_size:,:]
                    print('test mode')

            
            # Frame IDs of the frames in the current dataset
            _, idx = np.unique(frames, return_index=True)
            frameList = frames[np.sort(idx)]
            numFrames = len(frameList)

            all_frame_data.append([])
            valid_frame_data.append([])

            for ind, frame in enumerate(frameList):
                # Extract all pedestrians in current frame
                pedsInFrame = datas[frames[:,0] == frame,:]

                pedsInFrame = np.reshape(pedsInFrame,(pedsInFrame.shape[1]))
                # all_frame_data[dataset_index].append(np.array(pedsInFrame))


                # At inference time, data generation and if dataset is a validation dataset, no validation data
                if (ind >= numFrames * self.val_fraction) or (self.infer) or (self.generate) or (validation_set):
                    # Add the details of all the peds in the current frame to all_frame_data
                    all_frame_data[dataset_index].append(np.array(pedsInFrame))

                else:
                    valid_frame_data[dataset_index].append(np.array(pedsInFrame))

        dataset_index += 1
        print('Total size (file) --->', len(all_frame_data))
        # Save the arrays in the pickle file
        f = open(data_file, "wb")
        pickle.dump((all_frame_data, valid_frame_data), f, protocol=2)
        # pickle.dump((all_frame_data, frameList_data, numPeds_data, valid_numPeds_data, valid_frame_data, pedsList_data, valid_pedsList_data, target_ids, orig_data), f, protocol=2)
        f.close()


    def load_preprocessed(self, data_file, validation_set=False):
        '''
        Function to load the pre-processed data into the DataLoader object
        params:
        data_file : the path to the pickled data file
        validation_set : flag for validation dataset
        '''
        # Load data from the pickled file
        if(validation_set):
            print("Loading validaton datasets: ", data_file)
        else:
            print("Loading train or test dataset: ", data_file)

        f = open(data_file, 'rb')
        self.raw_data = pickle.load(f)
        f.close()

        # Get all the data from the pickle file
        self.data = self.raw_data[0]
        self.valid_data = self.raw_data[1]

        counter = 0
        valid_counter = 0
        print('Sequence size(frame) ------>',self.seq_length)
        print('One batch size (frame)--->', self.batch_size * self.seq_length)

        # For each dataset
        for dataset in range(len(self.data)):
            # get the frame data for the current dataset
            all_frame_data = self.data[dataset]
            valid_frame_data = self.valid_data[dataset]
            dataset_name = self.data_dirs[dataset].split('/')[-1]
            # calculate number of sequence 
            if self.infer != False:
                num_seq_in_dataset = int(len(all_frame_data) / (self.seq_length))
                if num_seq_in_dataset==0:
                    num_seq_in_dataset=1
                num_valid_seq_in_dataset = int(len(valid_frame_data) / (self.seq_length))
            else:
                num_seq_in_dataset = int(len(all_frame_data) - (self.seq_length))
                if num_seq_in_dataset==0:
                    num_seq_in_dataset=1
                num_valid_seq_in_dataset = int(len(valid_frame_data) - (self.seq_length))
            # num_seq_in_dataset = int(len(all_frame_data) - (self.seq_length))
            # if num_seq_in_dataset==0:
            #     num_seq_in_dataset=1
            # num_valid_seq_in_dataset = int(len(valid_frame_data) - (self.seq_length))

            if not validation_set:
                print('Training data from training dataset(name, # frame, #sequence)--> ', dataset_name, ':', len(all_frame_data),':', (num_seq_in_dataset))
                print('Validation data from training dataset(name, # frame, #sequence)--> ', dataset_name, ':', len(valid_frame_data),':', (num_valid_seq_in_dataset))
            else: 
                print('Validation data from validation dataset(name, # frame, #sequence)--> ', dataset_name, ':', len(all_frame_data),':', (num_seq_in_dataset))
            
            # Increment the counter with the number of sequences in the current dataset
            counter += num_seq_in_dataset
            valid_counter += num_valid_seq_in_dataset

        # Calculate the number of batches
        # if self.infer != False:
        #     self.num_batches = 1
        #     self.valid_num_batches = 1
        #     self.batch_size = counter
        # else:
        #     self.num_batches = math.ceil(counter/self.batch_size)
        #     self.valid_num_batches = math.ceil(valid_counter/self.batch_size)
        self.num_batches = math.ceil(counter/self.batch_size)
        self.valid_num_batches = math.ceil(valid_counter/self.batch_size)

        if not validation_set:
            print('Total number of training batches:', self.num_batches)
            print('Total number of validation batches:', self.valid_num_batches)
        else:
            print('Total number of validation batches:', self.num_batches)

        # self.valid_num_batches = self.valid_num_batches * 2

    def next_batch(self, randomUpdate=True):
        '''
        Function to get the next batch of points
        '''
        # Source data
        x_batch = []
        # Target data
        y_batch = []
        # Dataset data
        d = []

        # Iteration index
        i = 0
        while i < self.batch_size:
            
            # Extract the frame data of the current dataset
            frame_data = self.data[self.dataset_pointer]
            # Get the frame pointer for the current dataset
            idx = self.frame_pointer
            # print(idx )

            # While there is still seq_length number of frames left in the current dataset
            if idx + self.seq_length+1 < len(frame_data):
                # All the data in this sequence
                # seq_frame_data = frame_data[idx:idx+self.seq_length+1]
                seq_source_frame_data = frame_data[idx:idx+self.seq_length]
                seq_target_frame_data = frame_data[idx+1:idx+self.seq_length+1]

                # Number of unique peds in this sequence of frames
                x_batch.append(seq_source_frame_data)
                y_batch.append(seq_target_frame_data)

                # advance the frame pointer to a random point
                if randomUpdate:
                    self.frame_pointer += random.randint(1, self.seq_length)
                else:
                    if self.infer != False:
                        self.frame_pointer += self.seq_length
                    else:
                        self.frame_pointer += 1
                    

                d.append(self.dataset_pointer)
                i += 1

            else:
                # Not enough frames left
                # Increment the dataset pointer and set the frame_pointer to zero
                # self.tick_batch_pointer(valid=False)
                # break
                self.reset_batch_pointer(valid=False)

        return x_batch, y_batch, d


    def next_valid_batch(self, randomUpdate=True):
        '''
        Function to get the next Validation batch of points
        '''
        # Source data
        x_batch = []
        # Target data
        y_batch = []
        # Dataset data
        d = []
        # Iteration index
        i = 0
        while i < self.batch_size:
            # Extract the frame data of the current dataset
            frame_data = self.valid_data[self.valid_dataset_pointer]
            # Get the frame pointer for the current dataset
            idx = self.valid_frame_pointer
            # While there is still seq_length number of frames left in the current dataset
            if idx + self.seq_length+1 < len(frame_data):
                # All the data in this sequence
                # seq_frame_data = frame_data[idx:idx+self.seq_length+1]
                seq_source_frame_data = frame_data[idx:idx+self.seq_length]
                seq_target_frame_data = frame_data[idx+1:idx+self.seq_length+1]

                # Number of unique peds in this sequence of frames
                x_batch.append(seq_source_frame_data)
                y_batch.append(seq_target_frame_data)

                # advance the frame pointer to a random point
                if randomUpdate:
                    self.valid_frame_pointer += random.randint(1, self.seq_length)
                else:
                    self.valid_frame_pointer += 1

                d.append(self.valid_dataset_pointer)
                i += 1

            else:
                # Not enough frames left
                # Increment the dataset pointer and set the frame_pointer to zero
                # break
                self.reset_batch_pointer(valid=True)

        return x_batch, y_batch, d

    def tick_batch_pointer(self, valid=False):
        '''
        Advance the dataset pointer
        '''
        if not valid:
            # Go to the next dataset
            self.dataset_pointer += 1
            # Set the frame pointer to zero for the current dataset
            self.frame_pointer = 0
            # If all datasets are done, then go to the first one again
            if self.dataset_pointer >= len(self.data):
                self.dataset_pointer = 0
        else:
            # Go to the next dataset
            self.valid_dataset_pointer += 1
            # Set the frame pointer to zero for the current dataset
            self.valid_frame_pointer = 0
            # If all datasets are done, then go to the first one again
            if self.valid_dataset_pointer >= len(self.valid_data):
                self.valid_dataset_pointer = 0

    def reset_batch_pointer(self, valid=False):
        '''
        Reset all pointers
        '''
        if not valid:
            # Go to the first frame of the first dataset
            self.dataset_pointer = 0
            self.frame_pointer = 0
        else:
            self.valid_dataset_pointer = 0
            self.valid_frame_pointer = 0

    def create_folder_file_dict(self):
        # create a helper dictionary folder name:file name
        self.folder_file_dict = {}
        for dir_ in self.base_data_dirs:
            folder_name = dir_.split('/')[-2]
            file_name = dir_.split('/')[-1]
            self.add_element_to_dict(
                self.folder_file_dict, folder_name, file_name)

    def switch_to_dataset_type(self, train = False, load_data = True):
        # function for switching between train and validation datasets during training session
        print('--------------------------------------------------------------------------')
        if not train: # if train mode, switch to validation mode
            if self.additional_validation:
                print("Dataset type switching: training ----> validation")
                self.orig_seq_lenght, self.seq_length = self.seq_length, self.orig_seq_lenght
                self.data_dirs = self.validation_dataset
                self.numDatasets = len(self.data_dirs)
                if load_data:
                    self.load_preprocessed(self.data_file_vl, True)
                    self.reset_batch_pointer(valid=False)
            else: 
                print("There is no validation dataset.Aborted.")
                return
        else:# if validation mode, switch to train mode
            print("Dataset type switching: validation -----> training")
            self.orig_seq_lenght, self.seq_length = self.seq_length, self.orig_seq_lenght
            self.data_dirs = self.train_dataset
            self.numDatasets = len(self.data_dirs)
            if load_data:
                self.load_preprocessed(self.data_file_tr)
                self.reset_batch_pointer(valid=False)
                self.reset_batch_pointer(valid=True)

    def write_to_plot_file(self, data, path):
        # write plot file for further visualization in pkl format
        self.reset_batch_pointer()

        file_name = 'test_ouput.pkl'
        print("Writing to plot file  path: %s, file_name: %s"%(path, file_name))
        with open(os.path.join(path, file_name), 'wb') as f:
            pickle.dump(data, f)

    def fit_transform_StandardScaler(self, data):
        self.sc = StandardScaler()
        return self.sc.fit_transform(data)

    def inverse_transform_StandardScaler(self, y):
        return self.sc.inverse_transform(y)

    def fit_transform_MinMaxScaler(self, data):
        self.sc2 = MinMaxScaler()
        return self.sc2.fit_transform(data)
        
    def inverse_transform_MinMaxScaler(self, y):
        return self.sc2.inverse_transform(y)