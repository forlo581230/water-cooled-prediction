import numpy as np
import torch
from torch.autograd import Variable

import os
import shutil
from os import walk
import math

# from olstm_model import OLSTMModel
# from vlstm_model import VLSTMModel



#one time set dictionary for a exist key
class WriteOnceDict(dict):
    def __setitem__(self, key, value):
        if not key in self:
            super(WriteOnceDict, self).__setitem__(key, value)



def sample_gaussian_2d(mux, muy, sx, sy, corr, nodesPresent, look_up):
    '''
    Parameters
    ==========

    mux, muy, sx, sy, corr : a tensor of shape 1 x numNodes
    Contains x-means, y-means, x-stds, y-stds and correlation

    nodesPresent : a list of nodeIDs present in the frame
    look_up : lookup table for determining which ped is in which array index

    Returns
    =======

    next_x, next_y : a tensor of shape numNodes
    Contains sampled values from the 2D gaussian
    '''
    o_mux, o_muy, o_sx, o_sy, o_corr = mux[0, :], muy[0, :], sx[0, :], sy[0, :], corr[0, :]

    numNodes = mux.size()[1]
    next_x = torch.zeros(numNodes)
    next_y = torch.zeros(numNodes)
    converted_node_present = [look_up[node] for node in nodesPresent]
    for node in range(numNodes):
        if node not in converted_node_present:
            continue
        mean = [o_mux[node], o_muy[node]]
        cov = [[o_sx[node]*o_sx[node], o_corr[node]*o_sx[node]*o_sy[node]], [o_corr[node]*o_sx[node]*o_sy[node], o_sy[node]*o_sy[node]]]

        next_values = np.random.multivariate_normal(mean, cov, 1)
        next_x[node] = next_values[0][0]
        next_y[node] = next_values[0][1]

    return next_x, next_y

def get_mean_error(ret_nodes, nodes, using_cuda=False):
    '''
    Parameters
    ==========

    ret_nodes : A tensor of shape pred_length x numNodes x 2
    Contains the predicted positions for the nodes

    nodes : A tensor of shape pred_length x numNodes x 2
    Contains the true positions for the nodes

    nodesPresent lists: A list of lists, of size pred_length
    Each list contains the nodeIDs of the nodes present at that time-step

    look_up : lookup table for determining which ped is in which array index

    Returns
    =======

    Error : Mean euclidean distance between predicted trajectory and the true trajectory
    '''
    pred_length = ret_nodes.size()[0]
    error = torch.zeros(pred_length)

    if using_cuda:
        error = error.cuda()


    for tstep in range(pred_length):
        counter = 0
        pred_pos = ret_nodes[tstep, :]
        true_pos = nodes[tstep, :]
        error[tstep] = torch.norm(pred_pos - true_pos, p=2)


    return torch.mean(error)

def get_rms_error(ret_nodes, nodes, using_cuda):
    '''
    Parameters
    ==========

    ret_nodes : A tensor of shape pred_length x numNodes x 2
    Contains the predicted positions for the nodes

    nodes : A tensor of shape pred_length x numNodes x 2
    Contains the true positions for the nodes

    nodesPresent lists: A list of lists, of size pred_length
    Each list contains the nodeIDs of the nodes present at that time-step

    look_up : lookup table for determining which ped is in which array index

    Returns
    =======

    Error : Mean euclidean distance between predicted trajectory and the true trajectory
    '''
    pred_length = ret_nodes.size()[0]
    error = torch.zeros(pred_length)

    if using_cuda:
        error = error.cuda()


    for tstep in range(pred_length):
        counter = 0
        pred_pos = ret_nodes[tstep, :]
        true_pos = nodes[tstep, :]

        error[tstep] = (torch.norm(pred_pos, p=2) - torch.norm(true_pos, p=2))**2
        

    return torch.sqrt(torch.mean(error))



def getGaussianlikehood(x, mMat, covMat, k):
    det_covMat = torch.det(covMat)
    inv_covMat = torch.inverse(covMat)
    result = torch.exp(-torch.mm(torch.mm(torch.t(x-mMat), inv_covMat), (x-mMat))/2)


    demon = torch.sqrt(((2*np.pi)**k)*det_covMat)

    result = result/demon

    # Numerical stability
    epsilon = 1e-20
    
    result = -torch.log(torch.clamp(result, min=epsilon))

    return result[0,0]

def sample_gaussian_3d(mux, muy, muz, sx, sy, sz, corr_xy, corr_yz, corr_xz):


    next_x = torch.zeros(1)
    next_y = torch.zeros(1)
    next_z = torch.zeros(1)

    i=0
    mean = [mux[i], muy[i], muz[i]]
    cov = [[sx[i]*sx[i], corr_xy[i]*sx[i]*sy[i], corr_xz[i]*sx[i]*sz[i]],
        [corr_xy[i]*sx[i]*sy[i], sy[i]*sy[i], corr_yz[i]*sy[i]*sz[i]],
        [corr_xz[i]*sx[i]*sz[i], corr_yz[i]*sy[i]*sz[i], sz[i]*sz[i]]]
    
    cov = np.array(cov, dtype=np.float)
    mean = np.array(mean, dtype=np.float)
    # print(cov)
    # print(mean,"===")

    
    next_values = np.random.multivariate_normal(mean, cov, 1)
    # print(next_values)
    next_x[0] = next_values[0][0]
    next_y[0] = next_values[0][1]
    next_z[0] = next_values[0][2]

    return next_x, next_y, next_z

def getCoef_2(outputs):
    '''
    Extracts the mean, standard deviation and correlation
    params:
    outputs : Output of the SRNN model
    '''
    mux, muy, muz, sx, sy, sz, corr_xy, corr_yz, corr_xz = outputs[:, :, 0], outputs[:, :, 1], outputs[:, :, 2], outputs[:, :, 3], outputs[:, :, 4], outputs[:, :, 5], outputs[:, :, 6], outputs[:, :, 7], outputs[:, :, 8]

    mux = torch.abs(mux)
    muy = torch.abs(muy)
    muz = torch.abs(muz)
    sx = torch.exp(sx)
    sy = torch.exp(sy)
    sz = torch.exp(sz)

    epsilon = np.sqrt(np.exp(1)/3)
    corr_xy = torch.clamp(torch.sigmoid(corr_xy), min=epsilon)
    corr_yz = torch.clamp(torch.sigmoid(corr_yz), min=epsilon)
    corr_xz = torch.clamp(torch.sigmoid(corr_xz), min=epsilon)



    return mux, muy, muz, sx, sy, sz, corr_xy, corr_yz, corr_xz

def Gaussian3DLikehoodm(outputs, targets):
    '''
    outputs: shape 1 x seq_length x output_size
    targets: shape 1 x seq_length x input_size
    '''
    mux, muy, muz, sx, sy, sz, corr_xy, corr_yz, corr_xz = getCoef_2(outputs)

    loss =0
    counter=0
    for i in range(targets.size()[1]):
        x = targets[:,i:i+1,:]

        #cov matrix
        covList = [[sx[:,i]*sx[:,i], corr_xy[:,i]*sx[:,i]*sy[:,i], corr_xz[:,i]*sx[:,i]*sz[:,i]],
            [corr_xy[:,i]*sx[:,i]*sy[:,i], sy[:,i]*sy[:,i], corr_yz[:,i]*sy[:,i]*sz[:,i]],
            [corr_xz[:,i]*sx[:,i]*sz[:,i], corr_yz[:,i]*sy[:,i]*sz[:,i], sz[:,i]*sz[:,i]]]

        normx=x[:,0,0]-mux[:,i]
        normy=x[:,0,1]-muy[:,i]
        normz=x[:,0,2]-muz[:,i]
        
        result_x = normx/covList[0][0]+ normy/covList[1][0]+ normz/covList[2][0]
        result_y = normx/covList[0][1]+ normy/covList[1][1]+ normz/covList[2][1]
        result_z = normx/covList[0][2]+ normy/covList[1][2]+ normz/covList[2][2]
        
        w = result_x**2 +result_y**2 + result_z**2
        result = torch.exp(-w/2)

        det = ((1+2*corr_xy[:,i]*corr_yz[:,i]*corr_xz[:,i])-(corr_xy[:,i]**2 + corr_yz[:,i]**2 + corr_xz[:,i]**2))
        demon = sx[:,i]*sy[:,i]*sz[:,i]*torch.sqrt(((2*np.pi)**3)*det)
        result=result/demon

        # Numerical stability
        epsilon = 1e-20
        result = -(torch.log(torch.clamp(result, min=epsilon)))
        loss = loss + torch.sum(result)
        counter = counter + targets.size()[0]
        
    # print(mux[0].data.cpu().numpy(), muy[0].data.cpu().numpy(), muz[0].data.cpu().numpy(), sx[0].data.cpu().numpy(), sy[0].data.cpu().numpy(), sz[0].data.cpu().numpy(), corr_xy[0].data.cpu().numpy(), corr_yz[0].data.cpu().numpy(), corr_xz[0].data.cpu().numpy())
    return loss/counter/targets.size()[1]

def Gaussian3DLikehood(outputs, targets):
    '''
    outputs: shape 1 x seq_length x output_size
    targets: shape 1 x seq_length x input_size
    '''
   
    # mux, muy, muz = 401948.07474566, 421633.64055084, 242232.41954159
    # sx, sy, sz = 1126.78896139, 1334.07157799, 1128.23035485
    # corr_xy, corr_yz, corr_xz = 0.9659817681129849, 0.618034828384499, 0.6649225299909955
 
    mux, muy, muz, sx, sy, sz, corr_xy, corr_yz, corr_xz = getCoef_2(outputs)

    loss =0
    counter=0
    for i in range(targets.size()[1]):
        x = targets[:,i:i+1,:]
        x = x.view(x.size()[1],x.size()[2])
        x = torch.t(x)

        det = ((1+2*corr_xy[i]*corr_yz[i]*corr_xz[i])-(corr_xy[i]**2 + corr_yz[i]**2 + corr_xz[i]**2))

        if det <= 0:
            print(det,"===")
            print(mux[i].data.cpu().numpy(), muy[i].data.cpu().numpy(), muz[i].data.cpu().numpy(), sx[i].data.cpu().numpy(), sy[i].data.cpu().numpy(), sz[i].data.cpu().numpy(), corr_xy[i].data.cpu().numpy(), corr_yz[i].data.cpu().numpy(), corr_xz[i].data.cpu().numpy())
        
        #協方差矩陣
        covList = [[sx[i]*sx[i], corr_xy[i]*sx[i]*sy[i], corr_xz[i]*sx[i]*sz[i]],
            [corr_xy[i]*sx[i]*sy[i], sy[i]*sy[i], corr_yz[i]*sy[i]*sz[i]],
            [corr_xz[i]*sx[i]*sz[i], corr_yz[i]*sy[i]*sz[i], sz[i]*sz[i]]]

        normx=x[0,0]-mux[i]
        normy=x[1,0]-muy[i]
        normz=x[2,0]-muz[i]

        result_x = normx/covList[0][0]+ normy/covList[1][0]+ normz/covList[2][0]
        result_y = normx/covList[0][1]+ normy/covList[1][1]+ normz/covList[2][1]
        result_z = normx/covList[0][2]+ normy/covList[1][2]+ normz/covList[2][2]
        
        w = result_x**2 +result_y**2 + result_z**2
        result = torch.exp(-w/2)

        # det = (covList[0][0]*covList[1][1]*covList[2][2]+covList[0][1]*covList[1][2]*covList[2][0]+covList[0][2]*covList[1][0]*covList[2][1])-(covList[0][2]*covList[1][1]*covList[2][0]+covList[0][1]*covList[1][0]*covList[2][2]+covList[0][0]*covList[1][2]*covList[2][1])


        demon = sx[i]*sy[i]*sz[i]*torch.sqrt(((2*np.pi)**3)*det)
        result=result/demon

        # Numerical stability
        epsilon = 1e-20
        result = -(torch.log(torch.clamp(result, min=epsilon)))
        # print(result)

        loss = loss + result
        counter = counter+1
        # print(result)
    # print('----')
    # print(mux[0].data.cpu().numpy(), muy[0].data.cpu().numpy(), muz[0].data.cpu().numpy(), sx[0].data.cpu().numpy(), sy[0].data.cpu().numpy(), sz[0].data.cpu().numpy(), corr_xy[0].data.cpu().numpy(), corr_yz[0].data.cpu().numpy(), corr_xz[0].data.cpu().numpy())

    return (loss)/counter

def Gaussian2DLikelihoodInference(outputs, targets, nodesPresent, pred_length, look_up):
    '''
    Computes the likelihood of predicted locations under a bivariate Gaussian distribution at test time

    Parameters:

    outputs: Torch variable containing tensor of shape seq_length x numNodes x 1 x output_size
    targets: Torch variable containing tensor of shape seq_length x numNodes x 1 x input_size
    nodesPresent : A list of lists, of size seq_length. Each list contains the nodeIDs that are present in the frame
    '''
    seq_length = outputs.size()[0]
    obs_length = seq_length - pred_length

    # Extract mean, std devs and correlation
    mux, muy, sx, sy, corr = getCoef(outputs)

    # Compute factors
    normx = targets[:, :, 0] - mux
    normy = targets[:, :, 1] - muy
    sxsy = sx * sy

    z = (normx/sx)**2 + (normy/sy)**2 - 2*((corr*normx*normy)/sxsy)
    negRho = 1 - corr**2

    # Numerator
    result = torch.exp(-z/(2*negRho))
    # Normalization factor
    denom = 2 * np.pi * (sxsy * torch.sqrt(negRho))

    # Final PDF calculation
    result = result / denom

    # Numerical stability
    epsilon = 1e-20

    result = -torch.log(torch.clamp(result, min=epsilon))
    #print(result)

    loss = 0
    counter = 0

    for framenum in range(obs_length, seq_length):
        nodeIDs = nodesPresent[framenum]
        nodeIDs = [int(nodeID) for nodeID in nodeIDs]

        for nodeID in nodeIDs:

            nodeID = look_up[nodeID]
            loss = loss + result[framenum, nodeID]
            counter = counter + 1

    if counter != 0:
        return loss / counter
    else:
        return loss


def Gaussian2DLikelihood(outputs, targets, nodesPresent, look_up):
    '''
    params:
    outputs : predicted locations
    targets : true locations
    assumedNodesPresent : Nodes assumed to be present in each frame in the sequence
    nodesPresent : True nodes present in each frame in the sequence
    look_up : lookup table for determining which ped is in which array index

    '''
    seq_length = outputs.size()[0]
    # Extract mean, std devs and correlation
    mux, muy, sx, sy, corr = getCoef(outputs)

    # Compute factors
    normx = targets[:, :, 0] - mux
    normy = targets[:, :, 1] - muy
    sxsy = sx * sy

    z = (normx/sx)**2 + (normy/sy)**2 - 2*((corr*normx*normy)/sxsy)
    negRho = 1 - corr**2

    # Numerator
    result = torch.exp(-z/(2*negRho))
    # Normalization factor
    denom = 2 * np.pi * (sxsy * torch.sqrt(negRho))

    # Final PDF calculation
    result = result / denom

    # Numerical stability
    epsilon = 1e-20

    result = -torch.log(torch.clamp(result, min=epsilon))

    loss = 0
    counter = 0

    for framenum in range(seq_length):

        nodeIDs = nodesPresent[framenum]
        nodeIDs = [int(nodeID) for nodeID in nodeIDs]

        for nodeID in nodeIDs:
            nodeID = look_up[nodeID]
            loss = loss + result[framenum, nodeID]
            counter = counter + 1

    if counter != 0:
        return loss / counter
    else:
        return loss

##################### Data related methods ######################

def remove_file_extention(file_name):
    # remove file extension (.txt) given filename
    return file_name.split('.')[0]

def add_file_extention(file_name, extention):
    # add file extension (.txt) given filename

    return file_name + '.' + extention

def clear_folder(path):
    # remove all files in the folder
    if os.path.exists(path):
        shutil.rmtree(path)
        print("Folder succesfully removed: ", path)
    else:
        print("No such path: ",path)

def delete_file(path, file_name_list):
    # delete given file list
    for file in file_name_list:
        file_path = os.path.join(path, file)
        try:
            if os.path.isfile(file_path):
                os.remove(file_path)
                print("File succesfully deleted: ", file_path)
            else:    ## Show an error ##
                print("Error: %s file not found" % file_path)        
        except OSError as e:  ## if failed, report it back to the user ##
            print ("Error: %s - %s." % (e.filename,e.strerror))

def get_all_file_names(path):
    # return all file names given directory
    files = []
    for (dirpath, dirnames, filenames) in walk(path):
        files.extend(filenames)
        break
    return files

def create_directories(base_folder_path, folder_list):
    # create folders using a folder list and path
    for folder_name in folder_list:
        directory = os.path.join(base_folder_path, folder_name)
        if not os.path.exists(directory):
            os.makedirs(directory)

def unique_list(l):
  # get unique elements from list
  x = []
  for a in l:
    if a not in x:
      x.append(a)
  return x

def angle_between(p1, p2):
    # return angle between two points
    ang1 = np.arctan2(*p1[::-1])
    ang2 = np.arctan2(*p2[::-1])
    return ((ang1 - ang2) % (2 * np.pi))

def vectorize_seq(x_seq, PedsList_seq, lookup_seq):
    #substract first frame value to all frames for a ped.Therefore, convert absolute pos. to relative pos.
    first_values_dict = WriteOnceDict()
    vectorized_x_seq = x_seq.clone()
    for ind, frame in enumerate(x_seq):
        for ped in PedsList_seq[ind]:
            first_values_dict[ped] = frame[lookup_seq[ped], 0:2]
            vectorized_x_seq[ind, lookup_seq[ped], 0:2]  = frame[lookup_seq[ped], 0:2] - first_values_dict[ped][0:2]

    return vectorized_x_seq, first_values_dict

def translate(x_seq, PedsList_seq, lookup_seq, value):
    # translate al trajectories given x and y values
    vectorized_x_seq = x_seq.clone()
    for ind, frame in enumerate(x_seq):
        for ped in PedsList_seq[ind]:
            vectorized_x_seq[ind, lookup_seq[ped], 0:2]  = frame[lookup_seq[ped], 0:2] - value[0:2]

    return vectorized_x_seq

def revert_seq(x_seq, PedsList_seq, lookup_seq, first_values_dict):
    # convert velocity array to absolute position array
    absolute_x_seq = x_seq.clone()
    for ind, frame in enumerate(x_seq):
        for ped in PedsList_seq[ind]:
            absolute_x_seq[ind, lookup_seq[ped], 0:2] = frame[lookup_seq[ped], 0:2] + first_values_dict[ped][0:2]

    return absolute_x_seq


def rotate(origin, point, angle):
        """
        Rotate a point counterclockwise by a given angle around a given origin.

        The angle should be given in radians.
        """
        ox, oy = origin
        px, py = point

        qx = ox + math.cos(angle) * (px - ox) - math.sin(angle) * (py - oy)
        qy = oy + math.sin(angle) * (px - ox) + math.cos(angle) * (py - oy)
        #return torch.cat([qx, qy])
        return [qx, qy]

def time_lr_scheduler(optimizer, epoch, lr_decay=0.1, lr_decay_epoch=10):
    """Decay learning rate by a factor of lr_decay every lr_decay_epoch epochs"""
    if epoch == 0:
        return optimizer
    if epoch % lr_decay_epoch:
        return optimizer
    
    print("Optimizer learning rate has been decreased.")

    for param_group in optimizer.param_groups:
        param_group['lr'] *= (1. / (1. + lr_decay*10))
        print(param_group['lr'], "-*************************")
    return optimizer



