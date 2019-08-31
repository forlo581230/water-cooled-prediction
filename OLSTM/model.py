import torch
import torch.nn as nn
import numpy as np
from torch.autograd import Variable

class OLSTM(nn.Module):
    def __init__(self, args, infer=False):
        super(OLSTM, self).__init__()


        self.embedding_size = 128
        self.input_size = args.input_size
        self.output_size = args.output_size
        self.rnn_size = args.rnn_size

        # Linear layer to embed the input position
        self.input_embedding_layer = nn.Linear(self.input_size, self.embedding_size)
        # ReLU and dropout unit
        self.relu = nn.ReLU()

        self.rnn = nn.LSTM(
            self.embedding_size,
            self.rnn_size,
            3,
            batch_first=True
        )
        self.output = nn.Linear(self.embedding_size, args.output_size)
        self.dropout = nn.Dropout(0.2)

    def forward(self, data, h0, c0):
        # Variable(torch.zeros(numNodes, args.rnn_size)).cuda()
        # seq = input.shape[0]

        input_embedded =  self.dropout(self.relu(self.input_embedding_layer(data)))
        output, (hn, cn) = self.rnn(input_embedded,(h0, c0))
        # output, _ = self.rnn(input)
        output =self.output(output[:, :, :])
        
        return output, hn, cn

class VLSTM(nn.Module):
    def __init__(self, args, infer=False):
        super(VLSTM, self).__init__()

        self.embedding_size = 128
        self.input_size = args.input_size
        self.output_size = args.output_size
        self.rnn_size = args.rnn_size

        self.rnn = nn.LSTM(
            self.input_size,
            self.rnn_size,
            3,
            batch_first=True
        )
        self.output = nn.Linear(self.embedding_size, args.output_size)
        self.dropout = nn.Dropout(0.2)

    def forward(self, data, h0, c0):

        output, (hn, cn) = self.rnn(data,(h0, c0))
        output =self.output(output[:, :, :])
        
        return output, hn, cn

class VGRU(nn.Module):
    def __init__(self, args, infer=False):
        super(VGRU, self).__init__()

        self.embedding_size = 128
        self.input_size = args.input_size
        self.output_size = args.output_size
        self.rnn_size = args.rnn_size

        self.rnn = nn.GRU(
            self.input_size,
            self.rnn_size,
            3,
            batch_first=True
        )
        self.output = nn.Linear(self.embedding_size, args.output_size)
        self.dropout = nn.Dropout(0.2)

    def forward(self, data, h0, c0):

        output, hn = self.rnn(data,h0)
        output =self.output(output[:, :, :])
        
        return output, hn, None

# class LSTM(nn.Module):
#     def __init__(self, args, infer=False):
#         super(LSTM, self).__init__()

#         self.use_cuda = args.use_cuda

#         self.embedding_size = 64
#         self.input_size = args.input_size
#         self.output_size = args.output_size
#         self.rnn_size = args.rnn_size

#         # Linear layer to embed the input position
#         self.input_embedding_layer = nn.Linear(self.input_size, self.embedding_size)
#         self.residuals_embedding_layer = nn.Linear(2*self.rnn_size, self.embedding_size)
#         # ReLU and dropout unit
#         self.relu = nn.ReLU()

#         # The LSTM cell
#         self.cell = nn.LSTMCell(2*self.embedding_size, self.rnn_size)

#         self.output_layer = nn.Linear(self.rnn_size, self.output_size)
#         self.dropout = nn.Dropout(0.2)

#     def get_residuals(self, residulas, hidden_states):
#         '''
#         Computes the social tensor for a given grid mask and hidden states of all peds
#         params:
#         grid : Grid masks
#         hidden_states : Hidden states of all peds
#         '''

#         # Construct the variable
#         residuals_tensor = Variable(torch.zeros(residulas.size()[0], 2, self.rnn_size))
#         if self.use_cuda:
#             residuals_tensor = residuals_tensor.cuda()
        
#         for i in range(residulas.size()[0]):
#             residuals_tensor[i] = torch.mm(torch.t(residulas[i]), hidden_states[i:i+1])

#         # Reshape the social tensor
#         residuals_tensor = residuals_tensor.view(residulas.size()[0], 1, 2*self.rnn_size)

#         return residuals_tensor

#     def forward(self, data, residuals, h0, c0):
#         # Variable(torch.zeros(numNodes, args.rnn_size)).cuda()
#         # seq = input.shape[0]

#         seq_length = data.size()[1]
#         outputs = Variable(torch.zeros(data.size()[0], seq_length, self.output_size))

#         if self.use_cuda:
#             outputs = outputs.cuda()

#         for framenum in range(seq_length):
            
#             residulas_current = residuals[:, framenum:framenum+1]

#             # print(residulas_current.size(),"++++")

#             # Compute the social tensor
#             residuals_tensor = self.get_residuals(residulas_current, h0)

#             # Embed inputs
#             input_embedded = self.dropout(self.relu(self.input_embedding_layer(data[:, framenum:framenum+1,:])))
#             # Embed the social tensor
#             residuals_embedded = self.dropout(self.relu(self.residuals_embedding_layer(residuals_tensor)))

            
#             # Concat input
#             concat_embedded = torch.cat((input_embedded, residuals_embedded), 2)
            
#             h_nodes, c_nodes = self.cell(concat_embedded[:,0,:], (h0, c0))

#             # Compute the output
#             outputs[:,framenum] = self.output_layer(h_nodes)

#             # Update hidden and cell states
#             hidden_states = h_nodes
#             cell_states = c_nodes

#         return outputs, hidden_states, cell_states