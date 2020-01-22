##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
## Created by: Albert Berenguel
## Computer Vision Center (CVC). Universitat Autonoma de Barcelona
## Email: aberenguel@cvc.uab.es
## Copyright (c) 2017
##
## This source code is licensed under the MIT-style license found in the
## LICENSE file in the root directory of this source tree
##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

import torch
import torch.nn as nn
from torch.autograd import Variable
import unittest


class BidirectionalLSTM(nn.Module):
    def __init__(self, layer_sizes, batch_size, vector_dim, num_layers=1, dropout=-1, layer_sizes_second_lstm=None, batch_size_second_lstm=None, vector_dim_second_lstm=None, num_layers_second_lstm=1, dropout_second_lstm=-1):
        super(BidirectionalLSTM, self).__init__()
        """
        Initializes a multi layer bidirectional LSTM
        :param layer_sizes: A list containing the neuron numbers per layer 
                            e.g. [100, 100, 100] returns a 3 layer, 100
        :param batch_size: The experiments batch size
        """
        self.batch_size = batch_size
        self.hidden_size = layer_sizes[0]
        self.vector_dim = vector_dim
        self.num_layers = len(layer_sizes)*num_layers
        
        '''
        input_size: The number of expected features in the input x
        hidden_size: The number of features in the hidden state h
        num_layers: Number of recurrent layers.
        bias: If False, then the layer does not use bias weights b_ih and b_hh. Default: True
        batch_first: If True, then the input and output tensors are provided as (batch, seq, feature)
        dropout: If non-zero, introduces a dropout layer on the outputs of each RNN layer except the last layer
        bidirectional: If True, becomes a bidirectional RNN. Default: False
        '''
        if dropout == -1:
            self.lstm = nn.LSTM(input_size=self.vector_dim,
                                num_layers=self.num_layers,
                                hidden_size=self.hidden_size,
                                bidirectional=True)
        else:
            self.lstm = nn.LSTM(input_size=self.vector_dim,
                                num_layers=self.num_layers,
                                hidden_size=self.hidden_size,
                                bidirectional=True,
                                dropout=dropout)
        print( self.lstm )
                            
        self.second_lstm = None
        if layer_sizes_second_lstm is not None:
            if dropout_second_lstm == -1:
                self.second_lstm = nn.LSTM(input_size=vector_dim_second_lstm,
                                    num_layers=len(layer_sizes_second_lstm)*num_layers_second_lstm,
                                    hidden_size=layer_sizes_second_lstm[0],
                                    bidirectional=True)
            else:
                self.second_lstm = nn.LSTM(input_size=vector_dim_second_lstm,
                                    num_layers=len(layer_sizes_second_lstm)*num_layers_second_lstm,
                                    hidden_size=layer_sizes_second_lstm[0],
                                    bidirectional=True,
                                    dropout=dropout)
            print( self.second_lstm )

        
    def forward(self, inputs):
        """
        Runs the bidirectional LSTM, produces outputs and saves both forward and backward states as well as gradients.
        :param x: The inputs should be a list of shape [sequence_length, batch_size, 64]
        :return: Returns the LSTM outputs, as well as the forward and backward hidden states.
        """
        if torch.cuda.is_available():
            #c0 = Variable(torch.rand(self.lstm.num_layers*2, self.batch_size, self.lstm.hidden_size),requires_grad=False).cuda()
            c0 = Variable(torch.rand(self.lstm.num_layers*2, 1, self.lstm.hidden_size),requires_grad=False).cuda()
            #h0 = Variable(torch.rand(self.lstm.num_layers*2, self.batch_size, self.lstm.hidden_size),requires_grad=False).cuda()
            h0 = Variable(torch.rand(self.lstm.num_layers*2, 1, self.lstm.hidden_size),requires_grad=False).cuda()
        else:
            #c0 = Variable(torch.rand(self.lstm.num_layers*2, self.batch_size, self.lstm.hidden_size),requires_grad=False).cuda()
            c0 = Variable(torch.rand(self.lstm.num_layers*2, 1, self.lstm.hidden_size),requires_grad=False)
            #h0 = Variable(torch.rand(self.lstm.num_layers*2, self.batch_size, self.lstm.hidden_size),requires_grad=False).cuda()
            h0 = Variable(torch.rand(self.lstm.num_layers*2, 1, self.lstm.hidden_size),requires_grad=False)
        #print("c0 ", c0.shape, " h0 ", h0.shape)
        output, (hn, cn) = self.lstm(inputs, (h0, c0))
        #print( "inputs.shape ", inputs.shape, output.shape )
        
        if self.second_lstm is not None:
            if torch.cuda.is_available():
                #c0 = Variable(torch.rand(self.lstm.num_layers*2, self.batch_size, self.lstm.hidden_size),requires_grad=False).cuda()
                c0 = Variable(torch.rand(self.second_lstm.num_layers*2, 1, self.second_lstm.hidden_size),requires_grad=False).cuda()
                #h0 = Variable(torch.rand(self.lstm.num_layers*2, self.batch_size, self.lstm.hidden_size),requires_grad=False).cuda()
                h0 = Variable(torch.rand(self.second_lstm.num_layers*2, 1, self.second_lstm.hidden_size),requires_grad=False).cuda()
            else:
                #c0 = Variable(torch.rand(self.lstm.num_layers*2, self.batch_size, self.lstm.hidden_size),requires_grad=False).cuda()
                c0 = Variable(torch.rand(self.second_lstm.num_layers*2, 1, self.second_lstm.hidden_size),requires_grad=False)
                #h0 = Variable(torch.rand(self.lstm.num_layers*2, self.batch_size, self.lstm.hidden_size),requires_grad=False).cuda()
                h0 = Variable(torch.rand(self.second_lstm.num_layers*2, 1, self.second_lstm.hidden_size),requires_grad=False)
            #print("c0 ", c0.shape, " h0 ", h0.shape)
            output, (hn, cn) = self.second_lstm(output, (h0, c0))
        
        return output, hn, cn


class BidirectionalLSTMTest(unittest.TestCase):
    def setUp(self):
        pass
    def tearDown(self):
        pass

    def test_forward(self):
        pass


if __name__ == '__main__':
    unittest.main()

