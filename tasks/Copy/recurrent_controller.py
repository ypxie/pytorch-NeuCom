import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from neucom.controller import BaseController
from torch.autograd import Variable
from neucom.utils import *

class RecurrentController(BaseController):
    def __init__(self, nhid = 64, nlayer=1, **kwargs):
        super(RecurrentController, self).__init__(**kwargs)

        self.nhid = nhid
        initrange = 0.1
        ninp = self.nn_input_size
        #for id in range(nlayer):
        self.W_lstm =  nn.Parameter( (torch.randn(ninp, nhid * 4)).uniform_(-initrange, initrange) )
        self.U_lstm =  nn.Parameter( (torch.eye(nhid, nhid * 4)))
        self.b_lstm =  nn.Parameter( (torch.zeros(1, nhid *4)))
        
        self.hid_out = nn.Parameter( (torch.randn(nhid, self.nn_output_size)).uniform_(-initrange, initrange) )

    def network_op(self, X, states, mask=None):
        """
        perform one step of the forward computation.abs
        
        Parameters:
        ----------
        X     --- Tensor (batch_size, input_dim)
        states --- tuple of tensor (ten_1, ten_2), each of size (batch_size, hid_dim)
        mask  --- Tensor (batch_size,1) 
 
        Returns:
        -------
        output ---
        updated tuple -----
        """
        #it seems that using relu will result to nan error in cosine_distance, all 0 key is fatal.
        h_tm1 = states[0]
        c_tm1 = states[1]

        Z = torch.mm(X, self.W_lstm) + torch.mm(h_tm1, self.U_lstm)
        Z = self.b_lstm.expand_as(Z) + Z

        z0 = Z[:, :self.nhid]
        z1 = Z[:, self.nhid: 2 * self.nhid]
        z2 = Z[:, 2 * self.nhid: 3 * self.nhid]
        z3 = Z[:, 3 * self.nhid:]
        
        i = F.sigmoid(z0)
        f = F.sigmoid(z1)
        c = f * c_tm1 + i * F.tanh(z2)
        o = F.sigmoid(z3)
        
        h = o* F.tanh(c)
        out = F.tanh(torch.mm(h, self.hid_out))
        return out, (h, c)

    def get_state(self, batch_size):
        h = Variable(self.W_lstm.data.new(batch_size, self.nhid).fill_(0.0), requires_grad = True)
        c = Variable(self.W_lstm.data.new(batch_size, self.nhid).fill_(0.0), requires_grad = True)
        return (h,c)
        

        
