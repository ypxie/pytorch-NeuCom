import torch
from torch.autograd import Variable
import  torch.nn.functional as F
import torch.nn as nn

import numpy as np
from collections import namedtuple
from neucom.utils import *

class Memory(nn.Module):
    def __init__(self,mem_slot=256, mem_size=64, read_heads=4, batch_size=1):
        """
        constructs a memory matrix with read heads and a write head as described
        in the DNC paper
        http://www.nature.com/nature/journal/vaop/ncurrent/full/nature20101.html

        Parameters:
        ----------
        words_num: int
            the maximum number of words that can be stored in the memory at the
            same time
        mem_size: int
            the size of the individual word in the memory
        read_heads: int
            the number of read heads that can read simultaneously from the memory
        batch_size: int
            the size of input data batch
        """

        self.__dict__.update(locals())
        self.I = Variable(torch.eye(mem_slot))

        self.memory_tuple = namedtuple('mem_tuple', 'mem_mat, mem_usage, pre_vec, \
                                        link_mat, write_weight, read_weight, read_vec')

    def init_memory(self):
        """
        return a tuple of the intial values pertinetn to 
        the memorys
        Returns: namedtuple('mem_tuple', 'mem_mat, mem_usage, pre_vec, \
                            link_mat, write_weight, read_weight, read_vec')
        """
        mem_list = [Variable(torch.zeros(self.batch_size, self.mem_slot, self.mem_size).fill_(1e-6)), #initial memory matrix
            Variable(torch.zeros(self.batch_size, self.mem_slot)), #initial memory usage vector
            Variable(torch.zeros(self.batch_size, self.mem_slot)), #initial precedence vector
            Variable(torch.zeros(self.batch_size, self.mem_slot, self.mem_slot)), #initial link matrix
            
            Variable(torch.zeros(self.batch_size, self.mem_slot).fill_(1e-6)), #initial write weighting
            Variable(torch.zeros(self.batch_size, self.mem_slot, self.read_heads).fill_(1e-6)), #initial read weighting
            Variable(torch.zeros(self.batch_size, self.mem_size, self.read_heads).fill_(1e-6))] #initial read vector
        return self.memory_tuple._make(mem_list)

    def get_content_address(self, memory_matrix, keys, strengths):
        """
        retrives a content-based adderssing weighting given the keys

        Parameters:
        ----------
        memory_matrix: Tensor (batch_size, mem_slot, mem_size)
            the memory matrix to lookup in
        keys: Tensor (batch_size, mem_size, number_of_keys)
            the keys to query the memory with
        strengths: Tensor (batch_size, number_of_keys, )
            the list of strengths for each lookup key
        
        Returns: Tensor (batch_size, mem_slot, number_of_keys)
            The list of lookup weightings for each provided key
        """
        # cos_dist is (batch_size, mem_slot, number_of_keys)
        cos_dist = cosine_distance(memory_matrix, keys)
        
        strengths = expand_dims(strengths, 1)

        return softmax(cos_dist*strengths, 1)


mem = Memory()
memo_state = mem.init_memory()

print memo_state