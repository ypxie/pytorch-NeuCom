import torch
from torch.autograd import Variable
import  torch.nn.functional as F
import torch.nn as nn
import numpy as np
from .neucom.utils import *

class BaseController(nn.Module):
    def __init__(self, input_size, output_size, read_heads, nn_output_size,
                 mem_size, batch_size=1, recurrent=True):
        """
        constructs a controller as described in the DNC paper:
        http://www.nature.com/nature/journal/vaop/ncurrent/full/nature20101.html

        Parameters:
        ----------
        input_size: int
            the size of the data input vector
        output_size: int
            the size of the data output vector
        memory_read_heads: int
            the number of read haeds in the associated external memory
        mem_size: int
            the size of the word in the associated external memory
        batch_size: int
            the size of the input data batch [optional, usually set by the DNC object]
        """
        #self.__dict__.update(locals())
        self.input_size = input_size
        self.output_size = output_size
        self.read_heads = read_heads
        self.mem_size = mem_size
        #size of the internal neural network
        self.nn_output_size = nn_output_size 
        self.recurrent = recurrent

        #nn_input_size should be infered from input
        self.nn_input_size = self.mem_size * self.read_heads + self.input_size
        #ToDo: this need to be adjust
        self.interface_vector_size = self.mem_size * self.read_heads + 3 * self.mem_size + 5 * self.read_heads + 3

        self.init_weights()
    
    def init_weights():
        '''
        set the intial values of the controller
        '''
        raise NotImplemented('network init_weights is not implemented.')
    
    def parse_interface_vector(self, interface_vector):
        """
        pasres the flat interface_vector into its various components with their
        correct shapes

        Parameters:
        ----------
        interface_vector: Tensor (batch_size, interface_vector_size)
            the flattened inetrface vector to be parsed

        Returns: dict
            a dictionary with the components of the interface_vector parsed
        """

        parsed = {}
        r_keys_end = self.mem_size * self.read_heads
        r_strengths_end = r_keys_end + self.read_heads
        w_key_end = r_strengths_end + self.mem_size
        erase_end = w_key_end + 1 + self.mem_size
        write_end = erase_end + self.mem_size
        free_end = write_end + self.read_heads

        r_keys_shape = (-1, self.mem_size, self.read_heads)
        r_strengths_shape = (-1, self.read_heads)
        w_key_shape = (-1, self.mem_size, 1)
        write_shape = erase_shape = (-1, self.mem_size)
        free_shape = (-1, self.read_heads)
        modes_shape = (-1, 3, self.read_heads)

        # parsing the vector into its individual components
        parsed['read_keys'] = interface_vector[:,0:r_keys_end].view(*r_keys_shape)
        parsed['read_strengths'] = interface_vector[:, r_keys_end:r_strengths_end].view(*r_strengths_shape)
        parsed['write_key'] = interface_vector[:, r_strengths_end:w_key_end].view(*w_key_shape)
        parsed['write_strength'] = interface_vector[:, w_key_end].view(-1, 1)
        parsed['erase_vector'] = interface_vector[:, w_key_end + 1:erase_end].view(*erase_shape)
        parsed['write_vector'] = interface_vector[:, erase_end:write_end].view(*write_shape)
        parsed['free_gates'] = interface_vector[:, write_end:free_end].view(*free_shape)
        parsed['allocation_gate'] = expand_dims(interface_vector[:, free_end], 1)
        parsed['write_gate'] = expand_dims(interface_vector[:, free_end + 1], 1)
        parsed['read_modes'] = interface_vector[:, free_end + 2:].view(*modes_shape)

        # transforming the components to ensure they're in the right ranges
        parsed['read_strengths'] = 1 + F.softplus(parsed['read_strengths'])
        parsed['write_strength'] = 1 + F.softplus(parsed['write_strength'])
        parsed['erase_vector'] = F.sigmoid(parsed['erase_vector'])
        parsed['free_gates'] =  F.sigmoid(parsed['free_gates'])
        parsed['allocation_gate'] =  F.sigmoid(parsed['allocation_gate'])
        parsed['write_gate'] =  F.sigmoid(parsed['write_gate'])
        parsed['read_modes'] = F.softmax(parsed['read_modes'], 1)

        return parsed

    def process_input(self, X, last_read_vectors, state=None):
        """
            processes input data through the controller network and returns the
            pre-output and interface_vector

            Parameters:
            ----------
            X: Tensor (batch_size, input_size)
                the input data batch
            last_read_vectors: (batch_size, mem_size, read_heads)
                the last batch of read vectors from memory
            state: Tuple
                state vectors if the network is recurrent

            Returns: Tuple
                pre-output: Tensor (batch_size, output_size)
                parsed_interface_vector: dict
        """

        flat_read_vectors = last_read_vectors.view(-1, self.mem_size * self.read_heads)
        complete_input = torch.cat( [X, flat_read_vectors], 1)
        nn_output, nn_state = None, None

        if self.recurrent:
            nn_output, nn_state = self.network_op(complete_input, state)
        else:
            nn_output = self.network_op(complete_input)

        pre_output = torch.mm(nn_output, self.nn_output_weights)
        interface = torch.mm(nn_output, self.interface_weights)
        parsed_interface = self.parse_interface_vector(interface)

        if self.recurrent:
            return pre_output, parsed_interface, nn_state
        else:
            return pre_output, parsed_interface


    def final_output(self, pre_output, new_read_vectors):
        """
        returns the final output by taking rececnt memory changes into account

        Parameters:
        ----------
        pre_output: Tensor (batch_size, output_size)
            the ouput vector from the input processing step
        new_read_vectors: Tensor (batch_size, words_size, read_heads)
            the newly read vectors from the updated memory

        Returns: Tensor (batch_size, output_size)
        """

        flat_read_vectors = new_read_vectors.view(-1, self.mem_size * self.read_heads)

        final_output = pre_output + torch.mm(flat_read_vectors, self.mem_output_weights)

        return final_output
