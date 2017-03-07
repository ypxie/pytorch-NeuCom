import torch
from torch.autograd import Variable
import  torch.nn.functional as F
from copy import copy
import numpy as np
import torch.nn as nn
from collections import namedtuple
from neucom.utils import *
from neucom.memory import Memory

class DNC(nn.Module):
    def __init__(self, nhid=64, nn_output_size = 64, nlayer=1, controller_class = None, 
                 input_size = 10, output_size = 10, mem_slot = 256, 
                 mem_size = 64, read_heads = 4, batch_size = 1, use_cuda=True):
        """
        The agent class.

        Parameters:
        -----------
        controller_class: BaseController
            a concrete implementation of the BaseController class
        input_size: int
            the size of the input vector
        output_size: int
            the size of the output vector
        max_sequence_length: int
            the maximum length of an input sequence
        mem_slot: int
            the number of memory slots that can be stored in memory
        mem_size: int
            the size of an individual word in memory
        read_heads: int
            the number of read heads in the memory
        batch_size: int
            the size of the data batch
        """
        self.__dict__.update(locals())
        super(DNC, self).__init__()

        self.memory = Memory(self.mem_slot, self.mem_size,
                             self.read_heads, self.batch_size, use_cuda = self.use_cuda)

        self.controller = self.controller_class(
                                nhid           = self.nhid,
                                nlayer         = self.nlayer,
                                input_size     = self.input_size, 
                                output_size    = self.output_size,
                                read_heads     = self.read_heads,
                                nn_output_size = self.nn_output_size,
                                mem_size       = self.mem_size, 
                                batch_size     = self.batch_size
                            )
        #apply_dict(locals())
        
    def forward(self, input_data, mask=None):
        time_step = input_data.size()[0]
        batch_size = input_data.size()[1]
        
        memory_state = self.memory.init_memory(batch_size)
        controller_state = self.controller.get_state(batch_size) if \
                           self.controller.recurrent \
                           else (torch.zeros(1), torch.zeros(1))

        outputs_time = [[]]*time_step
        free_gates_time = [[]]*time_step
        allocation_gates_time = [[]]*time_step
        write_gates_time = [[]]*time_step
        read_weights_time = [[]]*time_step
        write_weights_time = [[]]*time_step
        usage_vectors_time = [[]]*time_step
        
        last_read_vectors = memory_state.read_vec
        mem_mat = memory_state.mem_mat
        mem_usage = memory_state.mem_usage
        read_weight = memory_state.read_weight
        write_weight = memory_state.write_weight
        pre_vec = memory_state.pre_vec
        link_mat = memory_state.link_mat

        pre_output, interface, nn_state = None, None, None
        #TODO: perform matmul(input, W) before loops
        
        for time in range(time_step):
            step_input = input_data[time]
                  
            if self.controller.recurrent:
                pre_output, interface, nn_state = self.controller.process_input(step_input, \
                                                    last_read_vectors, controller_state)
            else:
                pre_output, interface = self.controller.process_input(step_input, last_read_vectors)
            
            usage_vector, write_weight, mem_mat, \
            link_mat, pre_vec = self.memory.write \
                                            (
                                                mem_mat, 
                                                mem_usage, 
                                                read_weight,
                                                write_weight, 
                                                pre_vec, 
                                                link_mat,
                                                
                                                interface['write_key'],
                                                interface['write_strength'],
                                                interface['free_gates'],
                                                interface['allocation_gate'],
                                                interface['write_gate'],
                                                interface['write_vector'],
                                                interface['erase_vector']
                                            )
                                            
            read_weight, last_read_vectors = self.memory.read \
                                            (
                                                mem_mat,
                                                read_weight,
                                                interface['read_keys'],
                                                interface['read_strengths'],
                                                link_mat,
                                                interface['read_modes'],
                                            )
            
            
            outputs_time[time] =  self.controller.final_output(pre_output, last_read_vectors).clone()
            free_gates_time[time] = interface['free_gates'].clone()
            allocation_gates_time[time] = interface['allocation_gate'].clone()
            write_gates_time[time] = interface['write_gate'].clone()
            read_weights_time[time] = read_weight.clone()
            write_weights_time[time] = write_weight.clone()
            usage_vectors_time[time] = usage_vector.clone()
            
            controller_state = (
                                nn_state[0] if nn_state is not None else torch.zeros(1),
                                nn_state[1] if nn_state is not None else torch.zeros(1)
                                )

        

        packed_output = torch.stack(outputs_time)
        packed_memory_view = {
            'free_gates':       torch.stack(free_gates_time),
            'allocation_gates': torch.stack(allocation_gates_time),
            'write_gates':      torch.stack(write_gates_time),
            'read_weights':     torch.stack(read_weights_time),
            'write_weights':    torch.stack(write_weights_time),
            'usage_vectors':    torch.stack(usage_vectors_time)
        }

        #apply_dict(locals())
        return   packed_output,  packed_memory_view 

    def save(self, ckpts_dir, name):
        raise NotImplementedError
    
    def restore(self, ckpts_dir, name):
        raise NotImplementedError

    #def __call__(self,*args, **kwargs):
    #    return self.forward(*args, **kwargs)


