import torch
from torch.autograd import Variable
import  torch.nn.functional as F
import torch.nn as nn

import numpy as np
from collections import namedtuple
from neucom.utils import *
from neucom.memory import Memory

class NeuCom(nn.Module):
    def __init__(self, controller_class, input_size, output_size,
                 mem_slot = 256, mem_size = 64, read_heads = 4, batch_size = 1):
        """
        constructs a complete DNC architecture as described in the DNC paper
        http://www.nature.com/nature/journal/vaop/ncurrent/full/nature20101.html

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
            the number of words that can be stored in memory
        mem_size: int
            the size of an individual word in memory
        read_heads: int
            the number of read heads in the memory
        batch_size: int
            the size of the data batch
        """
        self.__dict__.update(locals())

        self.memory = Memory(self.mem_slot, self.mem_size, 
                             self.read_heads, self.batch_size)

        self.controller = controller_class(self.input_size, self.output_size, 
                                            self.read_heads, self.mem_size, self.batch_size)
    
    def _step_op(self, step_input, memory_state, controller_state=None):
        """
        performs a step operation on the input step data

        Parameters:
        ----------
        step_input: Tensor (batch_size, input_size)
        memory_state: Tuple
            a tuple of current memory parameters
        controller_state: Tuple
            the state of the controller if it's recurrent

        Returns: Tuple
            output: Tensor (batch_size, output_size)
            memory_view: dict
        """

        #namedtuple('mem_tuple', 'mem_mat, mem_usage, pre_vec, \
        #                    link_mat, write_weight, read_weight, read_vec')
        last_read_vectors = memory_state.read_vec
        pre_output, interface, nn_state = None, None, None
        
        if self.controller.recurrent:
            pre_output, interface, nn_state = self.controller.process_input(step_input, \
                                                   last_read_vectors, controller_state)
        else:
            pre_output, interface = self.controller.process_input(step_input, last_read_vectors)

        usage_vector, write_weight, memory_matrix, \
        link_matrix, precedence_vector = self.memory.write \
        (
            memory_state.mem_mat, 
            memory_state.mem_usage, 
            memory_state.read_weight,
            memory_state.write_weight, 
            memory_state.pre_vec, 
            memory_state.link_mat,
            
            interface['write_key'],
            interface['write_strength'],
            interface['free_gates'],
            interface['allocation_gate'],
            interface['write_gate'],
            interface['write_vector'],
            interface['erase_vector']
        )

        read_weights, read_vectors = self.memory.read(
            memory_matrix,
            memory_state.read_weight,
            interface['read_keys'],
            interface['read_strengths'],
            link_matrix,
            interface['read_modes'],
        )

        return [
            # report new memory state to be updated outside the condition branch
            memory_matrix,
            usage_vector,
            precedence_vector,
            link_matrix,
            write_weight,
            read_weights,
            read_vectors,

            self.controller.final_output(pre_output, read_vectors),
            interface['free_gates'],
            interface['allocation_gate'],
            interface['write_gate'],

            # report new state of RNN if exists
            nn_state[0] if nn_state is not None else torch.zeros(1),
            nn_state[1] if nn_state is not None else torch.zeros(1)
        ]

    def forward(self, input, mask=None):
        time_step = input.size()[0]
        memory_state = self.memory.init_memory()
        controller_state = self.controller.get_state() if \
                           self.controller.recurrent \
                           else (torch.zeros(1), torch.zeros(1))

        outputs_time = input.new(self.sequence_length)
        free_gates_time = input.new(self.sequence_length)
        allocation_gates_time = input.new( self.sequence_length)
        write_gates_time = input.new(self.sequence_length)
        read_weights_time = input.new( self.sequence_length)
        write_weights_time = input.new( self.sequence_length)
        usage_vectors_time = input.new(self.sequence_length)
        
        last_read_vectors = memory_state.read_vec
        mem_mat = memory_state.mem_mat
        mem_usage = memory_state.mem_mat
        read_weight = memory_state.read_weight
        write_weight = memory_state.write_weight
        pre_vec = memory_state.pre_vec
        link_mat = memory_state.link_mat

        pre_output, interface, nn_state = None, None, None
        #TODO: perform matmul(input, W) before loop
        
        for time in xrange(time_step):
            step_input = input[time]
            output_list = self._step_op(step_input, 
                          memory_state, controller_state)
            
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
            mem_list = [mem_mat, mem_usage, pre_vec, \
                        link_mat, write_weight, read_weight, last_read_vectors]     
            memory_state = self.memory.memory_tuple._make(mem_list)      

            outputs_time[time] =  self.controller.final_output(pre_output, last_read_vectors)
            free_gates_time[time] = interface['free_gates']
            allocation_gates_time[time] = interface['allocation_gate']
            write_gates_time[time] = interface['write_gate']
            read_weights_time[time] = read_weight
            write_weights_time[time] = write_weight
            usage_vectors_time[time] = usage_vector
            
            controller_state = (
                                nn_state[0] if nn_state is not None else torch.zeros(1),
                                nn_state[1] if nn_state is not None else torch.zeros(1)
                                )

        self.packed_output = outputs_time
        self.packed_memory_view = {
            'free_gates':       free_gates_time,
            'allocation_gates': allocation_gates_time,
            'write_gates':      write_gates_time,
            'read_weights':     read_weights_time,
            'write_weights':    write_weights_time,
            'usage_vectors':    usage_vectors_time
        }
        return   self.packed_output,  self.packed_memory_view 

    def get_output(self):
        """
        returns the graph nodes for the output and memory view

        Returns: Tuple
            outputs: Tensor (batch_size, time_steps, output_size)
            memory_view: dict
        """
        return self.packed_output, self.packed_memory_view
    
    def save(self, ckpts_dir, name):
        raise NotImplementedError
    
    def restore(self, ckpts_dir, name):
        raise NotImplementedError

    def __call__(self,**kwargs):
        return self.forward(**kwargs)


