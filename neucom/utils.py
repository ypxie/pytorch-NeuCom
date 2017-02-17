import numpy as np
import torch
import torch.nn.functional as F
from torch.autograd import Variable
import torch.nn as nn

class MyModule(nn.Module):
    def __init__(self):
        super(MyModule, self).__init__()
    
    def get_parameters(self, name, tensor):
        if not name in self._parameters:
            self.register_parameter(name, nn.Parameter(tensor))
        return self._parameter[name]

def cumprod(inputs, dim = 1, exclusive=True):
    if type(inputs) is not Variable:
        temp = torch.cumprod(inputs, dim)
        if not exclusive:
            return temp
        else:
            temp =  temp / (input[0].expand_dims(0).expand_as(temp) + 1e-8)
            temp[-1] = temp[-1]/(input[-1]+1e-8)
            return temp
    else:
        shape_ = inputs.size()
        ndim = len(shape_)
        n_slot = shape_[dim]
        output = Variable(inputs.data.new(*shape_).fill_(1.0))
        slice_ = [slice(0,None,1) for _ in range(ndim)]
        for ind in range(1, n_slot):
            this_slice, last_slice = slice_.copy(), slice_.copy()
            this_slice[dim] = ind
            last_slice[dim] = ind-1
            
           
            
            this_slice = tuple(this_slice)
            last_slice = tuple(last_slice)
            print(output[this_slice])
            #torch.addcmul(output[this_slice], 1, output[last_slice], inputs[this_slice])
            output[this_slice]  = output[last_slice]*inputs[this_slice]
        if exclusive:
            return output
        else:
            first_slice = slice_.copy()
            first_slice[dim] = 0
            fist_input = input(tuple(first_slice))
            return output*fist_input.expand_dims(dim).expand_as(output)
            
def expand_dims(input, axis=0):
    input_shape = list(input.size())
    input_shape.insert(axis, 1)
    return input.view(*input_shape)

def matmal(left, right):
    '''
    left is of size (*N, n1,n2), where N is a list
    right is of size(*M, m1,m2), where M is a list
    output is of size
    '''
    pass

def cosine_distance(memory_matrix, keys):
    """
    compute the cosine similarity between keys to each of the 
    memory slot.

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
    memory_norm = torch.norm(memory_matrix, 2, 2)
    keys_norm   = torch.norm(keys, 2, 1)

    normalized_mem = torch.div(memory_matrix, memory_norm.expand_as(memory_matrix) + 1e-9)
    normalized_keys = torch.div(keys,keys_norm.expand_as(keys) + 1e-9)

    return torch.bmm(normalized_mem, normalized_keys)

def softmax(input, axis=1):
    """ 
    Apply softmax on input at certain axis.
    
    Parammeters:
    ----------
    input: Tensor (N*L or rank>2)
    axis: the axis to apply softmax
    
    Returns: Tensor with softmax applied on that dimension.
    """
    
    input_size = input.size()
    
    trans_input = input.transpose(axis, len(input_size)-1)
    trans_size = trans_input.size()

    input_2d = trans_input.contiguous().view(-1, trans_size[-1])
    soft_max_2d = F.softmax(input_2d)
    
    soft_max_nd = soft_max_2d.view(*trans_size)
    
    return soft_max_nd.transpose(axis, len(input_size)-1)
