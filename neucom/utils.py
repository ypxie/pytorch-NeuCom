import numpy as np
import torch
import torch.nn.functional as F
from torch.autograd import Variable
import torch.nn as nn

def pairwise_add(u, v=None, is_batch=False):
    """
    performs a pairwise summation between vectors (possibly the same)

    Parameters:
    ----------
    u: Tensor (m, ) | (m, 1)
    v: Tensor (n, ) | (n, 1) [optional]
    is_batch: bool
        a flag for whether the vectors come in a batch
        ie.: whether the vectors has a shape of (b,n) or (b,n,1)

    Returns: Tensor (m, n) or (b, m, n)
    Raises: ValueError
    """
    u_shape = u.size()

    if len(u_shape) > 2 and not is_batch:
        raise ValueError("Expected at most 2D tensors, but got %dD" % len(u_shape))
    if len(u_shape) > 3 and is_batch:
        raise ValueError("Expected at most 2D tensor batches, but got %dD" % len(u_shape))

    if v is None:
        v = u
    v_shape = v.size()

    m = u_shape[0] if not is_batch else u_shape[1]
    n = v_shape[0] if not is_batch else v_shape[1]
    
    u = expand_dims(u, axis=-1)
    new_u_shape = u.size()
    new_u_shape[-1] = n
    U = u.view(*new_u_shape)

    v = expand_dims(v, axis=-2)
    new_v_shape = v.size()
    new_v_shape[-2] = m
    V = v.view(*new_v_shape)

    return U + V

def to_device(src, ref):
    return src.cuda(ref.get_device()) if ref.is_cuda else src

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
    if axis < 0:
        axis = len(input_shape) + axis + 1
    new_shape = input_shape.insert(axis, 1)
    return input.view(*new_shape)


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
    memory_norm = expand_dims(torch.norm(memory_matrix, 2, 2),axis = 2)
    keys_norm = expand_dims(torch.norm(keys, 2, 1), axis = 1)

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
