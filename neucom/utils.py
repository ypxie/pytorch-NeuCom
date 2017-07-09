from __future__ import print_function
import numpy as np
from copy import copy
import torch
import torch.nn.functional as F
from torch.autograd import Variable
import torch.nn as nn

def apply_var(v, k):
    if isinstance(v, Variable) and v.requires_grad:
            v.register_hook(inves(k))
            
def apply_dict(dic):
    for k, v in dic.iteritems():
        apply_var(v, k)
        if isinstance(v, nn.Module):
           key_list = [a for a in dir(v) if not a.startswith('__')]
           for key in key_list:
               apply_var(getattr(v, key), key)
           for pk, pv in v._parameters.iteritems():
               apply_var(pv, pk)
        
def inves(name=''):
    def f(tensor):
        if np.isnan(torch.mean(tensor).data.cpu().numpy() ):
            print('\ngradient of {} :'.format(name))
            print(tensor)
            assert 0, 'nan gradient'
            return tensor
    return f
        
def reduce_sum(inputs, dim=None, keep_dim=False):
    if dim is None:
        return torch.sum(inputs)
    output = torch.sum(inputs, dim)
    if not keep_dim:
        return output
    else:
        return expand_dims(output, dim)
        
    
def pairwise_add(u, v=None, is_batch=False):
    """
    performs a pairwise summation between vectors (possibly the same)
    can also be performed on batch of vectors.
    Parameters:
    ----------
    u, v: Tensor (m,) or (b,m)

    Returns: 
    ---------
    Tensor (m, n) or (b, m, n)
    
    """
    u_shape = u.size()
    if v is None:
        v = u
    v_shape = v.size()

    if len(u_shape) > 2 and not is_batch:
        raise ValueError("Expected at most 2D tensor or 3D tensor with batch")
    if len(v_shape) > 2 and not is_batch:
        raise ValueError("Expected at most 2D tensor or 3D tensor with batch")

    m = u_shape[0] if not is_batch else u_shape[1]
    n = v_shape[0] if not is_batch else v_shape[1]
    
    u = expand_dims(u, axis=-1)
    new_u_shape = list(u.size())
    new_u_shape[-1] = n
    U_ = u.expand(*new_u_shape)

    v = expand_dims(v, axis=-2)
    new_v_shape = list(v.size())
    new_v_shape[-2] = m
    V_ = v.expand(*new_v_shape)

    return U_ + V_

def to_device(src, ref):
    return src.cuda(ref.get_device()) if ref.is_cuda else src

def cumprod(inputs, dim = 1, exclusive=True):
    # https://github.com/tensorflow/tensorflow/blob/master/tensorflow/g3doc/api_docs/python/functions_and_classes/shard0/tf.cumprod.md

    if type(inputs) is not Variable:
        temp = torch.cumprod(inputs, dim)
        if not exclusive:
            return temp
        else:
            temp =  temp / (inputs[0].expand_as(temp) + 1e-8)
            temp[-1] = temp[-1]/(inputs[-1]+1e-8)
            return temp
    else:
        shape_ = inputs.size()
        ndim = len(shape_)
        n_slot = shape_[dim]
        slice_ = [slice(0,None,1) for _ in range(ndim)]
        results = [[]] * n_slot
            
        for ind in range(0, n_slot):   
            this_slice, last_slice = copy(slice_), copy(slice_)
            this_slice[dim] = ind
            last_slice[dim] = ind-1      
            this_slice = tuple(this_slice)
            last_slice = tuple(last_slice)
            if exclusive: 
                if ind > 0:   
                    results[ind]  = results[ind-1]*inputs[last_slice]
                else:
                    results[ind] =  torch.div(inputs[this_slice], inputs[this_slice]+1e-8)
            else:    
                if ind > 0:   
                    results[ind]  = results[ind - 1]*inputs[this_slice]
                else:
                    results[ind] =  inputs[this_slice]
        
        return torch.stack(results, dim)

            
def expand_dims(input, axis=0):
    input_shape = list(input.size())
    if axis < 0:
        axis = len(input_shape) + axis + 1
    input_shape.insert(axis, 1)
    return input.view(*input_shape)


def matmal(left, right):
    '''
    left is of size (*N, n1,n2), where N is a list
    right is of size(*M, m1,m2), where M is a list
    output is of size
    '''
    pass

def cosine_distance(memory_matrix, cos_keys):
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
    memory_norm = torch.norm(memory_matrix, 2, 2, keepdim=True)
    keys_norm = torch.norm(cos_keys, 2, 1,keepdim=True)
    
    normalized_mem  = torch.div(memory_matrix, memory_norm.expand_as(memory_matrix) + 1e-8)
    normalized_keys = torch.div(cos_keys, keys_norm.expand_as(cos_keys) + 1e-8)
    
    out =  torch.bmm(normalized_mem, normalized_keys)
    
    #print(normalized_keys)
    #print(out)
    #apply_dict(locals())
    
    return out
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
    #apply_dict(locals())
    return soft_max_nd.transpose(axis, len(input_size)-1)
