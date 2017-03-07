import warnings
warnings.filterwarnings('ignore')

import numpy as np
import getopt
import sys
import os
import math
import time
import argparse

sys.path.insert(0, os.path.join('..','..'))

import torch
from torch.autograd import Variable
import torch.nn.functional  as F
import torch.optim as optim

from neucom.utils import inves, apply_dict
from neucom.dnc import DNC
from recurrent_controller import RecurrentController

parser = argparse.ArgumentParser(description='PyTorch Differentiable Neural Computer')
parser.add_argument('--input_size', type=int, default= 6,
                    help='dimension of input feature')

parser.add_argument('--nhid', type=int, default= 64,
                    help='humber of hidden units of the inner nn')
                    
parser.add_argument('--nn_output', type=int, default= 64,
                    help='humber of output units of the inner nn')

parser.add_argument('--nlayer', type=int, default=2,
                    help='number of layers')
parser.add_argument('--lr', type=float, default= 1e-2,
                    help='initial learning rate')
parser.add_argument('--clip', type=float, default=0.5,
                    help='gradient clipping')

parser.add_argument('--batch_size', type=int, default= 2, metavar='N',
                    help='batch size')
parser.add_argument('--mem_size', type=int, default= 16,
                    help='memory dimension')
parser.add_argument('--mem_slot', type=int, default= 15,
                    help='number of memory slots')
parser.add_argument('--read_heads', type=int, default=1,
                    help='number of read heads')

parser.add_argument('--sequence_max_length', type=int, default= 4, metavar='N',
                    help='sequence_max_length')
parser.add_argument('--cuda', action='store_true', default= True,
                    help='use CUDA')
parser.add_argument('--log-interval', type=int, default=200, metavar='N',
                    help='report interval')

parser.add_argument('--iterations', type=int, default= 100000, metavar='N',
                    help='total number of iteration')
parser.add_argument('--summerize_freq', type=int, default= 100, metavar='N',
                    help='summerise frequency')
parser.add_argument('--check_freq', type=int, default= 100, metavar='N',
                    help='check point frequency')

args = parser.parse_args()

args.cuda = args.cuda and torch.cuda.is_available()
if args.cuda:
    print('Using CUDA.')
else:
    print('Using CPU.')
    


def llprint(message):
    sys.stdout.write(message)
    sys.stdout.flush()

def generate_data(batch_size, length, size, cuda=False):

    input_data = np.zeros((batch_size, 2 * length + 1, size), dtype=np.float32)
    target_output = np.zeros((batch_size, 2 * length + 1, size), dtype=np.float32)

    sequence = np.random.binomial(1, 0.5, (batch_size, length, size - 1))

    input_data[:, :length, :size - 1] = sequence
    input_data[:, length, -1] = 1  # the end symbol
    target_output[:, length + 1:, :size - 1] = sequence

    input_data = torch.from_numpy(input_data)
    target_output = torch.from_numpy(target_output)
    if cuda:
        input_data = input_data.cuda()
        target_output = target_output.cuda()

    return Variable(input_data), Variable(target_output)

def criterion(predictions, targets):
    return torch.mean(
        -1 * F.logsigmoid(predictions) * (targets) - torch.log(1 - F.sigmoid(predictions) + 1e-9) * (1 - targets)
    )
    
def clip_gradient(model, clip):
    """Computes a gradient clipping coefficient based on gradient norm."""
    totalnorm = 0
    for p in model.parameters():
        modulenorm = p.grad.data.norm()
        totalnorm += modulenorm ** 2
    totalnorm = math.sqrt(totalnorm)
    return min(args.clip, args.clip / (totalnorm + 1e-6))

def register_nan_checks(model):
    def check_grad(module, grad_input, grad_output):
        # print(module) you can add this to see that the hook is called
        print(module)
        if  any(np.all(np.isnan(gi.data.numpy())) for gi in grad_input if gi is not None):
            print('NaN gradient in ' + type(module).__name__)
        
    model.apply(lambda module: module.register_backward_hook(check_grad))

if __name__ == '__main__':

    dirname = os.path.dirname(__file__)
    ckpts_dir = os.path.join(dirname , 'checkpoints')
    if not os.path.isdir(ckpts_dir):
        os.mkdir(ckpts_dir)

    batch_size = args.batch_size
    sequence_max_length = args.sequence_max_length
    iterations = args.iterations
    summerize_freq  = args.summerize_freq
    check_freq = args.check_freq

    input_size = output_size = args.input_size
    mem_slot = args.mem_slot
    mem_size = args.mem_size
    read_heads = args.read_heads
    
    from_checkpoint = None
    
    options,_ = getopt.getopt(sys.argv[1:], '', ['checkpoint=', 'iterations='])

    for opt in options:
        if opt[0] == '--checkpoint':
            from_checkpoint = opt[1]
        elif opt[0] == '--iterations':
            iterations = int(opt[1])

    ncomputer = DNC(
                args.nhid,
                args.nn_output,
                args.nlayer,
                RecurrentController,
                input_size,
                output_size,
                mem_slot,
                mem_size,
                read_heads,
                batch_size,
                use_cuda = args.cuda
               )

    
    register_nan_checks(ncomputer)   

    if args.cuda:
        ncomputer = ncomputer.cuda()
        
    if from_checkpoint is not None:
        ncomputer.load_state_dict(torch.load(from_checkpoint) )# 12)

    last_save_losses = []

    #optimizer = optim.RMSprop(ncomputer.parameters(), lr=args.lr)
    optimizer = optim.Adam(ncomputer.parameters(), lr=args.lr)
    #optimizer = optim.SGD(ncomputer.parameters(), lr=args.lr, momentum = 0.9)
    
    for epoch in range(iterations + 1):
        llprint("\rIteration {ep}/{tot}".format(ep=epoch, tot=iterations))
        optimizer.zero_grad()

        random_length = np.random.randint(1, sequence_max_length + 1)
        
        input_data, target_output = generate_data(batch_size, random_length, input_size, args.cuda)
        input_data = input_data.transpose(0,1).contiguous()
        target_output = target_output.transpose(0,1).contiguous()

        output, _ = ncomputer.forward(input_data)

        loss = criterion((output), target_output)
        #if np.isnan(loss.data.cpu().numpy()):
        #    llprint('\nGot nan loss, contine to jump the backward \n')
            
        #apply_dict(locals())
        loss.backward()
        
        optimizer.step()
        loss_value = loss.data[0]
        
        summerize = (epoch % summerize_freq == 0)
        take_checkpoint = (epoch != 0) and (epoch % check_freq == 0)

        last_save_losses.append(loss_value)

        if summerize:
            llprint("\n\tAvg. Logistic Loss: %.4f\n" % (np.mean(last_save_losses)))
            last_save_losses = []
        
        if take_checkpoint:
            llprint("\nSaving Checkpoint ... "),
            check_ptr = os.path.join(ckpts_dir, 'step_{}.pth'.format(epoch))
            cur_weights = ncomputer.state_dict()
            torch.save(cur_weights, check_ptr)
            llprint("Done!\n")
