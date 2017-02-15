
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import getopt
import sys
import os
import math
import time
import argparse

import torch
from torch.autograd import Variable
import torch.nn.functional  as F
import torch.optim as optim

from neucom.neucom import NeuCom
from recurrent_controller import RecurrentController

parser = argparse.ArgumentParser(description='PyTorch Differentiable Neural Computer')
parser.add_argument('--input_size', type=int, default= 10,
                    help='dimension of input feature')

parser.add_argument('--nhid', type=int, default=64,
                    help='humber of hidden units of the inner nn')
                    
parser.add_argument('--nn_output', type=int, default=64,
                    help='humber of output units of the inner nn')

parser.add_argument('--nlayer', type=int, default=2,
                    help='number of layers')
parser.add_argument('--lr', type=float, default= 1e-4,
                    help='initial learning rate')
parser.add_argument('--clip', type=float, default=0.5,
                    help='gradient clipping')
parser.add_argument('--epochs', type=int, default=6,
                    help='upper epoch limit')
parser.add_argument('--batch-size', type=int, default= 32, metavar='N',
                    help='batch size')
parser.add_argument('--mem_size', type=int, default=10,
                    help='memory dimension')
parser.add_argument('--mem_slot', type=int, default=15,
                    help='number of memory slots')
parser.add_argument('--read_heads', type=int, default=2,
                    help='number of read heads')

parser.add_argument('--cuda', action='store_false',
                    help='use CUDA')
parser.add_argument('--log-interval', type=int, default=200, metavar='N',
                    help='report interval')
parser.add_argument('--save', type=str,  default='model.pt',
                    help='path to save the final model')
args = parser.parse_args()


def llprint(message):
    sys.stdout.write(message)
    sys.stdout.flush()

def generate_data(batch_size, length, size):

    input_data = np.zeros((batch_size, 2 * length + 1, size), dtype=np.float32)
    target_output = np.zeros((batch_size, 2 * length + 1, size), dtype=np.float32)

    sequence = np.random.binomial(1, 0.5, (batch_size, length, size - 1))

    input_data[:, :length, :size - 1] = sequence
    input_data[:, length, -1] = 1  # the end symbol
    target_output[:, length + 1:, :size - 1] = sequence

    input_data = torch.from_numpy(input_data)
    target_output = torch.from_numpy(target_output)
    if args.cuda:
        input_data = input_data.cuda()
        target_output = target_output.cuda()

    return Variable(input_data), Variable(target_output)


def criterion(predictions, targets):
    return F.cross_entropy(predictions, targets)

def clip_gradient(model, clip):
    """Computes a gradient clipping coefficient based on gradient norm."""
    totalnorm = 0
    for p in model.parameters():
        modulenorm = p.grad.data.norm()
        totalnorm += modulenorm ** 2
    totalnorm = math.sqrt(totalnorm)
    return min(1, args.clip / (totalnorm + 1e-6))

if __name__ == '__main__':

    dirname = os.path.dirname(__file__)
    ckpts_dir = os.path.join(dirname , 'checkpoints')
    tb_logs_dir = os.path.join(dirname, 'logs')

    batch_size = 2
    sequence_max_length = 20

    input_size = output_size = args.input_size
    mem_slot = args.mem_slot
    mem_size = args.mem_size
    read_heads = args.read_heads
    
    from_checkpoint = None
    iterations = 100000

    options,_ = getopt.getopt(sys.argv[1:], '', ['checkpoint=', 'iterations='])

    for opt in options:
        if opt[0] == '--checkpoint':
            from_checkpoint = opt[1]
        elif opt[0] == '--iterations':
            iterations = int(opt[1])

    ncomputer = NeuCom(
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


    if args.cuda:
        ncomputer = ncomputer.cuda()
    
    last_100_losses = []
    optimizer = optim.Adam(ncomputer.parameters(), lr=args.lr)


    for epoch in range(iterations + 1):
        llprint("\rIteration {ep}/{tot}".format(ep=epoch, tot=iterations))
        optimizer.zero_grad()

        random_length = np.random.randint(1, sequence_max_length + 1)

        input_data, target_output = generate_data(batch_size, random_length, input_size)

        output, _ = ncomputer.forward(input_data)
        loss = criterion(F.sigmoid(output), target_output)
        loss.backward()
        optimizer.step()
        loss_value = loss.data[0]

        summerize = (epoch % 100 == 0)
        take_checkpoint = (epoch != 0) and (epoch % iterations == 0)

        last_100_losses.append(loss_value)

        if summerize:
            llprint("\n\tAvg. Logistic Loss: %.4f\n" % (np.mean(last_100_losses)))
            last_100_losses = []
        
        if take_checkpoint:
            llprint("\nSaving Checkpoint ... "),
            check_ptr = os.path.join(ckpts_dir, 'step_{}'.format(epoch))
            with open(check_ptr, 'wb') as f:
                torch.save(model, f)
            llprint("Done!\n")
