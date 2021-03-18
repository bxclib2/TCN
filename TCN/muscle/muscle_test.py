import argparse
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
import sys



sys.path.append("../../")
from TCN.muscle.utils import data_generator
from TCN.muscle.model import TCN
import numpy as np


parser = argparse.ArgumentParser(description="Jirou Feng's data")
parser.add_argument('--cuda', action='store_false',
                    help='use CUDA (default: True)')
parser.add_argument('--dropout', type=float, default=0.25,
                    help='dropout applied to layers (default: 0.25)')
parser.add_argument('--clip', type=float, default=0.2,
                    help='gradient clip, -1 means no clip (default: 0.2)')
parser.add_argument('--epochs', type=int, default=100,
                    help='upper epoch limit (default: 100)')
parser.add_argument('--ksize', type=int, default=5,
                    help='kernel size (default: 5)')
parser.add_argument('--levels', type=int, default=4,
                    help='# of levels (default: 4)')
parser.add_argument('--log-interval', type=int, default=100, metavar='N',
                    help='report interval (default: 100')
parser.add_argument('--lr', type=float, default=1e-3,
                    help='initial learning rate (default: 1e-3)')
parser.add_argument('--optim', type=str, default='Adam',
                    help='optimizer to use (default: Adam)')
parser.add_argument('--nhid', type=int, default=150,
                    help='number of hidden units per layer (default: 150)')
parser.add_argument('--input_size', type=int, default=60,
                    help='input size (default: 60)')
parser.add_argument('--seed', type=int, default=1111,
                    help='random seed (default: 1111)')
parser.add_argument('--window_size', type=int, default=60,
                    help='window size')
parser.add_argument('--batch_size', type=int, default=32,
                    help='batch size')

args = parser.parse_args()


# Set the random seed manually for reproducibility.
torch.manual_seed(args.seed)
if torch.cuda.is_available():
    if not args.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")

print(args)

if 600000 % args.window_size != 0:
    raise('window size must be a divsior of 600000')

training_X, training_Y, validation_X, validation_Y, test_X, test_Y = data_generator()

input_size = args.input_size
n_channels = [args.nhid] * args.levels
kernel_size = args.ksize
dropout = args.dropout

model = TCN(input_size, input_size, n_channels, kernel_size, dropout=args.dropout)

if args.cuda:
    model.cuda()

criterion = nn.NLLLoss()
lr = args.lr
optimizer = getattr(optim, args.optim)(model.parameters(), lr=lr)


def train(ep, window_size, batch_size):
    model.train()
    total_loss = 0
    count = 0

    train_window_start = [idx for idx in range(0, training_X.shape[0], window_size)]
    np.random.shuffle(train_window_start)
    for s_idx in range(0, len(train_window_start), batch_size):
        e_idx = min(s_idx + batch_size, len(train_window_start))
        train_data_list = []
        train_label_list = []
        for t_idx in range(s_idx, e_idx):
            train_data_list.append(training_X[t_idx:t_idx + window_size, :])
            train_label_list.append(training_Y[t_idx:t_idx + window_size])

        x_batch = torch.stack(train_data_list).permute(0,2,1)

        y_batch = torch.stack(train_label_list)

        if args.cuda:
            x_batch, y_batch = x_batch.cuda(), y_batch.cuda()

        optimizer.zero_grad()
        output = model(x_batch)
        loss = criterion(output, y_batch)
        total_loss += loss.item()
        count += (e_idx - s_idx)

        if args.clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
        loss.backward()
        optimizer.step()
        if s_idx > 0 and s_idx % args.log_interval == 0:
            cur_loss = total_loss / count
            print("Epoch {:2d} | lr {:.5f} | loss {:.5f}".format(ep, lr, cur_loss))
            total_loss = 0.0
            count = 0


def evaluate(X, Y, window_size, name='Eval'):
    model.eval()
    eval_window_start = [idx for idx in range(0, X.shape[0], window_size)]
    total_loss = 0.0
    count = 0
    with torch.no_grad():
        for t_idx in eval_window_start:
            x, y = X[t_idx:t_idx+window_size, :].unsqueeze(0).permute(0,2,1), Y[t_idx:t_idx+window_size].unsqueeze(0)
            if args.cuda:
                x, y = x.cuda(), y.cuda()
            output = model(x)
            loss = criterion(output, y)
            total_loss += loss.item()
            count += 1
        eval_loss = total_loss / count
        print(name + " loss: {:.5f}".format(eval_loss))
        return eval_loss


if __name__ == "__main__":
    best_vloss = 1e8
    vloss_list = []
    model_name = "muscle_model.pt"
    for ep in range(1, args.epochs+1):
        train(ep, args.window_size, args.batch_size)
        vloss = evaluate(validation_X, validation_Y, args.window_size, name='Validation')
        tloss = evaluate(test_X, test_Y, args.window_size, name='Test')
        if vloss < best_vloss:
            with open(model_name, "wb") as f:
                torch.save(model, f)
                print("Saved model!\n")
            best_vloss = vloss
        if ep > 10 and vloss > max(vloss_list[-3:]):
            lr /= 10
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr

        vloss_list.append(vloss)

    print('-' * 89)
    model = torch.load(open(model_name, "rb"))
    tloss = evaluate(test_X, test_Y, args.window_size)