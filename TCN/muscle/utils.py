from scipy.io import loadmat
import torch
import numpy as np


def data_generator(dataset='./data/datalabel.mat'):
    print('data...')
    data = loadmat(dataset)

    concat_data = np.concatenate(data['datalabel'][0,0], data['datalabel'][0,1])

    training_data = concat_data[:600000, :]
    validation_data = concat_data[600000:700000, :]
    test_data = concat_data[700000:, :]

    training_X = torch.Tensor(training_data[:, :-1])
    training_Y = torch.LongTensor(training_data[:, -1])
    validation_X = torch.Tensor(validation_data[:, :-1])
    validation_Y = torch.LongTensor(validation_data[:, -1])
    test_X = torch.Tensor(test_data[:, :-1])
    test_Y = torch.LongTensor(test_data[:, -1])

    return training_X, training_Y, validation_X, validation_Y, test_X, test_Y



