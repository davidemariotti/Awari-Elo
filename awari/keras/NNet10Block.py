import argparse
import os
import shutil
import time
import random
import numpy as np
import math
import sys
sys.path.append('..')
from utils import *
from NeuralNet import NeuralNet

from keras.callbacks import TensorBoard

import argparse
from .AwariNNet import AwariNNet as onnet

"""
NeuralNet wrapper class for the AwariNNet.

Based on the NNet by SourKream and Surag Nair.
"""

args = dotdict({
    
    'lr': 0.01,
    'dropout': 0.3,# not used by residual model
    # 'epochs': 15,
    'epochs': 10,
    'batch_size': 32,
    'cuda': False,
    'num_channels': 512,# not used by residual model
    
    # 'cnn_filter_num': 256, 
    'cnn_filter_num': 64, 
    # 'cnn_first_filter_size': 5,
    'cnn_first_filter_size': 3,
    'cnn_filter_size': 3,
    'residual_block_num': 10,
    # 'residual_block_num': 2,
    'l2_reg': 1e-4,
    'value_fc_size': 256,
    'trainer_loss_weights': [1.0, 1.0], # not used // [policy, value] prevent value overfit in SL
    
    'image_stack_layers': 9,
})

def new_run_log_dir(base_dir):
    log_dir = os.path.join('./log', base_dir)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    return log_dir

class NNetWrapper(NeuralNet):
    def __init__(self, game):
        self.game = game
        self.nnet = onnet(game, args)
        # self.board_x, self.board_y = game.getBoardSize()
        self.board_x = game.getBoardSize(args.image_stack_layers)
        self.action_size = game.getActionSize()
        # tensorboard logging:
        self.log_dir = new_run_log_dir('keras-tensorboard')

    def train(self, examples):
        """
        examples: list of examples, each example is of form (board, pi, v)
        """
        input_boards, target_pis, target_vs = list(zip(*examples))
        # input_boards = np.asarray(input_boards)
        stacks = []
        for b in input_boards:
            stacks.append(self.game.getImageStack(b, args.image_stack_layers))
        input_boards = np.asarray(stacks)
        target_pis = np.asarray(target_pis)
        target_vs = np.asarray(target_vs)
        self.nnet.model.fit(x = input_boards, y = [target_pis, target_vs], batch_size = args.batch_size, epochs = args.epochs)
       

    def predict(self, board):
        """
        board: np array with board
        """
        
        board = self.game.getImageStack(board, args.image_stack_layers)
        board = board[np.newaxis, :, :, :]

        
        pi, v = self.nnet.model.predict(board)
        return pi[0], v[0]

    def save_checkpoint(self, folder='checkpoint', filename='checkpoint.pth.tar'):
        filepath = os.path.join(folder, filename)
        if not os.path.exists(folder):
            print("Checkpoint Directory does not exist! Making directory {}".format(folder))
            os.mkdir(folder)
        else:
            print("Checkpoint Directory exists! ")
        self.nnet.model.save_weights(filepath)

    def load_checkpoint(self, folder='checkpoint', filename='checkpoint.pth.tar'):
        # https://github.com/pytorch/examples/blob/master/imagenet/main.py#L98
        filepath = os.path.join(folder, filename)
        if not os.path.exists(filepath):
            raise("No model in path '{}'".format(filepath))
        self.nnet.model.load_weights(filepath)
