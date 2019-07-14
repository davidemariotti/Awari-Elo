import Arena
from MCTS import MCTS
from awari.AwariGame import AwariGame, display
from awari.keras.NNet import NNetWrapper as NNet
from awari.keras.NNet2Block import NNetWrapper as NNet2Block
from awari.keras.NNet10Block import NNetWrapper as NNet10Block
from awari.keras.NNet1Layer import NNetWrapper as NNet1Layer
from awari.keras.NNet13Layer import NNetWrapper as NNet13Layer
from awari.keras.NNet15Layer import NNetWrapper as NNet15Layer
from awari.AwariPlayers import *
# to ask the oracle:
from awari.AwariLogic import Board
import sys
import itertools

import numpy as np
from utils import *

"""
use this script to play any two agents against each other, or play manually with
any agent.
"""

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("-d", "--dir",  type=str, default="./temp/awari-keras/")
#parser.add_argument("-f", "--file", type=str, default="best.pth.tar")
parser.add_argument("-n", "--number", type=int, default=2)
parser.add_argument("-m", "--mcts", type=int, default=50)
parser.add_argument("-c", "--cpuct", type=float, default=1.0)
#parser.add_argument("-o", "--opponent", type=str, default="fop8")
#parser.add_argument("-p", "--player", type=str, default="nn")
args = parser.parse_args()


print("directory: %s" % args.dir)
#print("file: %s" % args.file)
print("number: %d" % args.number)
print("mcts: %d" % args.mcts)
print("cpuct: %f" % args.cpuct)
#print("player: %s" % args.player)
#print("opponent: %s" % args.opponent)

g = AwariGame()
# General pit
# test3 = 'test_3_best_20iter.pth.tar'
# test4 = 'test_4_best_20iter.pth.tar'
# test5 = 'test_5_best_20iter.pth.tar'
# test6 = 'test_6_best_20iter.pth.tar'
# test7 = 'test_7_best_20iter.pth.tar'
# test8 = 'test_8_best_20iter.pth.tar'
# test9 = 'test_9_best_20iter.pth.tar'
# test10 = 'test_10_best_20iter.pth.tar'
# test11 =  'test_11_best_20iter.pth.tar'
# test12 = 'test_12_best_20iter.pth.tar'
# test13 = 'test_13_best_20iter.pth.tar'
# test14 = 'test_14_best_20iter.pth.tar'
# test15 = 'test_15_best_20iter.pth.tar'
# fop3 = 'fop3'
# fop4 = 'fop4'
# fop5 = 'fop5'
# tests = [test3, test4, test5, test6, test7, test8, test9, test10, test11, test12, test13, test14, test15, fop3, fop4, fop5]

#pit 100 ietrations
#test3 = 'test_3_best_100iter.pth.tar'
#test4 = 'test_4_best_100iter.pth.tar'
#test12 = 'test_12_best_100iter.pth.tar'
#test13 = 'test_13_best_100iter.pth.tar'
#test14 = 'test_14_best_100iter.pth.tar'
#fop3 = 'fop3'
#fop4 = 'fop4'
#fop5 = 'fop5'
#tests = [test3, test4, test12, test13, test14, fop3, fop4, fop5]

#pit 500 ietrations
test3 = 'test_3_best_500iter.pth.tar'
test4 = 'test_4_best_500iter.pth.tar'
test13 = 'test_13_best_500iter.pth.tar'
test14 = 'test_14_best_500iter.pth.tar'
fop3 = 'fop3'
fop4 = 'fop4'
fop5 = 'fop5'
tests = [test3, test4, test13, test14, fop3, fop4, fop5]


class AwariNeuralNetPlayer():
    def __init__(self, game):
        self.game = game
        if file1 =="test_3_best_20iter.pth.tar" or file1 == "test_8_best_20iter.pth.tar" or file1 == "test_9_best_20iter.pth.tar" or file1 == "test_10_best_20iter.pth.tar" or file1 =="test_11_best_20iter.pth.tar" or file1 =="test_12_best_20iter.pth.tar" or file1 == 'test_3_best_100iter.pth.tar' or file1 =="test_12_best_100iter.pth.tar":
            self.n1 = NNet(game)
            self.n1.load_checkpoint('', file1)
        elif file1 =="test_4_best_20iter.pth.tar" or file1 == "test_6_best_20iter.pth.tar" or file1 =="test_4_best_100iter.pth.tar":
            self.n1 = NNet2Block(game)
            self.n1.load_checkpoint('', file1)
        elif file1 =="test_5_best_20iter.pth.tar" or file1 == "test_7_best_20iter.pth.tar":
            self.n1 = NNet10Block(game)
            self.n1.load_checkpoint('', file1)
        elif file1 =="test_13_best_20iter.pth.tar" or file1 =="test_13_best_100iter.pth.tar":
            self.n1 = NNet1Layer(game)
            self.n1.load_checkpoint('', file1)
        elif file1 =="test_14_best_20iter.pth.tar" or file1 =="test_14_best_100iter.pth.tar":
            self.n1 = NNet13Layer(game)
            self.n1.load_checkpoint('', file1)
        elif file1 =="test_15_best_20iter.pth.tar":
            self.n1 = NNet15Layer(game)
            self.n1.load_checkpoint('', file1)
        else:
            print("Unkown player: " + player)
            return None


    def play(self, board):
        args1 = dotdict({'numMCTSSims': args.mcts, 'cpuct': args.cpuct})
        mcts1 = MCTS(self.game, self.n1, args1)
        actions = mcts1.getActionProb(board, temp=1)
        select = np.argmax(actions)
        #print('board: ', end="")
        #print(board)
        #print('action p-values: ' + str(actions))
        
        b = Board(6)
        b.pieces = np.copy(board)
        b.check_board(select, prefix = "nn: ")

        return select

# all players
def getPlayer(player):
    if player == "random":
        return RandomAwariPlayer(g).play
    elif player == "nn":
        return AwariNeuralNetPlayer(g).play
    elif player == "fop3":
        return OracleAwariPlayer(g, 0.30, 15).play
    elif player == "fop4":
        return OracleAwariPlayer(g, 0.40, 15).play
    elif player == "fop5":
        return OracleAwariPlayer(g, 0.50, 15).play
    else:
        print("Unkown player: " + player)
        return None


for a, b in itertools.combinations(tests, 2):
    #print(a,b)
    file1 = a
    if 'test_' in file1: 
        player = getPlayer('nn')
    elif 'fop' in file1: 
        player = getPlayer(file1)
    file1 = b
    if 'test_' in file1: 
        opponent = getPlayer('nn')
    elif 'fop' in file1: 
        opponent = getPlayer(file1)
    print('awari|'+a+'|'+b+'|')
    arena = Arena.Arena(player, opponent, g, display=display)
    print("awariscore", arena.playGames(args.number, verbose=False))

