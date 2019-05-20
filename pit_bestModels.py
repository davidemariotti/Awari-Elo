import Arena
from MCTS import MCTS
from awari.AwariGame import AwariGame, display
from awari.keras.NNet import NNetWrapper as NNet
from awari.keras.NNet2Block import NNetWrapper as NNet2Block
from awari.keras.NNet10Block import NNetWrapper as NNet10Block
from awari.AwariPlayers import *
# to ask the oracle:
from awari.AwariLogic import Board
import sys
# from subprocess import check_output

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
parser.add_argument("-n", "--number", type=int, default=20)
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

class AwariNeuralNetPlayer():
    def __init__(self, game):
        self.game = game
        if file1 =="test_3_best_20iter.pth.tar" or file1 == "test_8_best_20iter.pth.tar" or file1 == "test_9_best_20iter.pth.tar" or file1 == "test_10_best_20iter.pth.tar":
            self.n1 = NNet(game)
            self.n1.load_checkpoint('', file1)
        elif file1 =="test_4_best_20iter.pth.tar" or file1 == "test_6_best_20iter.pth.tar":
            self.n1 = NNet2Block(game)
            self.n1.load_checkpoint('', file1)
        elif file1 =="test_5_best_20iter.pth.tar" or file1 == "test_7_best_20iter.pth.tar":
            self.n1 = NNet10Block(game)
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
    else:
        print("Unkown player: " + player)
        return None

#opponent = getPlayer(args.opponent)
#if opponent == None:
   # sys.exit(1)


#player = getPlayer(args.player)
#if player == None:
    #sys.exit(1)



#arena = Arena.Arena(player, opponent, g, display=display)
#print("win/lost/draw", arena.playGames(args.number, verbose=True))


for i in range(3, 16):
    for j in range(i+1, 16): #i+1
        file1 = 'test_'+str(i)+'_best_20iter.pth.tar'
        player = getPlayer('nn')
        file1 = 'test_'+str(j)+'_best_20iter.pth.tar'
        opponent = getPlayer('nn')
        print('awari|test_'+str(i)+'_best_20iter.pth.tar|test_'+str(j)+'_best_20iter.pth.tar|')
        arena = Arena.Arena(player, opponent, g, display=display)
        
        print("awariscore", arena.playGames(args.number, verbose=False))
        

