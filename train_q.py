#! /usr/bin/env python
"""
Execute a training run of deep-Q-Leaning.

"""

import launcher
import sys

class Defaults:
    # ----------------------
    # Experiment Parameters
    # ----------------------
    STEPS_PER_EPOCH = 250000 # 250000
    EPOCHS = 200
    STEPS_PER_TEST = 125000 # 125000

    # ----------------------
    # ALE Parameters
    # ----------------------
    BASE_ROM_PATH = "../aleroms/"
    ROM = 'breakout.bin'
    FRAME_SKIP = 4

    # ----------------------
    # Agent/Network parameters:
    # ----------------------
    MODEL_INDEX = 1
    UPDATE_RULE = 'rmsprop'
    LEARNING_RATE = .00025
    DISCOUNT = .99
    RMS_DECAY = .95 # (Rho)
    RMS_EPSILON = .01
    CLIP_DELTA = 1.0
    EPSILON_START = 0.9
    EPSILON_MIN = 0.1 #or 0.01 for tuned ddqn
    EPSILON_DECAY = 1000000
    PHI_LENGTH = 4
    UPDATE_FREQUENCY = 4
    REPLAY_MEMORY_SIZE = 1000000
    BATCH_SIZE = 32
    NETWORK_TYPE = "nature_dnn"
    FREEZE_INTERVAL = 10000 #30000 for tuned ddqn
    REPLAY_START_SIZE = 50000 #50000
    RESIZE_METHOD = 'scale' #scale vs crop
    RESIZED_WIDTH = 84
    RESIZED_HEIGHT = 84
    OFFSET = 18
    DEATH_ENDS_EPISODE = True
    CAP_REWARD = True
    MAX_START_NULLOPS = 30
    DETERMINISTIC = True
    PRED_COST_WEIGHT = 1.0
    OPTIMAL_EPS = 0.05 #0.05 or 0.001 for tuned ddqn
    DOUBLE_Q = False
    DEEPMODEL = "DQN"

    # ---------------------
    # Bootstrap params:
    # ---------------------
    HEADS_NUM = 10
    P = 1


    # ---------------------
    # HeapSum params:
    # ---------------------

    #TRANSFER LEARNING PARAMS
    TRANSFER = False
    TRANSFER_TESTING_GAMES = ['pong.bin','ice_hockey.bin']
    TRANSFER_TRAINING_GAMES = ['bowling.bin','asterix.bin']


if __name__ == "__main__":
    launcher.launch(sys.argv[1:], Defaults, __doc__)
