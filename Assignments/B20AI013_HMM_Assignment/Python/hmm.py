from enum import IntEnum 

class SuspectState(IntEnum):
    Planning = 1,
    Scouting = 2,
    Burglary = 3,
    Migrating = 4,
    Misc = 5

class Daytime(IntEnum):
    Day = 6
    Evening = 7
    Night = 8

class Action(IntEnum):
    Roaming = 9,
    Eating = 10,
    Home = 11,
    Untracked = 12,

class Observation:
    def __init__(self, d:Daytime, a:Action) -> None:
        self.daytime = d
        self.action = a


# This function reads the string file 
# that contains the sequence of observations
# that is required to learn the HMM model.
def ReadDataset() -> list:
    # Converts integer to Daytime enum
    def getDay(p: int) -> Daytime:
        if p == 6:
            return Daytime.Day
        elif p == 7:
            return Daytime.Evening
        elif p == 8:
            return Daytime.Night
        else:
            assert False, 'Unexpected Daytime!'

    # Converts integer to Action enum
    def getAct(p: int) -> Action:
        if p == 9:
            return Action.Roaming
        elif p == 10:
            return Action.Eating
        elif p == 11:
            return Action.Home
        elif p == 12:
            return Action.Untracked
        else:
            assert False, 'Unexpected Action!'
    filepath = 'database.txt' 
    with open(filepath, 'r') as file:
        seq_count = int(file.readline())
        seq_list = []
        for _ in range(seq_count):
            w = file.readline().split(' ')
            len = int(w[0])
            seq_i = []
            for k in range(0, len):
                idx = (k*2) + 1
                day = int(w[idx])
                act = int(w[idx + 1])
                o = Observation(getDay(day), getAct(act))
                seq_i.append(o)
            seq_list.append(seq_i)
    return seq_list


#  --------------Do not change anything above this line---------------
import os
# os.system('pip install numpy')
# os.system('pip install tqdm')
import numpy as np
from collections import defaultdict
from tqdm import tqdm
class HMM:
    # Complete the HMM implementation.
    # The three function below must
    # be implemented.

    def __init__(self):
        N = len(SuspectState)
        self._N =  N
        self._pi = np.zeros([N, 1])
        self._pi[0][0] = 0.5
        self._pi[1][0] = 0.0
        self._pi[2][0] = 0.0
        self._pi[3][0] = 0.0
        self._pi[4][0] = 0.5


        # self._transitions = np.ones([N, N]) / N
        # self._emissions = np.ones([5, 3, 4]) / 3 * 4

        self._transitions = np.array(
            [
                [0.6, 0.4, 0.0, 0.0, 0.0], # SuspectState.Planning
                [0.0, 0.6, 0.2, 0.0, 0.2], # SuspectState.Scouting
                [0.0, 0.0, 0.0, 1.0,  0.0], # SuspectState.Burglary
                [0.2, 0.0, 0.0, 0.3, 0.5], # SuspectState.Migrating
                [0.6, 0.2, 0.0, 0.0,  0.2], # SuspectState.Misc
            ]
        )

        [0.4, 0.2, 0.0, 0.0, 0.0, 0.2, 0.2], # SuspectState.Planning
        [0.0, 0.2, 0.7/3, 0.7/3, 0.7/3, 0.1, 0.0], # SuspectState.Scouting
        [0.0, 0.0, 0.0, 1.0,  0.0, 0.0, 0.0], # SuspectState.Burglary
        [0.3, 0.0, 0.0, 0.1, 0.4, 0.1, 0.1], # SuspectState.Migrating
        [0.2, 0.0, 0.0, 0.0,  0.2, 0.6, 0.0], # SuspectState.Misc
        [0.5, 0.0, 0.0, 0.5/2, 0.5/2, 0.0, 0.0], # SuspectState.Sleeping
        [0.0, 0.0, 0.8, 0.0, 0.0, 0.0, 0.2], # SuspectState.Misc2
        self._emissions = np.array(
            [
                [ # 1 SuspectState.Planning
                    # Action.Roaming, Action.Eating, Action.Home, Action.Untracked
                    # 9 10 11 12
                    [0.1/3, 0.1/3, 0.7/3, 0.1/3], # 6 Daytime.Day
                    [0.1/3, 0.1/3, 0.7/3, 0.1/3], # 7 Daytime.Evening
                    [0.1/3, 0.1/3, 0.7/3, 0.1/3], # 8 Daytime.Night
                ],

                [ # 2 SuspectState.Scouting
                    # Action.Roaming, Action.Eating, Action.Home, Action.Untracked
                    # 9 10 11 12
                    [0.05, 0.1/3, 0.0, 0.1/3], # 6 Daytime.Day
                    [0.05, 0.1/3, 0.0, 0.1/3], # 7 Daytime.Evening
                    [0.7 , 0.1/3, 0.0, 0.1/3], # 8 Daytime.Night
                ],

                [ # 3 SuspectState.Burglary
                    # Action.Roaming, Action.Eating, Action.Home, Action.Untracked
                    # 9 10 11 12
                    [0.1/3, 0.1/3, 0.1/3, 0.7/3], # 6 Daytime.Day
                    [0.1/3, 0.1/3, 0.1/3, 0.7/3], # 7 Daytime.Evening
                    [0.1/3, 0.1/3, 0.1/3, 0.7/3], # 8 Daytime.Night
                ],

                [ # 4 SuspectState.Migrating
                    # Action.Roaming, Action.Eating, Action.Home, Action.Untracked
                    # 9 10 11 12
                    [0.1/3, 0.1/3, 0.0/3, 0.8/3], # 6 Daytime.Day
                    [0.1/3, 0.1/3, 0.0/3, 0.8/3], # 7 Daytime.Evening
                    [0.1/3, 0.1/3, 0.0/3, 0.8/3], # 8 Daytime.Night
                ],

                [ # 5 SuspectState.Misc
                    # Action.Roaming, Action.Eating, Action.Home, Action.Untracked
                    # 9 10 11 12
                    [0.1, 0.1, 0.1/3, 0.1/3], # 6 Daytime.Day
                    [0.1, 0.3 , 0.1/3, 0.1/3], # 7 Daytime.Evening
                    [0.1, 0.1, 0.1/3, 0.1/3], # 8 Daytime.Night
                ]
            ]
        )
        # # softmaxing the probabilities
        # self._emissions = np.exp(self._emissions) / np.sum(np.sum(np.exp(self._emissions), axis=1, keepdims=True), axis=2, keepdims=True)
        # self._transitions = np.exp(self._transitions) / np.sum(np.exp(self._transitions), axis=1, keepdims=True)
        




    def A(self, a: SuspectState, b: SuspectState) -> float:
        return self._transitions[a-1][b-1]
        # Compute the probablity of going from one
        # state a to the other state b
        # Hint: The necessary code does not need
        # to be within this function only
        # pass

    def B(self, a: SuspectState, b: Observation) -> float:
        return self._emissions[a-1][b.daytime-6][b.action-9]
        # Compute the probablity of obtaining
        # observation b from state a
        # Hint: The necessary code does not need
        # to be within this function only
        # pass

    def Pi(self, a: SuspectState) -> float:
        return self._pi[a-1][0]
        # Compute the initial probablity of
        # being at state a
        # Hint: The necessary code does not need
        # to be within this function only
        # pass

# Part I
# ---------

# Reads the dataset of array of sequence of observation
# and initializes a HMM model from it.
# returns Initialized HMM.
# Here, the parameter `dataset` is
# a list of list of `Observation` class.
# Each (inner) list represents the sequence of observation
# from start to end as mentioned in the question


def LearnModel(dataset: list) -> HMM:
    # initialize the model
    model = HMM()
    debug = False
    original_transitions = model._transitions
    original_emissions = model._emissions
    epsilon = 3e-1
    gap_norm = np.inf
    iteration = 0
    while gap_norm > epsilon:
        iteration += 1
        print(f'iteration#{iteration}: {gap_norm:.3f}')
        previous_emissions = model._emissions.copy()
        previous_transitions = model._transitions.copy()
        A = model._transitions
        B = model.B
        N = model._N
        Pi = model.Pi   
        obs_stats = {}
        ###################################################################### E -step
        for obs_index, obs_list in enumerate(dataset):
            T = len(obs_list)
            alpha = {}
            beta = {}
            eta = {}
            gamma = {}
            obs_stats[obs_index] = {
                'obs_list': obs_list,
                'alpha'   : alpha,
                'beta'    : beta,
                'eta'     : eta,
                'gamma'   : gamma,
                'T'       : T,
            }
            alpha[0] = np.array([[Pi(SuspectState(j+1)) * B(SuspectState(j+1), obs_list[0])] for j in range(N)])
            for t in range(1, T):
                # print('alpha', alpha[0].shape)
                alpha[t] = alpha[t-1] * np.array([[B(SuspectState(j+1), obs_list[t])] for j in range(N)])
                alpha[t] = A.T @ alpha[t]

            beta[T-1] = np.array([[1] for j in range(N)])
            # print(f'beta;{beta[T-1].shape}')
            for t in reversed(range(T-1)):
                beta[t] = A.T @ (beta[t+1] * np.array([[B(SuspectState(j+1), obs_list[t+1])] for j in range(N)]))
            # print(beta[0].shape)
            
            for t in range(T-1):
                eta[t] = {
                    'numerator'  : alpha[t] * A * np.array([B(SuspectState(j+1), obs_list[t+1]) for j in range(N)]),
                    'denominator': sum([alpha[t][j][0] * beta[t][j][0] for j in range(N)]),
                }
                # print(eta[t]['denominator'])
            for t in range(T):
                gamma[t] = {
                    'numerator'  : (alpha[t] * beta[t]).flatten(),
                    'denominator': np.sum([alpha[t][j] * beta[t][j] for j in range(N)]),
                }
                # print(gamma[t]['numerator'] / gamma[t]['denominator'])
        
        ###################################################################### M -step
        for i in range(N):
            for j in range(N):
                numerator = 0
                denominator = 0
                for obs_index in range(len(dataset)):
                    obs_list = obs_stats[obs_index]['obs_list']
                    alpha    = obs_stats[obs_index]['alpha']
                    beta     = obs_stats[obs_index]['beta']
                    gamma    = obs_stats[obs_index]['gamma']
                    eta      = obs_stats[obs_index]['eta']
                    T        = obs_stats[obs_index]['T']
                    numerator += sum([eta[t]['numerator'][i][j] / eta[t]['denominator'] for t in range(T-1)])
                    denominator += sum([sum([eta[t]['numerator'][i][k] / eta[t]['denominator'] for k in range(N)]) for t in range(T-1)])
                A[i][j] = numerator / denominator
                # A[i][j]  = sum([eta[t]['numerator'][i][j] / eta[t]['denominator'] for t in range(T-1)])
                # A[i][j] /= sum([sum([eta[t]['numerator'][i][k] / eta[t]['denominator'] for k in range(N)]) for t in range(T-1)])
        
        for j in range(N):
            for daytime in range(3): # + 6
                for action in range(4): # + 9
                    numerator = 0
                    denominator = 0
                    vk = Observation(Daytime(daytime+6), Action(action+9))
                    for obs_index in range(len(dataset)):
                        obs_list = obs_stats[obs_index]['obs_list']
                        alpha    = obs_stats[obs_index]['alpha']
                        beta     = obs_stats[obs_index]['beta']
                        gamma    = obs_stats[obs_index]['gamma']
                        eta      = obs_stats[obs_index]['eta']
                        T        = obs_stats[obs_index]['T']
                        # print(len([gamma[t]['numerator'][j]/gamma[t]['denominator'] for t in range(T) if obs_list[t] == vk]))
                        numerator += sum([gamma[t]['numerator'][j]/gamma[t]['denominator'] for t in range(T) if (obs_list[t].daytime == vk.daytime and obs_list[t].action == vk.action)])
                        denominator += sum([gamma[t]['numerator'][j]/gamma[t]['denominator'] for t in range(T)])
                    # print(denominator, vk.action, vk.daytime, obs_list[T-1].action, obs_list[T-1].daytime, numerator)
                    model._emissions[j-1][daytime][action] = numerator / denominator

        gap_norm = np.linalg.norm(model._emissions - previous_emissions) + np.linalg.norm(model._transitions - previous_transitions)
        break
    if debug:
        print(model._transitions)
        print(model._emissions)
        

    print(f'final iteration#{iteration}: {gap_norm:.3f}')
    return model
    # pass

# Part II
# ---------

# Given an initialized HMM model,
# and some set of observations, this function evaluates
# the liklihood that this set of observation was indeed
# generated from the given model.
# Here, the obs_list is a list containing
# instances of the `Observation` class.
# The output returned has to be floatint point between
# 0 and 1


def Liklihood(model: HMM, obs_list: list) -> float:
    A = model._transitions
    B = model.B
    N = model._N
    Pi = model.Pi
    alpha = np.array([[Pi(SuspectState(i+1))] for i in range(N)])
    T = len(obs_list)
    for t in range(T):
        alpha  =  alpha * np.array([[B(SuspectState(j+1), obs_list[t])] for j in range(len(alpha))])
        if t == T-1:
            break
        alpha  =  A.T @ alpha
    return sum([alpha[i][0] for i in range(N)])

    # pass


# // Part III
# //---------

# Given an initialized model, and a sequence of observation,
# returns  a list of the same size, which contains
# the most likely # states the model
# was in to produce the given observations.
# returns An array/sequence of states that the model was in to
# produce the corresponding sequence of observations

def GetHiddenStates(model: HMM, obs_list: list) -> list:
    N = model._N
    A = model._transitions
    B = model.B
    Pi = model.Pi
    viterbi = np.array([[Pi(SuspectState(i+1)) * B(SuspectState(i+1), obs_list[0])] for i in range(N)])
    backtraces = []
    state_sequence = []
    T = len(obs_list)
    for t in range(T):
        if t < T-1:
            backtrace  = np.argmax(viterbi.flatten() * A.T * np.array([[B(SuspectState(j+1), obs_list[t])] for j in range(N)]), axis=1, keepdims=True)
            viterbi      = np.max (viterbi.flatten() * A.T * np.array([[B(SuspectState(j+1), obs_list[t])] for j in range(N)]), axis=1, keepdims=True)
        else:
            backtrace  = np.argmax(viterbi.flatten())
            score = np.max(viterbi.flatten())
        backtraces.append(backtrace)
        
    state_list = []
    while len(backtraces):
        bt = backtraces.pop()
        state_list.append(SuspectState(bt+1))
        if len(backtraces) == 0:
            break
        backtraces[-1] = backtraces[-1][bt]
    state_list = state_list[::-1]
    return state_list
    



    # pass


# database = ReadDataset()
# model = LearnModel(database)
# o1 = [
#     Observation(Daytime.Day, Action.Home),
#     Observation(Daytime.Evening, Action.Eating),
#     Observation(Daytime.Night, Action.Home),
#     Observation(Daytime.Day, Action.Home),
#     Observation(Daytime.Evening, Action.Eating),
#     Observation(Daytime.Night, Action.Roaming),
# ]
# p = Liklihood(model, o1)
