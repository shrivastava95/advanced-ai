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
    filepath = './database.txt' 
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

def argmax(l):
    index, max_val = -1, -1
    for i in range(len(l)):
        if l[i] > max_val:
            index, max_val = i, l[i]
    return index

class HMM(object):
    def __init__(self):
        self.transitions = [
           [0, 1, 0, 0, 0],
           [0, 0, 0.5, 0, 0.5],
           [0, 0, 0, 1, 0],
           [0.5, 0, 0, 0, 0.5],
           [0, 0, 1, 0, 0] 
        ]
        self.emissions = [
            [[0.1/3, 0.1/3, 0.1/3], [0.05, 0.05, 0.7], [0.1/3, 0.1/3, 0.1/3], [0.1/3, 0.1/3, 0.1/3], [0, 0, 0]],
            [[0.1/3, 0.1/3, 0.1/3], [0.1/3, 0.1/3, 0.1/3], [0.1/3, 0.1/3, 0.1/3], [0.1/3, 0.1/3, 0.1/3], [0.05, 0.7, 0.05]],
            [[0.7/3, 0.7/3, 0.7/3], [0, 0, 0], [0.1/3, 0.1/3, 0.1/3], [0.1/3, 0.1/3, 0.1/3], [0.1/3, 0.1/3, 0.1/3]],
            [[0.1/3, 0.1/3, 0.1/3], [0.1/3, 0.1/3, 0.1/3], [0.7/3, 0.7/3, 0.7/3], [0.7/3, 0.7/3, 0.7/3], [0.1/3, 0.1/3, 0.1/3]]
        ]
        self.pis = [1, 0, 0, 0, 0]
    
    # Complete the HMM implementation.
    # The three function below must
    # be implemented.

    def A(self, a, b) -> float:
        return self.transitions[a-1][b-1]
        # Compute the probablity of going from one
        # state a to the other state b
        # Hint: The necessary code does not need
        # to be within this function only
        # pass

    def B(self, a, b: Observation) -> float:
        return self.emissions[int(b.action)-9][a-1][int(b.daytime)-6]
        # Compute the probablity of obtaining
        # observation b from state a
        # Hint: The necessary code does not need
        # to be within this function only
        # pass

    def Pi(self, a) -> float:
        return self.pis[a-1]
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
    model = HMM()
    for obs_list in dataset:
        T = len(obs_list)
        alpha = [[0 for _ in range(T)] for _ in range(5)]
        beta = [[0 for _ in range(T)] for _ in range(5)]
        for i in range(5):
            alpha[i][0] = model.Pi(i+1)*model.B(i+1, obs_list[0])

        for t in range(1, T):
            for s in range(5):
                alpha[s][t] = sum([alpha[s_][t-1]*model.A(s_+1, s+1)*model.B(s+1, obs_list[t]) for s_ in range(5)])
        
        val_alpha = sum([alpha[s][T-1] for s in range(5)])
    
        for i in range(5):
            beta[i][4] = 1

        for t in range(T-1):
            for s in range(5):
                beta[s][t] = sum([beta[s_][t]*model.A(s_+1, s+1)*model.B(s+1, obs_list[t+1]) for s_ in range(5)])

        val_beta = sum([model.Pi(s+1) * model.B(s+1, obs_list[0]) * beta[s][0] for s in range(5)])

        # # E- step
        # gamma = [[0 for _ in range(T)] for _ in range(5)]
        # eta = [[0 for _ in range(T)] for _ in range(5)]
        # for t in range(0,T):
        #     for i in range(5):
        #         gamma[i][t] = (alpha[i][t] * beta[i][t])/ alpha[4][T-1]
        #         eta[i][t] = sum([alpha[i][t-1] * beta[s_][t]*model.A(s_+1, s+1)*model.B(s+1, obs_list[t+1]) for s_ in range(5)])

        # M - step

        # try:
        #     for i in range(5):
        #         for j in range(T):
        #             model.transitions[i][j] = sum([gamma[i][t] for t in range(T)])/

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
    T = len(obs_list)
    alpha = [[0 for _ in range(T)] for _ in range(5)]
    for i in range(5):
        alpha[i][0] = model.Pi(i+1)*model.B(i+1, obs_list[0])

    for t in range(1, T):
        for s in range(5):
            alpha[s][t] = sum([alpha[s_][t-1]*model.A(s_+1, s+1)*model.B(s+1, obs_list[t]) for s_ in range(5)])
    
    probability = sum([alpha[s][T-1] for s in range(5)])
    return probability
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
    T = len(obs_list)
    viterbi = [[0 for _ in range(T)] for _ in range(5)]
    backpointer = [[0 for _ in range(T)] for _ in range(5)]

    for i in range(5):
        viterbi[i][0] = model.Pi(i+1)*model.B(i+1, obs_list[0])

    for t in range(1, T):
        for s in range(5):
            viterbi[s][t] = max([viterbi[s_][t-1]*model.A(s_+1, s+1)*model.B(s+1, obs_list[t]) for s_ in range(5)])
            backpointer[s][t] = argmax([viterbi[s_][t-1]*model.A(s_+1, s+1)*model.B(s+1, obs_list[t]) for s_ in range(5)])

    bestpathprob = max([viterbi[s][T-1] for s in range(5)])
    bestpathpointer = argmax([viterbi[s][T-1] for s in range(5)])
    bestpath = backpointer[bestpathpointer]
    mapping = {}
    State_List = [SuspectState.Planning, SuspectState.Scouting,
              SuspectState.Burglary, SuspectState.Migrating, SuspectState.Misc]
    for i in range(5):
        mapping[str(i)] = State_List[i]
    return [mapping[str(i)] for i in bestpath]
    # pass
