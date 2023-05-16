from hmm import *
import sys

epsilon = 0.2
State_List = [SuspectState.Planning, SuspectState.Scouting,
              SuspectState.Burglary, SuspectState.Migrating, SuspectState.Misc]
Dt_List = [Daytime.Day, Daytime.Evening, Daytime.Night]
Act_List = [Action.Roaming, Action.Eating, Action.Home, Action.Untracked]


def p1a(model: HMM):
    for si in State_List:
        x = 0
        for sj in State_List:
            xi = model.A(si, sj)
            assert xi >= 0 and xi <= 1, f'A({si}, {sj}) probablity invalid'
            x += xi
        x -= 1
        x *= x
        assert x < epsilon, f'A({si}, ...) proabablity distribution invalid'


def p1b(model: HMM):
    for si in State_List:
        x = 0
        for dt in Dt_List:
            for act in Act_List:
                o = Observation(dt, act)
                xi = model.B(si, o)
                assert xi >= 0 and xi <= 1, f'B({si}, {o}) probablity invalid'
                x += xi
        x -= 1
        x *= x
        assert x < epsilon, f'B({si}, ...) proabablity distribution invalid'


def p1c(model: HMM):
    x = 0
    for si in State_List:
        xi = model.Pi(si)
        assert xi >= 0 and xi <= 1, f'Pi({si}) probablity invalid'
        x += xi
    x -= 1
    x *= x
    assert x < epsilon, f'Pi({si}, ...) proabablity distribution invalid'


def p1d(model: HMM):
    p = model.A(SuspectState.Scouting, SuspectState.Burglary)
    q = model.A(SuspectState.Burglary, SuspectState.Scouting)
    assert p > q, "Basic assumption test failed"


def p1e(model: HMM):
    p = model.B(SuspectState.Scouting, Observation(
        Daytime.Day, Action.Roaming))
    q = model.B(SuspectState.Scouting, Observation(
        Daytime.Evening, Action.Roaming))
    r = model.B(SuspectState.Scouting, Observation(
        Daytime.Night, Action.Roaming))
    assert r > q and r > p, "Basic assumption test failed"


def p1f(model: HMM):
    p = model.B(SuspectState.Misc, Observation(Daytime.Evening, Action.Eating))
    assert p > 0, "Basic assumption test failed"


def p1g(model: HMM):
    p = model.Pi(SuspectState.Planning) + model.Pi(SuspectState.Misc)
    q = model.Pi(SuspectState.Burglary) + \
        model.Pi(SuspectState.Scouting) + model.Pi(SuspectState.Migrating)
    assert p > q, "Basic assumption test failed"


def p2a(model: HMM):
    o1 = [
        Observation(Daytime.Day, Action.Home),
        Observation(Daytime.Evening, Action.Eating),
        Observation(Daytime.Night, Action.Home),
        Observation(Daytime.Day, Action.Home),
        Observation(Daytime.Evening, Action.Eating),
        Observation(Daytime.Night, Action.Roaming),
    ]
    p = Liklihood(model, o1)
    assert p >= 0 and p <= 1, 'Invalid value of Liklihood'


def p2b(model: HMM):
    o2 = [
        Observation(Daytime.Day, Action.Roaming),
        Observation(Daytime.Evening, Action.Roaming),
        Observation(Daytime.Night, Action.Untracked),
        Observation(Daytime.Day, Action.Untracked),
        Observation(Daytime.Evening, Action.Home),
        Observation(Daytime.Night, Action.Home),
    ]
    p = Liklihood(model, o2)
    assert p >= 0 and p <= 1, 'Invalid value of Liklihood'


def p2c(model: HMM):
    o1 = [
        Observation(Daytime.Day, Action.Home),
        Observation(Daytime.Evening, Action.Eating),
        Observation(Daytime.Night, Action.Home),
        Observation(Daytime.Day, Action.Home),
        Observation(Daytime.Evening, Action.Eating),
        Observation(Daytime.Night, Action.Roaming),
    ]
    p = Liklihood(model, o1)
    o2 = [
        Observation(Daytime.Day, Action.Roaming),
        Observation(Daytime.Evening, Action.Roaming),
        Observation(Daytime.Night, Action.Untracked),
        Observation(Daytime.Day, Action.Untracked),
        Observation(Daytime.Evening, Action.Home),
        Observation(Daytime.Night, Action.Home),
    ]
    q = Liklihood(model, o2)
    assert p > q, "Basic assumption test failed"


def p3a(model: HMM):
    o = [
        Observation(Daytime.Day, Action.Home),
        Observation(Daytime.Evening, Action.Eating),
        Observation(Daytime.Night, Action.Home),
        Observation(Daytime.Day, Action.Home),
        Observation(Daytime.Evening, Action.Eating),
        Observation(Daytime.Night, Action.Roaming),
    ]
    result = GetHiddenStates(model, o)
    assert isinstance(
        result, list), 'MLE should return a list of SuspectState'
    for x in result:
        assert isinstance(
            x, SuspectState), 'MLE should return a list of SuspectState'
    assert len(result) == len(o), 'Length of list of SuspectState returned by MLE must be equal to the length of observation list in argument'


if __name__ == "__main__":
    unit_tests_list = [
        p1a, p1b, p1c, p1d, p1e, p1f, p1g,
        p2a, p2b, p2c,
        p3a,
    ]
    total = len(unit_tests_list)
    failed = 0
    try:
        database = ReadDataset()
    except:
        print("Could not read database! Ensure that the file 'database.txt' is in the folder and try again!")
        sys.exit(1)
        
    model = LearnModel(database)
    for i, test_fn in enumerate(unit_tests_list):
        try:
            test_fn(model)
        except Exception as e:
            failed += 1
            print(f"Unit test #{i+1} failure: {str(e)}")

    if failed == 0:
        print("All tests have passed successfully!")
    else:
        print(f"{failed} tests failed!")
    sys.exit(failed)
