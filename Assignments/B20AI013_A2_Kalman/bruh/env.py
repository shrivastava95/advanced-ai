import numpy
from numpy.random import standard_normal
from copy import copy

def Vector3D(x=0, y=0, z=0) -> numpy.ndarray:
    return numpy.array([x, y, z], dtype=float)


def RandomVector3D() -> numpy.ndarray:
    return standard_normal((3))

def Error(vec1, vec2) -> float:
    def sq_diff(idx: int) -> float:
        p = vec1[idx] - vec2[idx]
        return p*p
    return sq_diff(0) + sq_diff(1) + sq_diff(2)


class State:
    position: numpy.ndarray
    velocity: numpy.ndarray

    def __init__(self, x, v) -> None:
        self.position = x
        self.velocity = v

# Numpy by default creates shallow copies, which can introduce pitfalls easily
def DeepCopy(s:State)->State:
    return State(copy(s.position), copy(s.velocity))

class AbstractTarget:
    state: State
    accel: numpy.ndarray
    noise_v: float
    noise_x: float

    def __init__(self, s, a, fn, vn, xn) -> None:
        self.state = s
        self.accel = a
        self.updatefn = fn
        self.noise_v = vn
        self.noise_x = xn

    def update(self):
        self.accel = self.updatefn(self)
        self.state.velocity += self.accel
        self.state.position += self.state.velocity


def RunEnvironment(trialCount: int, updateTime: int, a_fn, noise: float, estimator_type) -> float:
    error = 0.0

    estimator = estimator_type(noise, noise)
    
    target = AbstractTarget(
        State(Vector3D(), RandomVector3D()),
        Vector3D(), a_fn, noise, noise
    )

    for _ in range(trialCount):
        target.update()

        curstate = DeepCopy(target.state)
        n = RandomVector3D() * noise
        curstate.position += n
        n = RandomVector3D() * noise
        curstate.velocity += n

        estimator.input(DeepCopy(curstate), target.accel, True) # Read true reading of observation

        for _ in range(1, updateTime):
            target.update()
            estimator.input(DeepCopy(curstate), target.accel, False) # Now reading unupdated observation

        est: State = estimator.get_current_estimate()
        act: State = target.state
        error += Error(est.position, act.position)
    
    return error/trialCount
