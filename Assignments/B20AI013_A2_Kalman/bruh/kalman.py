from env import *
import numpy as np
class KalmanFilter:
    def __init__(self, noise_velocity, noise_position) -> None:
        # Complete this function to construct

        # Assume that nothing is known 
        # about the state of the target at this instance
        self.nv = noise_velocity
        self.np = noise_position


        # yk = Hk.sk + eta
        self.Hk = np.eye(6) 
        # Rk  = cov(eta) = [[ eye(3) * noise_position^2, 0                         ]
        #                   [ 0                        , eye(3) * noise_velocity^2 ]]
        self.Rk = np.concatenate(
            [
                np.concatenate([np.eye(3) * self.np**2, np.zeros_like(np.eye(3))], axis=1),
                np.concatenate([np.zeros_like(np.eye(3)), np.eye(3) * self.nv**2], axis=1),
            ], 
            axis=0,
        )

        # sk+1 = Fk.sk + Gk.Uk + ξk
        # [xk+1] = [1  Δt].[xk] + [0    0].[ 0] + ξk 
        # [vk+1]   [0   1] [vk]   [0   Δt] [ak]
        self.delta_t = 1
        self.Fk = np.concatenate(
            [
                np.concatenate([np.eye(3)       , np.eye(3) * self.delta_t], axis=1),
                np.concatenate([np.zeros([3, 3]), np.eye(3)               ], axis=1),
            ],
            axis=0
        )
        self.Gk = np.concatenate(
            [
                np.concatenate([np.zeros([3, 3]), np.zeros([3, 3])        ], axis=1),
                np.concatenate([np.zeros([3, 3]), np.eye(3) * self.delta_t], axis=1),
            ]
            , axis=0
        )
        # cov(ξk) = Qk
        # Qk = uncertainty associated with the dynamics of the system
        self.beta = 0.01
        self.Qk = np.eye(6) * self.beta


        # Pk = uncertainty associated with the current estimate of state
        self.Pk = np.eye(6) * self.beta

        # the initial estimate of the state
        self.state = np.zeros([6, 1])

    def input(self, observed_state:State, accel:numpy.ndarray, justUpdated:bool):
        # propagate step
        # self.yi = np.concatenate([observed_state.position.reshape([-1, 1]), observed_state.velocity.reshape([-1, 1])], axis=0)
        self.state = self.Fk @ self.state + self.Gk @ np.concatenate([np.zeros([3, 1]), accel.reshape([3, 1])], axis=0)
        self.Pk = self.Fk @ self.Pk @ self.Fk.T + self.Qk

        if justUpdated:
            # update step
            self.yi = np.concatenate([observed_state.position.reshape([-1, 1]), observed_state.velocity.reshape([-1, 1])], axis=0)
            self.Pk = np.linalg.inv(np.linalg.inv(self.Pk) + self.Hk.T @ np.linalg.inv(self.Rk) @ self.Hk)
            self.kalman_gain = self.Pk @ self.Hk.T @ np.linalg.inv(self.Rk)
            self.state = self.state + self.kalman_gain @ (self.yi - self.Hk @ self.state)
            


        # This function is executed multiple times during the reading.
        # When an observation is read, the `justUpdated` is true, otherwise it is false
        
        # accel is the acceleration(control) vector. 
        # It is dynamically obtained regardless of the state of the RADAR 
        # (i.e regardless of `justUpdated`) 

        # When `justUpdated` is false, the state is the same as the previously provided state
        # (i.e, given `observed_state` is not updated, it's the same as the previous outdated one)


        # Complete this function where current estimate of target is updated

    def get_current_estimate(self)->State:
        
        # Complete this function where the current state of the target is returned
        return State(x = np.array([si[0] for si in self.state[:3]]), v = np.array([si[0] for si in self.state[3:]]))
        
        pass