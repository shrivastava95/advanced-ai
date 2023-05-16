import kalman
import env
from kalman import *

Target_List = [
    (lambda x:numpy.zeros(3, dtype=float), 5),
    (lambda x:numpy.ones(3, dtype=float), 5),
    (lambda x:numpy.ones(3, dtype=float) * 3, 5),
    (lambda x:numpy.array([x.state.velocity[1], x.state.velocity[2], x.state.velocity[0]]), 5),
    (lambda x:numpy.array([x.state.velocity[1], -x.state.velocity[2], -x.state.velocity[0]]), 15),
]

if __name__ == "__main__":
    test_len = 50
    update_time = 5
    for acc_fn, noise in Target_List:
        target = AbstractTarget(
            State(Vector3D(), Vector3D()),
            Vector3D(), acc_fn, noise, noise
        )
        k_filter = KalmanFilter(noise, noise)
        error = env.RunEnvironment(
            test_len, update_time, 
            acc_fn, noise, KalmanFilter
        )
        print(f"Error after test run: {error}")