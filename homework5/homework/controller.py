import pystk
import numpy as np


def control(aim_point, current_vel):

    drift_window = 0.25
    steering_mag = 20
    max_velocity =20
    drift_steering_mag = 20

    action = pystk.Action()
    if current_vel < max_velocity:
        action.acceleration = 1
    else:
        action.acceleration = 0

    if aim_point[0] <= -drift_window:
        action.steer = np.tanh(aim_point[0] * drift_steering_mag)
        action.drift = True

    elif aim_point[0] >= drift_window:
        action.steer =np.tanh(aim_point[0] * drift_steering_mag)
        action.drift = True

    else:

        action.steer = np.tanh(aim_point[0] * steering_mag)
        action.nitro = True

    """
    Your code here
    Hint: Use action.acceleration (0..1) to change the velocity. Try targeting a target_velocity (e.g. 20).
    Hint: Use action.brake to True/False to brake (optionally)
    Hint: Use action.steer to turn the kart towards the aim_point, clip the steer angle to -1..1
    Hint: You may want to use action.drift=True for wide turns (it will turn faster)
    """

    return action


if __name__ == '__main__':
    from .utils import PyTux
    from argparse import ArgumentParser

    def test_controller(args):
        import numpy as np
        pytux = PyTux()
        for t in args.track:
            steps, how_far = pytux.rollout(t, control, max_frames=1000, verbose=args.verbose)
            print(steps, how_far)
        pytux.close()


    parser = ArgumentParser()
    parser.add_argument('track', nargs='+')
    parser.add_argument('-v', '--verbose', action='store_true')
    args = parser.parse_args()
    test_controller(args)
