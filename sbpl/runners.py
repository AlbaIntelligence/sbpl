
import _sbpl_module
import os
import numpy as np


def mprim_folder():
    current_dir = os.path.dirname(__file__)
    return os.path.abspath(os.path.join(current_dir, '../matlab/mprim'))


def env_examples_folder():
    current_dir = os.path.dirname(__file__)
    return os.path.abspath(os.path.join(current_dir, '../env_examples'))



if __name__ == '__main__':
    true_env = _sbpl_module.EnvironmentNAVXYTHETALAT(
        os.path.join(env_examples_folder(), 'nav3d/env1.cfg'))
    params = true_env.get_params()

    footprint = np.zeros((4, 2))
    halfwidth = 0.01
    halflength = 0.01
    footprint[0, :] = (-halflength, -halfwidth)
    footprint[1, :] = (halflength, -halfwidth)
    footprint[2, :] = (halflength, halfwidth)
    footprint[3, :] = (-halflength, halfwidth)

    motion_primitives = os.path.join(mprim_folder(), 'pr2.mprim')
    # motion_primitives = os.path.join(mprim_folder(), 'pr2_10cm.mprim')
    
    empty_map = np.zeros((params.size_y, params.size_x), dtype=np.uint8)
    env = _sbpl_module.EnvironmentNAVXYTHETALAT(footprint, motion_primitives, empty_map, params)

    _sbpl_module.planandnavigatexythetalat(
        # "arastar",
        "adstar",
        # os.path.join(env_examples_folder(), 'nav3d/willow-25mm-inflated-env.cfg'),
        true_env,
        env,
        os.path.join(mprim_folder(), 'pr2.mprim'),
        # os.path.join(mprim_folder(), 'pr2_10cm.mprim'),
        False)
