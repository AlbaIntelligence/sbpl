
import _sbpl_module
import os


def mprim_folder():
    current_dir = os.path.dirname(__file__)
    return os.path.abspath(os.path.join(current_dir, '../matlab/mprim'))


def env_examples_folder():
    current_dir = os.path.dirname(__file__)
    return os.path.abspath(os.path.join(current_dir, '../env_examples'))



if __name__ == '__main__':
    _sbpl_module.planandnavigatexythetalat(
        # "arastar",
        "adstar",
        # os.path.join(env_examples_folder(), 'nav3d/willow-25mm-inflated-env.cfg'),
        os.path.join(env_examples_folder(), 'nav3d/env1.cfg'),
        os.path.join(mprim_folder(), 'pr2.mprim'),
        # os.path.join(mprim_folder(), 'pr2_10cm.mprim'),
        False)
