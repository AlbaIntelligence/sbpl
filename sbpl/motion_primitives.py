from __future__ import print_function
from __future__ import absolute_import
from __future__ import division
import _sbpl_module

import numpy as np
import os
import cv2

from sbpl.utilities.map_drawing_utils import draw_trajectory, draw_arrow
from sbpl.utilities.path_tools import pixel_to_world, angle_discrete_to_cont


def mprim_folder():
    current_dir = os.path.dirname(__file__)
    return os.path.abspath(os.path.join(current_dir, '../matlab/mprim'))


class MotionPrimitives(object):
    def __init__(self, resolution, number_of_angles, mprim_list):
        self._primitives = mprim_list
        self._resolution = resolution
        self._number_of_angles = number_of_angles

    def get_primitives(self):
        return self._primitives

    def get_resolution(self):
        return self._resolution

    def get_number_of_angles(self):
        return self._number_of_angles


def load_motion_pritimives(mprim_filename):
    """Load motion primitives from file (uses proxy environment to use SBPL code)"""
    with open(mprim_filename) as f:
        l0 = f.readline().strip()
        if not l0.startswith('resolution_m: '):
            raise AssertionError("Invalid resolution entry")
        resolution = float(l0[len('resolution_m: '):])

        l1 = f.readline().strip()
        if not l1.startswith('numberofangles: '):
            raise AssertionError("Invalid number of angles entry")
        number_of_angles = int(l1[len('numberofangles: '):])

    params = _sbpl_module.EnvNAVXYTHETALAT_InitParms()
    params.size_x = 1
    params.size_y = 1
    params.numThetas = number_of_angles
    params.cellsize_m = resolution
    params.startx = 0
    params.starty = 0
    params.starttheta = 0

    params.goalx = 0
    params.goaly = 0
    params.goaltheta = 0
    empty_map = np.zeros((params.size_y, params.size_x), dtype=np.uint8)
    env = _sbpl_module.EnvironmentNAVXYTHETALAT(np.array([[0., 0.]]), mprim_filename, empty_map, params)
    return MotionPrimitives(resolution, number_of_angles, env.get_motion_primitives_list())


def check_motion_primitives(motion_primitives):
    angle_to_primitive = {}
    for p in motion_primitives.get_primitives():
        try:
            angle_to_primitive[p.starttheta_c].append(p)
        except KeyError:
            angle_to_primitive[p.starttheta_c] = [p]

    # every angle has the same number of primitives
    assert np.all(len(angle_to_primitive.values()[0]) == np.array([len(v) for v in angle_to_primitive.values()]))
    assert set(angle_to_primitive.keys()) == set(range(len(angle_to_primitive)))

    # angles are ordered
    angles = np.array([p.starttheta_c for p in motion_primitives.get_primitives()])
    assert np.all(np.diff(angles) >= 0)

    for angle_primitives in angle_to_primitive.values():
        # ids are ordered inside the angle
        primitive_ids = np.array([p.motprimID for p in angle_primitives])
        assert np.all(np.arange(len(angle_primitives)) == primitive_ids)

    return angle_to_primitive


def debug_motion_primitives(motion_primitives):
    angle_to_primitive = check_motion_primitives(motion_primitives)

    all_angles = np.arange(motion_primitives.get_number_of_angles())*np.pi*2/motion_primitives.get_number_of_angles()
    print(all_angles)
    print(np.degrees(all_angles))
    print(np.degrees(0.6654))

    for angle_c, primitives in angle_to_primitive.items():
        print('------', angle_c)
        for p in primitives:

            if np.all(p.get_intermediate_states()[0, :2] == p.get_intermediate_states()[:, :2]):
                turn_in_place = 'turn in place'
            else:
                turn_in_place = ''
            print(turn_in_place, p.endcell, p.get_intermediate_states()[0], p.get_intermediate_states()[-1])
            final_float_state = pixel_to_world(p.endcell[:2], np.zeros((2,)), motion_primitives.get_resolution())
            final_float_angle = angle_discrete_to_cont(p.endcell[2], motion_primitives.get_number_of_angles())
            try:
                np.testing.assert_array_almost_equal(final_float_state, p.get_intermediate_states()[-1][:2])
            except AssertionError:
                print("POSE DOESN'T LAND TO A CELL", final_float_state, p.get_intermediate_states()[-1])

            try:
                np.testing.assert_array_almost_equal(final_float_angle, p.get_intermediate_states()[-1][2], decimal=3)
            except AssertionError:
                print("ANGLE DOESN'T LAND TO AN ANGLE CELL", np.degrees(final_float_angle), np.degrees(p.get_intermediate_states()[-1][2]))

        endcells = np.array([p.endcell for p in primitives])
        image_half_width = np.amax(np.amax(np.abs(endcells), 0)[:2]) + 1
        zoom = 40
        img = np.full((zoom*image_half_width*2, zoom*image_half_width*2, 3), 255, dtype=np.uint8)
        resolution = motion_primitives.get_resolution()/zoom
        origin = np.array((-image_half_width*motion_primitives.get_resolution(),
                           -image_half_width*motion_primitives.get_resolution()))

        for p in primitives:
            draw_arrow(img, p.get_intermediate_states()[0], 0.7*motion_primitives.get_resolution(),
                       origin, resolution, color=(0, 0, 0))

            draw_trajectory(img, resolution, origin, p.get_intermediate_states()[:, :2], color=(0, 200, 0))
            draw_arrow(img, p.get_intermediate_states()[-1], 0.5*motion_primitives.get_resolution(),
                       origin, resolution, color=(0, 0, 200))
        cv2.imshow('a', img)
        cv2.waitKey(-1)


def dump_motion_primitives(motion_primitives, filename):
    check_motion_primitives(motion_primitives)

    with open(filename, 'w') as f:
        f.write('resolution_m: %.6f\n' % motion_primitives.get_resolution())
        f.write('numberofangles: %d\n' % motion_primitives.get_number_of_angles())
        f.write('totalnumberofprimitives: %d\n' % len(motion_primitives.get_primitives()))

        for p in motion_primitives.get_primitives():
            f.write('primID: %d\n' % p.motprimID)
            f.write('startangle_c: %d\n' % p.starttheta_c)
            f.write('endpose_c: %d %d %d\n' % (p.endcell[0], p.endcell[1], p.endcell[2]))
            f.write('additionalactioncostmult: %d\n' % p.additionalactioncostmult)
            states = p.get_intermediate_states()
            f.write('intermediateposes: %d\n' % len(states))
            for s in states:
                f.write('%.4f %.4f %.4f\n' % (s[0], s[1], s[2]))


def assert_motion_primitives_equal(motion_primitives_0, motion_primitives_1):
    assert motion_primitives_0.get_resolution() == motion_primitives_1.get_resolution()
    assert motion_primitives_0.get_number_of_angles() == motion_primitives_1.get_number_of_angles()
    assert len(motion_primitives_0.get_primitives()) == len(motion_primitives_1.get_primitives())

    for p0, p1 in zip(motion_primitives_0.get_primitives(), motion_primitives_1.get_primitives()):
        assert p0.motprimID == p1.motprimID
        assert p0.starttheta_c == p1.starttheta_c
        np.testing.assert_array_equal(p0.endcell, p1.endcell)
        assert p0.additionalactioncostmult == p1.additionalactioncostmult
        np.testing.assert_almost_equal(p0.get_intermediate_states(), p1.get_intermediate_states(), decimal=4)



if __name__ == '__main__':
    mprimtives = load_motion_pritimives(os.path.join(mprim_folder(), 'custom/gtx_32_10.mprim'))
    debug_motion_primitives(mprimtives)
