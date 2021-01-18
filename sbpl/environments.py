from __future__ import print_function
from __future__ import absolute_import
from __future__ import division


import sbpl._sbpl_module
import tempfile
import os
import shutil
import cv2
import numpy as np

from bc_gym_planning_env.utilities.coordinate_transformations import world_to_pixel, pixel_to_world
from bc_gym_planning_env.utilities.path_tools import get_pixel_footprint
from sbpl.motion_primitives import MotionPrimitives, dump_motion_primitives


class EnvNAVXYTHETALAT_InitParms(sbpl._sbpl_module.EnvNAVXYTHETALAT_InitParms):
    pass


class EnvironmentNAVXYTHETALAT(sbpl._sbpl_module.EnvironmentNAVXYTHETALAT):

    def __init__(self, footprint, motion_primitives, costmap_data, env_params,
                 override_primitive_kernels=True, use_full_kernels=False):
        primitives_folder = tempfile.mkdtemp()
        try:
            dump_motion_primitives(motion_primitives, os.path.join(primitives_folder, 'primitives.mprim'))
            sbpl._sbpl_module.EnvironmentNAVXYTHETALAT.__init__(
                self,
                footprint,
                os.path.join(primitives_folder, 'primitives.mprim'),
                costmap_data,
                env_params,
                not override_primitive_kernels
            )
            if override_primitive_kernels:
                self._override_primitive_kernels(motion_primitives, footprint, use_full_kernels)

        finally:
            shutil.rmtree(primitives_folder)

    @staticmethod
    def create_from_config(environment_config_filename):
        return sbpl._sbpl_module.EnvironmentNAVXYTHETALAT(environment_config_filename)

    def get_motion_primitives(self):
        params = self.get_params()
        return MotionPrimitives(params.cellsize_m, params.numThetas, self.get_motion_primitives_list())

    def _override_primitive_kernels(self, motion_primitives, footprint, use_full_kernels):
        resolution = motion_primitives.get_resolution()
        print('Setting up motion primitive kernels..')
        for p in motion_primitives.get_primitives():

            primitive_start = pixel_to_world(np.zeros((2,)), np.zeros((2,)), resolution)
            primitive_states = p.get_intermediate_states().copy()
            primitive_states[:, :2] += primitive_start

            full_cv_kernel_x = []
            full_cv_kernel_y = []
            for pose in primitive_states:
                kernel = get_pixel_footprint(pose[2], footprint, resolution)

                kernel_center = (kernel.shape[1] //  2, kernel.shape[0] //  2)
                kernel = np.where(kernel)

                px, py = world_to_pixel(pose[:2], np.zeros((2,)), resolution)
                full_cv_kernel_x.append(kernel[1] + (px-kernel_center[0]))
                full_cv_kernel_y.append(kernel[0] + (py-kernel_center[1]))

            full_cv_kernel_x = np.hstack(full_cv_kernel_x)
            full_cv_kernel_y = np.hstack(full_cv_kernel_y)
            full_cv_kernel = np.column_stack((full_cv_kernel_x, full_cv_kernel_y)).astype(np.int32)

            row_view = np.ascontiguousarray(full_cv_kernel).view(
                np.dtype((np.void, full_cv_kernel.dtype.itemsize * full_cv_kernel.shape[1])))
            _, idx = np.unique(row_view, return_index=True)
            full_cv_kernel = np.ascontiguousarray(full_cv_kernel[idx])

            if use_full_kernels:
                self.set_primitive_collision_pixels(p.starttheta_c, p.motprimID, full_cv_kernel)
            else:
                min_x, max_x = np.amin(full_cv_kernel[:, 0]), np.amax(full_cv_kernel[:, 0])
                min_y, max_y = np.amin(full_cv_kernel[:, 1]), np.amax(full_cv_kernel[:, 1])

                temp_img = np.zeros((max_y - min_y+1, max_x - min_x+1), dtype=np.uint8)
                temp_img[full_cv_kernel[:, 1] - min_y, full_cv_kernel[:, 0] - min_x] = 255
                _, contours, _ = cv2.findContours(temp_img.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
                contour = contours[0].reshape(-1, 2)

                perimeter_kernel = np.column_stack((contour[:, 0] + min_x, contour[:, 1] + min_y)).astype(np.int32)

                self.set_primitive_collision_pixels(p.starttheta_c, p.motprimID, perimeter_kernel)

        print('Done.')