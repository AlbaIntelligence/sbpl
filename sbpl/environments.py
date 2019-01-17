from __future__ import print_function
from __future__ import absolute_import
from __future__ import division


import sbpl._sbpl_module
import tempfile
import os
import shutil
from sbpl.motion_primitives import MotionPrimitives, dump_motion_primitives


class EnvNAVXYTHETALAT_InitParms(sbpl._sbpl_module.EnvNAVXYTHETALAT_InitParms):
    pass


class EnvironmentNAVXYTHETALAT(sbpl._sbpl_module.EnvironmentNAVXYTHETALAT):

    def __init__(self, footprint, motion_primitives, costmap_data, env_params):
        primitives_folder = tempfile.mkdtemp()
        try:
            dump_motion_primitives(motion_primitives, os.path.join(primitives_folder, 'primitives.mprim'))
            print(os.path.join(primitives_folder, 'primitives.mprim'))
            sbpl._sbpl_module.EnvironmentNAVXYTHETALAT.__init__(
                self,
                footprint,
                os.path.join(primitives_folder, 'primitives.mprim'),
                costmap_data,
                env_params)
        finally:
            pass
            # shutil.rmtree(primitives_folder)

    @staticmethod
    def create_from_config(environment_config_filename):
        return sbpl._sbpl_module.EnvironmentNAVXYTHETALAT(environment_config_filename)

    def get_motion_primitives(self):
        params = self.get_params()
        return MotionPrimitives(params.cellsize_m, params.numThetas, self.get_motion_primitives_list())
