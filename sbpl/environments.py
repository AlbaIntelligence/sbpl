from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

import _sbpl_module

from sbpl.motion_primitives import MotionPrimitives


class EnvironmentNAVXYTHETALAT(_sbpl_module.EnvironmentNAVXYTHETALAT):
    def get_motion_primitives(self):
        params = self.get_params()
        return MotionPrimitives(params.cellsize_m, params.numThetas, self.get_motion_primitives_list())
