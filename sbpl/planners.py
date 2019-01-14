from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

import _sbpl_module


def create_planner(planner_name, environment, forward_search):
    return {
        'arastar': _sbpl_module.ARAPlanner,
        'adstar': _sbpl_module.ADPlanner,
        'anastar': _sbpl_module.anaPlanner,
    }[planner_name](environment, forward_search)
