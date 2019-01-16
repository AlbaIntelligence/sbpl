from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

import sbpl._sbpl_module


def create_planner(planner_name, environment, forward_search):
    return {
        'arastar': sbpl._sbpl_module.ARAPlanner,
        'adstar': sbpl._sbpl_module.ADPlanner,
        'anastar': sbpl._sbpl_module.anaPlanner,
    }[planner_name](environment, forward_search)
