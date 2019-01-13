/**
 * @file python_wrapper.cpp
 *
 * @copyright Software License Agreement (BSD License)
 * Original work Copyright 2019 Oleg Y. Sinyavskiy
 * All Rights Reserved.
 * For a full description see the file named LICENSE.
 *
 * Original authors: Oleg Y. Sinyavskiy
 *
 */

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <pybind11/functional.h>

#include <iostream>
#include <limits>

#include <sbpl/headers.h>
#include <sbpl/runners.h>

namespace pybind11 {
    template <typename T>
    using safe_array = typename pybind11::array_t<T, pybind11::array::c_style>;
}

namespace py = pybind11;
using namespace pybind11::literals;


class EnvironmentNAVXYTHETALATWrapper {
public:
    EnvironmentNAVXYTHETALATWrapper(const char* envCfgFilename) {
        // initialize true map from the environment file without perimeter or motion primitives
        if (!_environment.InitializeEnv(envCfgFilename)) {
            throw SBPL_Exception("ERROR: InitializeEnv failed");
        }
    }

    EnvironmentNAVXYTHETALATWrapper(
        const py::safe_array<double>& footprint_array,
        const char* motPrimFilename,
        const py::safe_array<unsigned char>& map_data_array,
        EnvNAVXYTHETALAT_InitParms params) {
        auto footprint = footprint_array.unchecked<2>();

        std::vector<sbpl_2Dpt_t> perimeterptsV;
        if (footprint_array.shape()[1] != 2) {
            throw SBPL_Exception("Footprint has to be n by 2 dims");
        }

        if (map_data_array.shape()[0] != params.size_y || map_data_array.shape()[1] != params.size_x) {
            throw SBPL_Exception("Map data shape and params size_x size_y should be equal");
        }

        for (unsigned int i = 0; i < footprint_array.shape()[0]; i++) {
            perimeterptsV.push_back(sbpl_2Dpt_t(footprint(i, 0), footprint(i, 1)));
        }
        const unsigned char* map_data = &map_data_array.unchecked<2>()(0, 0);
        bool envInitialized = _environment.InitializeEnv(perimeterptsV, motPrimFilename, map_data, params);
        if (!envInitialized) {
            throw SBPL_Exception("ERROR: InitializeEnv failed");
        }
    }

    const EnvironmentNAVXYTHETALAT& env() const {return this->_environment;}
    EnvironmentNAVXYTHETALAT& env() {return this->_environment;}

    EnvNAVXYTHETALAT_InitParms get_params() const {

        EnvNAVXYTHETALAT_InitParms params;
        std::vector<SBPL_xytheta_mprimitive> motionprimitiveV;
        // get environment parameters from the true environment
        _environment.GetEnvParms(&params.size_x, &params.size_y, &params.numThetas,
                                 &params.startx, &params.starty, &params.starttheta,
                                 &params.goalx, &params.goaly, &params.goaltheta,
                                 &params.cellsize_m, &params.nominalvel_mpersecs,
                                 &params.timetoturn45degsinplace_secs, &params.obsthresh, &motionprimitiveV,
                                 &params.costinscribed_thresh, &params.costcircum_thresh);
        return params;
    }


    py::safe_array<unsigned char> get_costmap() const {

        EnvNAVXYTHETALAT_InitParms params = this->get_params();

        py::safe_array<unsigned char> result_array({params.size_y, params.size_x});
        auto result = result_array.mutable_unchecked();

        for (int y = 0; y < params.size_y; y++) {
            for (int x = 0; x < params.size_x; x++) {
                result(y, x) = _environment.GetMapCost(x, y);
            }
        }

        return result_array;
    }


private:
    EnvironmentNAVXYTHETALAT _environment;

};


class SBPLPlannerWrapper {
public:
    SBPLPlannerWrapper(SBPLPlanner* pPlanner)
       : _pPlanner(pPlanner)
    {

    }

    const SBPLPlanner* planner() const {return this->_pPlanner;}
    SBPLPlanner* planner() {return this->_pPlanner;}

    void set_start_goal_from_env(const EnvironmentNAVXYTHETALATWrapper& envWrapper) {

        MDPConfig MDPCfg;
        // initialize MDP info
        if (!envWrapper.env().InitializeMDPCfg(&MDPCfg)) {
            throw SBPL_Exception("ERROR: InitializeMDPCfg failed");
        }
        // set the start and goal states for the planner and other search variables
        if (_pPlanner->set_start(MDPCfg.startstateid) == 0) {
            throw SBPL_Exception("ERROR: failed to set start state");
        }
        if (_pPlanner->set_goal(MDPCfg.goalstateid) == 0) {
            throw SBPL_Exception("ERROR: failed to set goal state");
        }
    }

    void set_planning_params(double initialEpsilon, bool searchUntilFirstSolution) {
        _pPlanner->set_initialsolution_eps(initialEpsilon);
        _pPlanner->set_search_mode(searchUntilFirstSolution);
    }

private:
    SBPLPlanner* _pPlanner;
};


template<typename PlannerT>
class SpecificPlannerWrapper: public SBPLPlannerWrapper {
public:
    SpecificPlannerWrapper(EnvironmentNAVXYTHETALATWrapper& envWrapper, bool bforwardsearch)
       : _planner(&envWrapper.env(), bforwardsearch)
       , SBPLPlannerWrapper(&_planner)
    {

    }
private:
    PlannerT _planner;
};


typedef SpecificPlannerWrapper<ARAPlanner> ARAPlannerWrapper;
typedef SpecificPlannerWrapper<ADPlanner> ADPlannerWrapper;
typedef SpecificPlannerWrapper<anaPlanner> anaPlannerWrapper;



int run_planandnavigatexythetalat(
    const EnvironmentNAVXYTHETALATWrapper& trueEnvWrapper,
    EnvironmentNAVXYTHETALATWrapper& envWrapper,
    SBPLPlannerWrapper& plannerWrapper,
    char* motPrimFilename) {

    double allocated_time_secs_foreachplan = 10.0; // in seconds

    double goaltol_x = 0.001, goaltol_y = 0.001, goaltol_theta = 0.001;

    SBPLPlanner* planner = plannerWrapper.planner();

    // compute sensing as a square surrounding the robot with length twice that of the
    // longest motion primitive

    double maxMotPrimLengthSquared = 0.0;
    double maxMotPrimLength = 0.0;
    const EnvNAVXYTHETALATConfig_t* cfg = envWrapper.env().GetEnvNavConfig();
    for (int i = 0; i < (int)cfg->mprimV.size(); i++) {
        const SBPL_xytheta_mprimitive& mprim = cfg->mprimV.at(i);
        int dx = mprim.endcell.x;
        int dy = mprim.endcell.y;
        if (dx * dx + dy * dy > maxMotPrimLengthSquared) {
            std::cout << "Found a longer motion primitive with dx = " << dx << " and dy = " << dy
                << " from starttheta = " << (int)mprim.starttheta_c << std::endl;
            maxMotPrimLengthSquared = dx * dx + dy * dy;
        }
    }
    maxMotPrimLength = sqrt((double)maxMotPrimLengthSquared);
    std::cout << "Maximum motion primitive length: " << maxMotPrimLength << std::endl;

    int sensingRange = (int)ceil(maxMotPrimLength);

    // environment parameters
    EnvNAVXYTHETALAT_InitParms params = trueEnvWrapper.get_params();

    // create an empty map
    unsigned char* map = new unsigned char[params.size_x * params.size_y];
    for (int i = 0; i < params.size_x * params.size_y; i++) {
        map[i] = 0;
    }

    navigationLoop(
        envWrapper.env(),
        trueEnvWrapper.env(),
        map,
        planner,
        params,
        sensingRange,
        allocated_time_secs_foreachplan,
        goaltol_x, goaltol_y, goaltol_theta
    );

    delete[] map;

    return 1;
}


/**
 * @brief pybind module
 * @details pybind module for all planners, systems and interfaces
 *
 */
PYBIND11_MODULE(_sbpl_module, m) {
    m.doc() = "Python wrapper for SBPL planners";

    m.def("planandnavigatexythetalat", &run_planandnavigatexythetalat);

    py::class_<EnvironmentNAVXYTHETALATWrapper>(m, "EnvironmentNAVXYTHETALAT")
       .def(py::init<const char*>(),
           "config_filename"_a
       )
       .def(py::init<const py::safe_array<double>&,
                     const char*,
                     const py::safe_array<unsigned char>&,
                     EnvNAVXYTHETALAT_InitParms>())
       .def("get_params", &EnvironmentNAVXYTHETALATWrapper::get_params)
       .def("get_costmap", &EnvironmentNAVXYTHETALATWrapper::get_costmap)
    ;

    py::class_<EnvNAVXYTHETALAT_InitParms>(m, "EnvNAVXYTHETALAT_InitParms")
        .def_readwrite("size_x", &EnvNAVXYTHETALAT_InitParms::size_x)
        .def_readwrite("size_y", &EnvNAVXYTHETALAT_InitParms::size_y)
        .def_readwrite("numThetas", &EnvNAVXYTHETALAT_InitParms::numThetas)
        .def_readwrite("startx", &EnvNAVXYTHETALAT_InitParms::startx)
        .def_readwrite("starty", &EnvNAVXYTHETALAT_InitParms::starty)
        .def_readwrite("starttheta", &EnvNAVXYTHETALAT_InitParms::starttheta)
        .def_readwrite("goalx", &EnvNAVXYTHETALAT_InitParms::goalx)
        .def_readwrite("goaly", &EnvNAVXYTHETALAT_InitParms::goaly)
        .def_readwrite("goaltheta", &EnvNAVXYTHETALAT_InitParms::goaltheta)
        .def_readwrite("cellsize_m", &EnvNAVXYTHETALAT_InitParms::cellsize_m)
        .def_readwrite("nominalvel_mpersecs", &EnvNAVXYTHETALAT_InitParms::nominalvel_mpersecs)
        .def_readwrite("timetoturn45degsinplace_secs", &EnvNAVXYTHETALAT_InitParms::timetoturn45degsinplace_secs)
        .def_readwrite("obsthresh", &EnvNAVXYTHETALAT_InitParms::obsthresh)
        .def_readwrite("costinscribed_thresh", &EnvNAVXYTHETALAT_InitParms::costinscribed_thresh)
        .def_readwrite("costcircum_thresh", &EnvNAVXYTHETALAT_InitParms::costcircum_thresh)
    ;


    py::class_<SBPLPlannerWrapper> base_planner(m, "SBPLPlannerWrapper");
    base_planner
        .def("set_start_goal_from_env", &SBPLPlannerWrapper::set_start_goal_from_env)
        .def("set_planning_params", &SBPLPlannerWrapper::set_planning_params,
            "initial_epsilon"_a,
            "search_until_first_solution"_a
        )
    ;

    py::class_<ARAPlannerWrapper>(m, "ARAPlanner", base_planner)
       .def(py::init<EnvironmentNAVXYTHETALATWrapper&, bool>())
    ;

    py::class_<ADPlannerWrapper>(m, "ADPlanner", base_planner)
       .def(py::init<EnvironmentNAVXYTHETALATWrapper&, bool>())
    ;

    py::class_<anaPlannerWrapper>(m, "anaPlanner", base_planner)
       .def(py::init<EnvironmentNAVXYTHETALATWrapper&, bool>())
    ;

}
