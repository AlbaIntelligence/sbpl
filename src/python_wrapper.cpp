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



//struct SBPL_xytheta_mprimitive
//{
//    int motprimID;
//    unsigned char starttheta_c;
//    int additionalactioncostmult;
//    sbpl_xy_theta_cell_t endcell;
//    double turning_radius;
//    //intermptV start at 0,0,starttheta and end at endcell in continuous
//    //domain with half-bin less to account for 0,0 start
//    std::vector<sbpl_xy_theta_pt_t> intermptV;
//};

class SBPL_xytheta_mprimitiveWrapper {
public:
    SBPL_xytheta_mprimitiveWrapper(const SBPL_xytheta_mprimitive& motion_primitive)
        : _primitive(motion_primitive)
    { }

    int get_motprimID() const {return _primitive.motprimID;};
    unsigned char get_start_theta_cell() const {return _primitive.starttheta_c;};
    int get_additionalactioncostmult() const {return _primitive.additionalactioncostmult;};

    double get_turning_radius() const {return _primitive.turning_radius;};

    py::safe_array<int> get_endcell() const {
        py::safe_array<int> result_array({3});
        auto result = result_array.mutable_unchecked();
        result(0) = _primitive.endcell.x;
        result(1) = _primitive.endcell.y;
        result(2) = _primitive.endcell.theta;
        return result_array;
    }

    py::safe_array<double> get_intermediate_states() const {
        py::safe_array<double> result_array({(int)_primitive.intermptV.size(), 3});
        auto result = result_array.mutable_unchecked();
        for (auto i =0; i < _primitive.intermptV.size(); ++i) {
            result(i, 0) = _primitive.intermptV[i].x;
            result(i, 1) = _primitive.intermptV[i].y;
            result(i, 2) = _primitive.intermptV[i].theta;
        }
        return result_array;
    }


private:
    SBPL_xytheta_mprimitive _primitive;

};


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

    std::vector<SBPL_xytheta_mprimitiveWrapper> get_motion_primitives() const {
        std::vector<SBPL_xytheta_mprimitiveWrapper> primitives;
        const EnvNAVXYTHETALATConfig_t* cfg = _environment.GetEnvNavConfig();
        for (int i = 0; i < (int)cfg->mprimV.size(); i++) {
            primitives.push_back(SBPL_xytheta_mprimitiveWrapper(cfg->mprimV.at(i)));
        }

        return std::move(primitives);
    }

    py::safe_array<int> xytheta_real_to_cell(const py::safe_array<double>& pose_array) const {

        py::safe_array<int> result_array({3});
        auto result = result_array.mutable_unchecked();

        auto pose = pose_array.unchecked<1>();

        auto params = this->get_params();
        result(0) = CONTXY2DISC(pose(0), params.cellsize_m);
        result(1) = CONTXY2DISC(pose(1), params.cellsize_m);
        result(2) = ContTheta2Disc(pose(2), params.numThetas);

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


class IncrementalSensingWrapper {
public:
    IncrementalSensingWrapper(int sensingRange) {
        for (int x = -sensingRange; x <= sensingRange; x++) {
            for (int y = -sensingRange; y <= sensingRange; y++) {
                _sensecells.push_back(sbpl_2Dcell_t(x, y));
            }
        }
    }

    const std::vector<sbpl_2Dcell_t>& get_sensecells() const {return _sensecells;};

private:
    std::vector<sbpl_2Dcell_t> _sensecells;
};


py::tuple py_navigation_iteration(
    const EnvironmentNAVXYTHETALATWrapper& trueEnvWrapper,
    EnvironmentNAVXYTHETALATWrapper& envWrapper,
    SBPLPlannerWrapper& plannerWrapper,
    const py::safe_array<double>& start_pose_array,
    const IncrementalSensingWrapper& incrementalSensingWrapper,
    py::safe_array<unsigned char>& empty_map) {

    double allocated_time_secs_foreachplan = 10.0; // in seconds

    // environment parameters
    EnvNAVXYTHETALAT_InitParms params = trueEnvWrapper.get_params();

    unsigned char* map = &empty_map.mutable_unchecked<2>()(0, 0);

    auto start_pose = start_pose_array.unchecked<1>();

    double startx = start_pose(0);
    double starty = start_pose(1);
    double starttheta = start_pose(2);

    SBPLPlanner* planner = plannerWrapper.planner();

    double plan_time, solution_epsilon;
    std::vector<sbpl_xy_theta_pt_t> xythetaPath;
    std::vector<sbpl_xy_theta_cell_t> xythetaCellPath;

    navigationIteration(
        startx, starty, starttheta,
        trueEnvWrapper.env(),
        envWrapper.env(),
        incrementalSensingWrapper.get_sensecells(),
        map,
        planner,
        params,
        allocated_time_secs_foreachplan,
        plan_time,
        solution_epsilon,
        xythetaPath,
        xythetaCellPath
    );

    py::safe_array<double> new_start_pose_array({3});
    auto new_start_pose = new_start_pose_array.mutable_unchecked();
    new_start_pose(0) = startx;
    new_start_pose(1) = starty;
    new_start_pose(2) = starttheta;

    return py::make_tuple(new_start_pose_array);
}

/**
 * @brief pybind module
 * @details pybind module for all planners, systems and interfaces
 *
 */
PYBIND11_MODULE(_sbpl_module, m) {
    m.doc() = "Python wrapper for SBPL planners";

    m.def("navigation_iteration", &py_navigation_iteration);

    //struct SBPL_xytheta_mprimitive
//{
//    int motprimID;
//    unsigned char starttheta_c;
//    int additionalactioncostmult;
//    sbpl_xy_theta_cell_t endcell;
//    double turning_radius;
//    //intermptV start at 0,0,starttheta and end at endcell in continuous
//    //domain with half-bin less to account for 0,0 start
//    std::vector<sbpl_xy_theta_pt_t> intermptV;
//};

    py::class_<SBPL_xytheta_mprimitiveWrapper>(m, "XYThetaMotionPrimitive")
        .def_property_readonly("motprimID", &SBPL_xytheta_mprimitiveWrapper::get_motprimID)
        .def_property_readonly("starttheta_c", &SBPL_xytheta_mprimitiveWrapper::get_start_theta_cell)
        .def_property_readonly("additionalactioncostmult", &SBPL_xytheta_mprimitiveWrapper::get_additionalactioncostmult)
        .def_property_readonly("endcell", &SBPL_xytheta_mprimitiveWrapper::get_endcell)
        .def_property_readonly("turning_radius", &SBPL_xytheta_mprimitiveWrapper::get_turning_radius)
        .def("get_intermediate_states", &SBPL_xytheta_mprimitiveWrapper::get_intermediate_states)
    ;

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
       .def("get_motion_primitives", &EnvironmentNAVXYTHETALATWrapper::get_motion_primitives)
       .def("xytheta_real_to_cell", &EnvironmentNAVXYTHETALATWrapper::xytheta_real_to_cell)
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

    py::class_<IncrementalSensingWrapper>(m, "IncrementalSensing")
        .def(py::init<int>())
    ;
}
