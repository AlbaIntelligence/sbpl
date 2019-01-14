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

    py::safe_array<int> sense_environment(
        const py::safe_array<double>&  start_pose_array,
        const EnvironmentNAVXYTHETALATWrapper& true_environment_wrapper,
        EnvironmentNAVXYTHETALATWrapper& environment_to_update_wrapper) const {

        auto start_pose = start_pose_array.unchecked<1>();

        double startx = start_pose(0);
        double starty = start_pose(1);
        double starttheta = start_pose(2);

        const EnvironmentNAVXYTHETALAT& true_environment = true_environment_wrapper.env();
        EnvironmentNAVXYTHETALAT& environment_to_update = environment_to_update_wrapper.env();
        auto params = true_environment_wrapper.get_params();

        std::vector<nav2dcell_t> changedcellsV;
        // simulate sensing the cells
        for (int i = 0; i < (int)_sensecells.size(); i++) {
            int x = CONTXY2DISC(startx, params.cellsize_m) + _sensecells.at(i).x;
            int y = CONTXY2DISC(starty, params.cellsize_m) + _sensecells.at(i).y;

            // ignore if outside the map
            if (x < 0 || x >= params.size_x || y < 0 || y >= params.size_y) {
                continue;
            }

            int index = x + y * params.size_x;
            unsigned char truecost = true_environment.GetMapCost(x, y);
            // update the cell if we haven't seen it before
            if (environment_to_update.GetMapCost(x, y) != truecost) {
                environment_to_update.UpdateCost(x, y, truecost);
                // store the changed cells
                nav2dcell_t nav2dcell;
                nav2dcell.x = x;
                nav2dcell.y = y;
                changedcellsV.push_back(nav2dcell);
            }
        }

        py::safe_array<int> changed_cells_array({(int)changedcellsV.size(), 2});
        int* p_changed_cells = &changed_cells_array.mutable_unchecked()(0, 0);
        memcpy(p_changed_cells, &changedcellsV[0].x, sizeof(int)*changedcellsV.size()*2);

        return changed_cells_array;
    }

private:
    std::vector<sbpl_2Dcell_t> _sensecells;
};


py::tuple py_navigation_iteration(
    const EnvironmentNAVXYTHETALATWrapper& trueEnvWrapper,
    EnvironmentNAVXYTHETALATWrapper& envWrapper,
    SBPLPlannerWrapper& plannerWrapper,
    const py::safe_array<double>& start_pose_array,
    const py::safe_array<int>& changedcells_array) {

    double allocated_time_secs_foreachplan = 10.0; // in seconds
    // double allocated_time_secs_foreachplan = 2.; // in seconds

    // environment parameters
    EnvNAVXYTHETALAT_InitParms params = trueEnvWrapper.get_params();

    auto start_pose = start_pose_array.unchecked<1>();

    double startx = start_pose(0);
    double starty = start_pose(1);
    double starttheta = start_pose(2);

    SBPLPlanner* planner = plannerWrapper.planner();

    double plan_time, solution_epsilon;
    std::vector<sbpl_xy_theta_pt_t> xythetaPath;
    std::vector<sbpl_xy_theta_cell_t> xythetaCellPath;

    const EnvironmentNAVXYTHETALAT& trueenvironment_navxythetalat = trueEnvWrapper.env();
    EnvironmentNAVXYTHETALAT& environment_navxythetalat = envWrapper.env();

    auto changedcells = changedcells_array.unchecked<2>();
    std::vector<nav2dcell_t> changedcellsV;
    changedcellsV.resize(changedcells_array.shape(0));
    memcpy(&changedcellsV[0].x, &changedcells(0, 0), sizeof(int)*changedcellsV.size()*2);

    double TimeStarted = clock();
    std::vector<int> solution_stateIDs_V;

    // if necessary notify the planner of changes to costmap
    if (changedcellsV.size()) {
        if (dynamic_cast<ARAPlanner*> (planner) != NULL) {
            ((ARAPlanner*)planner)->costs_changed(); //use by ARA* planner (non-incremental)
        }
        else if (dynamic_cast<ADPlanner*> (planner) != NULL) {
            // get the affected states
            std::vector<int> preds_of_changededgesIDV;
            environment_navxythetalat.GetPredsofChangedEdges(&changedcellsV, &preds_of_changededgesIDV);
            // let know the incremental planner about them
            //use by AD* planner (incremental)
            ((ADPlanner*)planner)->update_preds_of_changededges(&preds_of_changededgesIDV);
            printf("%d states were affected\n", (int)preds_of_changededgesIDV.size());
        }
    }

    int startx_c = CONTXY2DISC(startx, params.cellsize_m);
    int starty_c = CONTXY2DISC(starty, params.cellsize_m);
    int starttheta_c = ContTheta2Disc(starttheta, params.numThetas);

    // plan a path
    bool bPlanExists = false;

    printf("new planning...\n");
    bPlanExists = (planner->replan(allocated_time_secs_foreachplan, &solution_stateIDs_V) == 1);
    printf("done with the solution of size=%d and sol. eps=%f\n", (unsigned int)solution_stateIDs_V.size(),
           planner->get_solution_eps());
    environment_navxythetalat.PrintTimeStat(stdout);

    plan_time = (clock() - TimeStarted) / ((double)CLOCKS_PER_SEC);
    solution_epsilon = planner->get_solution_eps();

    environment_navxythetalat.ConvertStateIDPathintoXYThetaPath(&solution_stateIDs_V, &xythetaPath);
    printf("actual path (with intermediate poses) size=%d\n", (unsigned int)xythetaPath.size());
    for (int j = 1; j < (int)solution_stateIDs_V.size(); j++) {
        sbpl_xy_theta_cell_t xytheta_cell;
        environment_navxythetalat.GetCoordFromState(solution_stateIDs_V[j], xytheta_cell.x, xytheta_cell.y, xytheta_cell.theta);
        xythetaCellPath.push_back(xytheta_cell);
    }

    int steps_along_the_path = 20;
    // move along the path
    if (bPlanExists && (int)xythetaPath.size() > 1) {
        //get coord of the successor
        int newx, newy, newtheta;

        // move until we move into the end of motion primitive
        environment_navxythetalat.GetCoordFromState(
            solution_stateIDs_V[std::min((int)solution_stateIDs_V.size()-1, steps_along_the_path)],
            newx, newy, newtheta);

        printf("moving from %d %d %d to %d %d %d\n", startx_c, starty_c, starttheta_c, newx, newy, newtheta);

        // this check is weak since true configuration does not know the actual perimeter of the robot
        if (!trueenvironment_navxythetalat.IsValidConfiguration(newx, newy, newtheta)) {
            throw SBPL_Exception("ERROR: robot is commanded to move into an invalid configuration according to true environment");
        }

        // move
        startx = DISCXY2CONT(newx, params.cellsize_m);
        starty = DISCXY2CONT(newy, params.cellsize_m);
        starttheta = DiscTheta2Cont(newtheta, params.numThetas);

        // update the environment
        int newstartstateID = environment_navxythetalat.SetStart(startx, starty, starttheta);

        // update the planner
        if (planner->set_start(newstartstateID) == 0) {
            throw SBPL_Exception("ERROR: failed to update robot pose in the planner");
        }
    }
    else {
        printf("No move is made\n");
    }

    py::safe_array<double> new_start_pose_array({3});
    auto new_start_pose = new_start_pose_array.mutable_unchecked();
    new_start_pose(0) = startx;
    new_start_pose(1) = starty;
    new_start_pose(2) = starttheta;

    py::safe_array<double> xytheta_path_array({(int)xythetaPath.size(), 3});
    double* p_xytheta_path = &xytheta_path_array.mutable_unchecked()(0, 0);
    memcpy(p_xytheta_path, &xythetaPath[0], sizeof(double)*xythetaPath.size()*3);

    py::safe_array<int> xytheta_cell_path_array({(int)xythetaCellPath.size(), 3});
    int* p_xytheta_cell_path = &xytheta_cell_path_array.mutable_unchecked()(0, 0);
    memcpy(p_xytheta_cell_path, &xythetaCellPath[0], sizeof(int)*xythetaCellPath.size()*3);

    return py::make_tuple(new_start_pose_array, xytheta_path_array, xytheta_cell_path_array);
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
        .def("sense_environment", &IncrementalSensingWrapper::sense_environment)
    ;
}
