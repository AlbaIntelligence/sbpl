
#include <cmath>
#include <cstring>
#include <iostream>
#include <sstream>
#include <string>

using namespace std;

#include <sbpl/headers.h>
#include <sbpl/runners.h>


std::string PlannerTypeToStr(PlannerType plannerType)
{
    switch (plannerType) {
    case PLANNER_TYPE_ADSTAR:
        return std::string("adstar");
    case PLANNER_TYPE_ARASTAR:
        return std::string("arastar");
    case PLANNER_TYPE_PPCP:
        return std::string("ppcp");
    case PLANNER_TYPE_RSTAR:
        return std::string("rstar");
    case PLANNER_TYPE_VI:
        return std::string("vi");
    case PLANNER_TYPE_ANASTAR:
        return std::string("anastar");
    default:
        return std::string("invalid");
    }
}

PlannerType StrToPlannerType(const char* str)
{
    if (!strcmp(str, "adstar")) {
        return PLANNER_TYPE_ADSTAR;
    }
    else if (!strcmp(str, "arastar")) {
        return PLANNER_TYPE_ARASTAR;
    }
    else if (!strcmp(str, "ppcp")) {
        return PLANNER_TYPE_PPCP;
    }
    else if (!strcmp(str, "rstar")) {
        return PLANNER_TYPE_RSTAR;
    }
    else if (!strcmp(str, "vi")) {
        return PLANNER_TYPE_VI;
    }
    else if (!strcmp(str, "anastar")) {
        return PLANNER_TYPE_ANASTAR;
    }
    else {
        return INVALID_PLANNER_TYPE;
    }
}


std::string EnvironmentTypeToStr(EnvironmentType environmentType)
{
    switch (environmentType) {
    case ENV_TYPE_2D:
        return std::string("2d");
    case ENV_TYPE_2DUU:
        return std::string("2duu");
    case ENV_TYPE_XYTHETA:
        return std::string("xytheta");
    case ENV_TYPE_XYTHETAMLEV:
        return std::string("xythetamlev");
    case ENV_TYPE_ROBARM:
        return std::string("robarm");
    default:
        return std::string("invalid");
    }
}

EnvironmentType StrToEnvironmentType(const char* str)
{
    if (!strcmp(str, "2d")) {
        return ENV_TYPE_2D;
    }
    else if (!strcmp(str, "2duu")) {
        return ENV_TYPE_2DUU;
    }
    else if (!strcmp(str, "xytheta")) {
        return ENV_TYPE_XYTHETA;
    }
    else if (!strcmp(str, "xythetamlev")) {
        return ENV_TYPE_XYTHETAMLEV;
    }
    else if (!strcmp(str, "robarm")) {
        return ENV_TYPE_ROBARM;
    }
    else {
        return INVALID_ENV_TYPE;
    }
}


void navigationIteration(
    double& startx, double& starty, double& starttheta,
    const EnvironmentNAVXYTHETALAT& trueenvironment_navxythetalat,
    EnvironmentNAVXYTHETALAT& environment_navxythetalat,
    vector<sbpl_2Dcell_t>& sensecells,
    unsigned char* map,
    SBPLPlanner* planner,
    const EnvNAVXYTHETALAT_InitParms& params,
    double allocated_time_secs_foreachplan,
    FILE* fSol
)
{

    vector<int> preds_of_changededgesIDV;
    vector<nav2dcell_t> changedcellsV;
    vector<int> solution_stateIDs_V;


    //simulate sensor data update
    bool bChanges = false;
    bool bPrint = false;


    // simulate sensing the cells
    for (int i = 0; i < (int)sensecells.size(); i++) {
        int x = CONTXY2DISC(startx, params.cellsize_m) + sensecells.at(i).x;
        int y = CONTXY2DISC(starty, params.cellsize_m) + sensecells.at(i).y;

        // ignore if outside the map
        if (x < 0 || x >= params.size_x || y < 0 || y >= params.size_y) {
            continue;
        }

        int index = x + y * params.size_x;
        unsigned char truecost = trueenvironment_navxythetalat.GetMapCost(x, y);
        // update the cell if we haven't seen it before
        if (map[index] != truecost) {
            map[index] = truecost;
            environment_navxythetalat.UpdateCost(x, y, map[index]);
            printf("setting cost[%d][%d] to %d\n", x, y, map[index]);
            bChanges = true;
            // store the changed cells
            nav2dcell_t nav2dcell;
            nav2dcell.x = x;
            nav2dcell.y = y;
            changedcellsV.push_back(nav2dcell);
        }
    }

    double TimeStarted = clock();

    // if necessary notify the planner of changes to costmap
    if (bChanges) {
        if (dynamic_cast<ARAPlanner*> (planner) != NULL) {
            ((ARAPlanner*)planner)->costs_changed(); //use by ARA* planner (non-incremental)
        }
        else if (dynamic_cast<ADPlanner*> (planner) != NULL) {
            // get the affected states
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

    // write the solution to sol.txt
    fprintf(fSol, "plan time=%.5f eps=%.2f\n", (clock() - TimeStarted) / ((double)CLOCKS_PER_SEC),
            planner->get_solution_eps());
    fflush(fSol);

    vector<sbpl_xy_theta_pt_t> xythetaPath;

    environment_navxythetalat.ConvertStateIDPathintoXYThetaPath(&solution_stateIDs_V, &xythetaPath);
    printf("actual path (with intermediate poses) size=%d\n", (unsigned int)xythetaPath.size());
    for (unsigned int i = 0; i < xythetaPath.size(); i++) {
        fprintf(fSol, "%.3f %.3f %.3f\n", xythetaPath.at(i).x, xythetaPath.at(i).y, xythetaPath.at(i).theta);
    }
    fprintf(fSol, "*********\n");

    for (int j = 1; j < (int)solution_stateIDs_V.size(); j++) {
        int newx, newy, newtheta = 0;
        environment_navxythetalat.GetCoordFromState(solution_stateIDs_V[j], newx, newy, newtheta);
        fprintf(fSol, "%d %d %d\n", newx, newy, newtheta);
    }
    fflush(fSol);

    // print the map (robot's view of the world and current plan)
//        int startindex = startx_c + starty_c * size_x;
//        int goalindex = goalx_c + goaly_c * size_x;
//        for (int y = 0; bPrintMap && y < size_y; y++) {
//            for (int x = 0; x < size_x; x++) {
//                int index = x + y * size_x;
//                int cost = map[index];
//                cost = environment_navxythetalat.GetMapCost(x, y);
//
//                // check to see if it is on the path
//                bool bOnthePath = false;
//                for (int j = 1; j < (int)solution_stateIDs_V.size(); j++) {
//                    int newx, newy, newtheta = 0;
//                    environment_navxythetalat.GetCoordFromState(solution_stateIDs_V[j], newx, newy, newtheta);
//                    if (x == newx && y == newy) bOnthePath = true;
//                }
//
//                if (index != startindex && index != goalindex && !bOnthePath) {
//                    printf("%3d ", cost);
//                }
//                else if (index == startindex) {
//                    printf("  X ");
//                }
//                else if (index == goalindex) {
//                    printf("  G ");
//                }
//                else if (bOnthePath) {
//                    printf("  * ");
//                }
//                else {
//                    printf("? ");
//                }
//            }
//            printf("\n");
//        }

    // move along the path
    if (bPlanExists && (int)xythetaPath.size() > 1) {
        //get coord of the successor
        int newx, newy, newtheta;

        // move until we move into the end of motion primitive
        environment_navxythetalat.GetCoordFromState(solution_stateIDs_V[1], newx, newy, newtheta);

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

    if (bPrint) {
        printf("System Pause (return=%d)\n", system("pause"));
    }
}



void navigationLoop(
    EnvironmentNAVXYTHETALAT& environment_navxythetalat,
    const EnvironmentNAVXYTHETALAT& trueenvironment_navxythetalat,
    unsigned char* map,
    SBPLPlanner* planner,
    const EnvNAVXYTHETALAT_InitParms& params,
    int sensingRange,
    double allocated_time_secs_foreachplan,
    double goaltol_x, double goaltol_y, double goaltol_theta) {

    vector<sbpl_2Dcell_t> sensecells;
    for (int x = -sensingRange; x <= sensingRange; x++) {
        for (int y = -sensingRange; y <= sensingRange; y++) {
            sensecells.push_back(sbpl_2Dcell_t(x, y));
        }
    }

    double startx = params.startx;
    double starty = params.starty;
    double starttheta = params.starttheta;

    // create a file to hold the solution vector
    const char* sol = "sol.txt";
    FILE* fSol = fopen(sol, "w");
    if (fSol == NULL) {
        throw SBPL_Exception("ERROR: could not open solution file");
    }

    // print the goal pose
    int goalx_c = CONTXY2DISC(params.goalx, params.cellsize_m);
    int goaly_c = CONTXY2DISC(params.goaly, params.cellsize_m);
    int goaltheta_c = ContTheta2Disc(params.goaltheta, params.numThetas);
    printf("goal_c: %d %d %d\n", goalx_c, goaly_c, goaltheta_c);

    // now comes the main loop
    while (fabs(startx - params.goalx) > goaltol_x || fabs(starty - params.goaly) > goaltol_y || fabs(starttheta - params.goaltheta)
        > goaltol_theta) {
        navigationIteration(
            startx, starty, starttheta,
            trueenvironment_navxythetalat,
            environment_navxythetalat,
            sensecells,
            map,
            planner,
            params,
            allocated_time_secs_foreachplan,
            fSol
        );
    }

    printf("goal reached!\n");

    fflush(NULL);
    fclose(fSol);
}



/*******************************************************************************
 * planandnavigatexythetalat
 * @brief An example simulation of how a robot would use (x,y,theta) lattice
 *        planning.
 *
 * @param envCfgFilename The environment config file. See
 *                       sbpl/env_examples/nav3d/ for examples
 *******************************************************************************/
int planandnavigatexythetalat(PlannerType plannerType, char* envCfgFilename, char* motPrimFilename, bool forwardSearch)
{

    double allocated_time_secs_foreachplan = 10.0; // in seconds
    double initialEpsilon = 3.0;
    bool bsearchuntilfirstsolution = false;
    bool bforwardsearch = forwardSearch;

    double goaltol_x = 0.001, goaltol_y = 0.001, goaltol_theta = 0.001;

    bool bPrintMap = false;

    EnvironmentNAVXYTHETALAT environment_navxythetalat;
    EnvironmentNAVXYTHETALAT trueenvironment_navxythetalat;

    // set the perimeter of the robot
    // it is given with 0, 0, 0 robot ref. point for which planning is done.
    vector<sbpl_2Dpt_t> perimeterptsV;
    sbpl_2Dpt_t pt_m;
    double halfwidth = 0.01;
    double halflength = 0.01;
    pt_m.x = -halflength;
    pt_m.y = -halfwidth;
    perimeterptsV.push_back(pt_m);
    pt_m.x = halflength;
    pt_m.y = -halfwidth;
    perimeterptsV.push_back(pt_m);
    pt_m.x = halflength;
    pt_m.y = halfwidth;
    perimeterptsV.push_back(pt_m);
    pt_m.x = -halflength;
    pt_m.y = halfwidth;
    perimeterptsV.push_back(pt_m);

    // initialize true map from the environment file without perimeter or motion primitives
    if (!trueenvironment_navxythetalat.InitializeEnv(envCfgFilename)) {
        throw SBPL_Exception("ERROR: InitializeEnv failed");
    }

    // environment parameters
    EnvNAVXYTHETALAT_InitParms params;
    std::vector<SBPL_xytheta_mprimitive> motionprimitiveV;

    // get environment parameters from the true environment
    trueenvironment_navxythetalat.GetEnvParms(
        &params.size_x, &params.size_y, &params.numThetas, &params.startx, &params.starty, &params.starttheta,
        &params.goalx, &params.goaly, &params.goaltheta,
        &params.cellsize_m, &params.nominalvel_mpersecs,
        &params.timetoturn45degsinplace_secs, &params.obsthresh, &motionprimitiveV,
        &params.costinscribed_thresh, &params.costcircum_thresh);

    // print the map
    if (bPrintMap) {
        printf("true map:\n");
        for (int y = 0; y < params.size_y; y++) {
            for (int x = 0; x < params.size_x; x++) {
                printf("%3d ", trueenvironment_navxythetalat.GetMapCost(x, y));
            }
            printf("\n");
        }
        printf("System Pause (return=%d)\n", system("pause"));
    }

    // create an empty map
    unsigned char* map = new unsigned char[params.size_x * params.size_y];
    for (int i = 0; i < params.size_x * params.size_y; i++) {
        map[i] = 0;
    }

    // check the start and goal obtained from the true environment
    printf("start: %f %f %f, goal: %f %f %f\n",
        params.startx, params.starty, params.starttheta,
        params.goalx, params.goaly, params.goaltheta);

    bool envInitialized = environment_navxythetalat.InitializeEnv(perimeterptsV, motPrimFilename, map, params);

    if (!envInitialized) {
        throw SBPL_Exception("ERROR: InitializeEnv failed");
    }

    MDPConfig MDPCfg;

    // initialize MDP info
    if (!environment_navxythetalat.InitializeMDPCfg(&MDPCfg)) {
        throw SBPL_Exception("ERROR: InitializeMDPCfg failed");
    }

    // create a planner
    SBPLPlanner* planner = NULL;
    switch (plannerType) {
    case PLANNER_TYPE_ARASTAR:
        printf("Initializing ARAPlanner...\n");
        planner = new ARAPlanner(&environment_navxythetalat, bforwardsearch);
        break;
    case PLANNER_TYPE_ADSTAR:
        printf("Initializing ADPlanner...\n");
        planner = new ADPlanner(&environment_navxythetalat, bforwardsearch);
        break;
    case PLANNER_TYPE_RSTAR:
        printf("Invalid configuration: xytheta environment does not support rstar planner...\n");
        return 0;
    case PLANNER_TYPE_ANASTAR:
        printf("Initializing anaPlanner...\n");
        planner = new anaPlanner(&environment_navxythetalat, bforwardsearch);
        break;
    default:
        printf("Invalid planner type\n");
        break;
    }

    // set the start and goal states for the planner and other search variables
    if (planner->set_start(MDPCfg.startstateid) == 0) {
        throw SBPL_Exception("ERROR: failed to set start state");
    }
    if (planner->set_goal(MDPCfg.goalstateid) == 0) {
        throw SBPL_Exception("ERROR: failed to set goal state");
    }
    planner->set_initialsolution_eps(initialEpsilon);
    planner->set_search_mode(bsearchuntilfirstsolution);

    // compute sensing as a square surrounding the robot with length twice that of the
    // longest motion primitive

    double maxMotPrimLengthSquared = 0.0;
    double maxMotPrimLength = 0.0;
    const EnvNAVXYTHETALATConfig_t* cfg = environment_navxythetalat.GetEnvNavConfig();
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

    navigationLoop(
        environment_navxythetalat,
        trueenvironment_navxythetalat,
        map,
        planner,
        params,
        sensingRange,
        allocated_time_secs_foreachplan,
        goaltol_x, goaltol_y, goaltol_theta
    );

    delete[] map;
    delete planner;

    return 1;
}
