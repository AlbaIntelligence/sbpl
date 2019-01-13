

enum PlannerType
{
    INVALID_PLANNER_TYPE = -1,
    PLANNER_TYPE_ADSTAR,
    PLANNER_TYPE_ARASTAR,
    PLANNER_TYPE_PPCP,
    PLANNER_TYPE_RSTAR,
    PLANNER_TYPE_VI,
    PLANNER_TYPE_ANASTAR,

    NUM_PLANNER_TYPES
};


enum EnvironmentType
{
    INVALID_ENV_TYPE = -1, ENV_TYPE_2D, ENV_TYPE_2DUU, ENV_TYPE_XYTHETA, ENV_TYPE_XYTHETAMLEV, ENV_TYPE_ROBARM,

    NUM_ENV_TYPES
};

std::string PlannerTypeToStr(PlannerType plannerType);
PlannerType StrToPlannerType(const char* str);

std::string EnvironmentTypeToStr(EnvironmentType environmentType);
EnvironmentType StrToEnvironmentType(const char* str);

int planandnavigatexythetalat(PlannerType plannerType, char* envCfgFilename, char* motPrimFilename, bool forwardSearch);


void navigationIteration(
    double& startx, double& starty, double& starttheta,
    const EnvironmentNAVXYTHETALAT& trueenvironment_navxythetalat,
    EnvironmentNAVXYTHETALAT& environment_navxythetalat,
    std::vector<sbpl_2Dcell_t>& sensecells,
    unsigned char* map,
    SBPLPlanner* planner,
    const EnvNAVXYTHETALAT_InitParms& params,
    double allocated_time_secs_foreachplan,
    double& plan_time,
    double& solution_epsilon,
    std::vector<sbpl_xy_theta_pt_t>& xythetaPath,
    std::vector<sbpl_xy_theta_cell_t>& xythetaCellPath
);

void navigationLoop(
    EnvironmentNAVXYTHETALAT& environment_navxythetalat,
    const EnvironmentNAVXYTHETALAT& trueenvironment_navxythetalat,
    unsigned char* map,
    SBPLPlanner* planner,
    const EnvNAVXYTHETALAT_InitParms& params,
    int sensingRange,
    double allocated_time_secs_foreachplan,
    double goaltol_x, double goaltol_y, double goaltol_theta);