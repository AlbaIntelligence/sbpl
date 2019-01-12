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


int run_planandnavigatexythetalat(char* plannerName, char* envCfgFilename, char* motPrimFilename, bool forwardSearch) {

    return planandnavigatexythetalat(
        StrToPlannerType(plannerName), envCfgFilename, motPrimFilename, forwardSearch);

}


/**
 * @brief pybind module
 * @details pybind module for all planners, systems and interfaces
 *
 */
PYBIND11_MODULE(_sbpl_module, m) {
   m.doc() = "Python wrapper for SBPL planners";

   m.def("planandnavigatexythetalat", &run_planandnavigatexythetalat);

}
