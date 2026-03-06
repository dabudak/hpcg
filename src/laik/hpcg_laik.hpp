/**
 * @file hpcg_laik.hpp
 *
 * @version 1.1
 * @date 2023-10-13
 *
 * @copyright Copyright (c) 2023
 *
 */

#ifndef HPCG_LAIK_HPP
#define HPCG_LAIK_HPP

/*
    Defs
*/
// #define REPARTITION

typedef long long allocation_int_t; // Index to the allocation buffer
/*
    Defs -END
*/

/*
    Includes
*/
#ifndef HPCG_NO_LAIK
extern "C"
{
#include <laik.h>
}

// All other custom laik headers within src/laik
#include "laik_x_vector.hpp"
#include "laik_debug.hpp"
#include "laik_reductions.hpp"
#ifdef REPARTITION
#include "laik_repartition.hpp"
#endif
#else
struct Laik_Instance;
struct Laik_Group;
struct Laik_Blob;
struct Laik_Space;
struct Laik_Partitioning;
struct Laik_Data;
struct Laik_Mapping;
struct Laik_TaskRange;
struct Laik_Range;
struct Laik_Reservation;
#endif
/*
    Includes -END
*/

/*
    Important global variables
*/
#ifndef HPCG_NO_LAIK
// Laik context
extern Laik_Instance *hpcg_instance;
extern Laik_Group *world;
#endif
/*
    Important global variables -END
*/

#endif // HPCG_LAIK_HPP
