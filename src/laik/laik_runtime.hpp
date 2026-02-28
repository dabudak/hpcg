#ifndef HPCG_LAIK_RUNTIME_HPP
#define HPCG_LAIK_RUNTIME_HPP

#include <laik.h>

extern Laik_Instance* hpcg_laik_instance;
extern Laik_Group* hpcg_laik_world;

void hpcg_set_laik_context(Laik_Instance* inst, Laik_Group* world);

#endif
