#include "laik_runtime.hpp"

Laik_Instance* hpcg_laik_instance = 0;
Laik_Group* hpcg_laik_world = 0;

void hpcg_set_laik_context(Laik_Instance* inst, Laik_Group* world) {
  hpcg_laik_instance = inst;
  hpcg_laik_world = world;
}
