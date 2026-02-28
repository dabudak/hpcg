#include "laik_reductions.hpp"

#include <cstring>
#include <map>
#include <utility>

#include "laik_runtime.hpp"

namespace {
std::map<std::pair<uint64_t, Laik_Type*>, Laik_Data*> data_objects;

void laik_helper(const void* sendBuf, void* recvBuf, uint64_t n, Laik_Type* data_type,
                 Laik_ReductionOperation ro_type, Laik_Partitioner* p1,
                 Laik_Partitioner* p2) {
  if (!hpcg_laik_instance || !hpcg_laik_world) return;

  Laik_Data* data = 0;
  auto key = std::make_pair(n, data_type);
  auto it = data_objects.find(key);
  if (it != data_objects.end()) {
    data = it->second;
  } else {
    Laik_Space* space = laik_new_space_1d(hpcg_laik_instance, (int64_t)n);
    data = laik_new_data(space, data_type);
    data_objects.insert(std::make_pair(key, data));
  }

  laik_switchto_new_partitioning(data, hpcg_laik_world, p1, LAIK_DF_None, LAIK_RO_None);

  uint64_t count = 0;
  if (data_type == laik_Int32) {
    int* base = 0;
    laik_get_map_1d(data, 0, (void**)&base, &count);
    std::memcpy((void*)base, sendBuf, count * sizeof(int));
  } else if (data_type == laik_Double) {
    double* base = 0;
    laik_get_map_1d(data, 0, (void**)&base, &count);
    std::memcpy((void*)base, sendBuf, count * sizeof(double));
  } else if (data_type == laik_Int64) {
    int64_t* base = 0;
    laik_get_map_1d(data, 0, (void**)&base, &count);
    std::memcpy((void*)base, sendBuf, count * sizeof(int64_t));
  } else if (data_type == laik_UInt64) {
    uint64_t* base = 0;
    laik_get_map_1d(data, 0, (void**)&base, &count);
    std::memcpy((void*)base, sendBuf, count * sizeof(uint64_t));
  }

  laik_switchto_new_partitioning(data, hpcg_laik_world, p2, LAIK_DF_Preserve, ro_type);

  if (data_type == laik_Int32) {
    int* base = 0;
    laik_get_map_1d(data, 0, (void**)&base, &count);
    std::memcpy(recvBuf, (void*)base, count * sizeof(int));
  } else if (data_type == laik_Double) {
    double* base = 0;
    laik_get_map_1d(data, 0, (void**)&base, &count);
    std::memcpy(recvBuf, (void*)base, count * sizeof(double));
  } else if (data_type == laik_Int64) {
    int64_t* base = 0;
    laik_get_map_1d(data, 0, (void**)&base, &count);
    std::memcpy(recvBuf, (void*)base, count * sizeof(int64_t));
  } else if (data_type == laik_UInt64) {
    uint64_t* base = 0;
    laik_get_map_1d(data, 0, (void**)&base, &count);
    std::memcpy(recvBuf, (void*)base, count * sizeof(uint64_t));
  }
}
}

void laik_allreduce(const void* sendBuf, void* recvBuf, uint64_t n, Laik_Type* data_type,
                    Laik_ReductionOperation ro_type) {
  laik_helper(sendBuf, recvBuf, n, data_type, ro_type, laik_All, laik_All);
}

void laik_broadcast(const void* sendBuf, void* recvBuf, uint64_t n, Laik_Type* data_type) {
  laik_helper(sendBuf, recvBuf, n, data_type, LAIK_RO_None, laik_Master, laik_All);
}

void laik_barrier(void) {
  int32_t data = 0;
  laik_helper((void*)&data, (void*)&data, 1, laik_Int32, LAIK_RO_None, laik_All, laik_All);
}
