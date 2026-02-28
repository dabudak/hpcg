#ifndef HPCG_LAIK_REDUCTIONS_HPP
#define HPCG_LAIK_REDUCTIONS_HPP

#include <cstdint>
#include <laik.h>

void laik_broadcast(const void* sendBuf, void* recvBuf, uint64_t n, Laik_Type* data_type);
void laik_allreduce(const void* sendBuf, void* recvBuf, uint64_t n, Laik_Type* data_type, Laik_ReductionOperation ro_type);
void laik_barrier(void);

#endif
