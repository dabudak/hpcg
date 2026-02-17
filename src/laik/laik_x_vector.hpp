#ifndef HPCG_LAIK_X_VECTOR_HPP
#define HPCG_LAIK_X_VECTOR_HPP

#include <laik.h>
#include <map>
#include <set>
#include <vector>
#include "SparseMatrix.hpp"

struct Laik_Blob {
  char* name;
  Laik_Data* values;
  bool exchangesValues;
  local_int_t localLength;
};

typedef struct _partition_d {
  global_int_t size;
  Geometry* geom;
  int numberOfNeighbours;
  int* neighbors;
  std::vector<global_int_t>* localToGlobalMap;
  local_int_t* elementsToSend;
  local_int_t* receiveLength;
  global_int_t* externalLocalToGlobal;
  local_int_t numberOfExternalValues;
  std::map<int, std::set<global_int_t>> receiveList;
  std::vector<global_int_t> allExternalIndices;
  std::vector<int> allExternalOffsets;
  bool halo;
} partition_d;

void partitioner_alg_for_x_vector(Laik_RangeReceiver* r, Laik_PartitionerParams* p);
void partitioner_alg_for_rows(Laik_RangeReceiver* r, Laik_PartitionerParams* p);
Laik_Blob* init_blob(const SparseMatrix& A, bool exchangesValues, char* name, Laik_Partitioning* localP, Laik_Partitioning* extP);
void init_partition_data(SparseMatrix& A, partition_d* local, partition_d* ext);
void init_partitionings(SparseMatrix& A, partition_d* local, partition_d* ext, Laik_Group* world);

#endif