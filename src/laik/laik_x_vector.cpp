#include "laik_x_vector.hpp"

#include <assert.h>
#include <cstdio>
#include <string>
#include <cstdlib>

void partitioner_alg_for_x_vector(Laik_RangeReceiver* r, Laik_PartitionerParams* p)
{
  partition_d* data = (partition_d*)laik_partitioner_data(p->partitioner);
  Laik_Space* x_space = p->space;

  assert((global_int_t)laik_space_size(x_space) == data->size);

  int rank = data->geom->rank;
  Laik_Range range;

  // Group contiguous global rows by owner to avoid creating one LAIK range
  // object per equation for the common local-ownership part of the vector.
  int prev_owner = -1;
  global_int_t seg_start = 0;
  for (global_int_t i = 0; i < data->size; i++) {
    int proc = ComputeRankOfMatrixRow(*data->geom, i);
    if (proc < 0 || proc >= data->geom->size) {
      fprintf(stderr, "LAIK partitioner: invalid owner %d for row %lld (size %d)\n",
              proc, (long long)i, data->geom->size);
      proc = data->geom->rank;
    }

    if (prev_owner == -1) {
      prev_owner = proc;
      seg_start = i;
      continue;
    }

    if (proc != prev_owner) {
      laik_range_init_1d(&range, x_space, seg_start, i);
      laik_append_range(r, prev_owner, &range, 1, 0);
      prev_owner = proc;
      seg_start = i;
    }
  }

  if (prev_owner != -1) {
    laik_range_init_1d(&range, x_space, seg_start, data->size);
    laik_append_range(r, prev_owner, &range, 1, 0);
  }

  if (data->halo) {
    // Add external indices needed by this rank
    for (int nb = 0; nb < data->numberOfNeighbours; nb++) {
      for (auto it = data->receiveList[data->neighbors[nb]].begin();
           it != data->receiveList[data->neighbors[nb]].end(); ++it) {
        global_int_t g = *it;
        laik_range_init_1d(&range, x_space, g, g + 1);
        laik_append_range(r, rank, &range, 1, 0);
      }
    }

    // Provide our local indices to neighbors that need them
    local_int_t index_elementsToSend = 0;
    for (int nb = 0; nb < data->numberOfNeighbours; nb++) {
      local_int_t num = data->receiveLength[nb]; // symmetric lengths
      for (local_int_t i = 0; i < num; i++) {
        global_int_t g = (*data->localToGlobalMap)[data->elementsToSend[index_elementsToSend++]];
        assert(ComputeRankOfMatrixRow(*data->geom, g) == data->geom->rank);
        laik_range_init_1d(&range, x_space, g, g + 1);
        laik_append_range(r, data->neighbors[nb], &range, 1, 0);
      }
    }
  }
}

void partitioner_alg_for_rows(Laik_RangeReceiver* r, Laik_PartitionerParams* p)
{
  partition_d* data = (partition_d*)laik_partitioner_data(p->partitioner);
  Laik_Space* space = p->space;
  int64_t spaceSize = (int64_t)laik_space_size(space);
  const bool is_prefix = (spaceSize == (int64_t)(data->size + 1));
  Laik_Range range;

  // Create one map per contiguous segment in global row order
  int prev_owner = -1;
  global_int_t seg_start = 0;
  int current_tag = 0;
  int next_tag = 1;

  // Assign rows [0..size) and, if prefix space, the extra boundary per segment
  for (global_int_t i = 0; i < data->size; i++) {
    int owner = ComputeRankOfMatrixRow(*data->geom, i);
    if (owner < 0 || owner >= data->geom->size) {
      fprintf(stderr, "LAIK rows partitioner: invalid owner %d for row %lld (size %d)\n",
              owner, (long long)i, data->geom->size);
      owner = data->geom->rank;
    }
    if (prev_owner == -1) {
      prev_owner = owner;
      seg_start = i;
      current_tag = 0;
      next_tag = 1;
    }
    if (owner != prev_owner) {
      global_int_t seg_end = i;
      global_int_t range_end = is_prefix ? (seg_end + 1) : seg_end;
      laik_range_init_1d(&range, space, seg_start, range_end);
      laik_append_range(r, prev_owner, &range, current_tag, 0);
      seg_start = i;
      prev_owner = owner;
      current_tag = next_tag++;
    }
  }
  if (prev_owner != -1) {
    global_int_t seg_end = data->size;
    global_int_t range_end = is_prefix ? (seg_end + 1) : seg_end;
    laik_range_init_1d(&range, space, seg_start, range_end);
    laik_append_range(r, prev_owner, &range, current_tag, 0);
  }
}

Laik_Blob* init_blob(const SparseMatrix& A, bool exchangesValues, const char* name, Laik_Partitioning* localP, Laik_Partitioning* extP)
{
  static_assert(sizeof(global_int_t) == sizeof(int64_t), "global_int_t must be 64-bit for LAIK vector layout");
  Laik_Blob* blob = (Laik_Blob*)malloc(sizeof(Laik_Blob));
  blob->values = laik_new_data(A.space, laik_Double);
  blob->localLength = A.localNumberOfRows;
  blob->exchangesValues = exchangesValues;
  blob->name = name;
  blob->reservation = 0;
  laik_data_set_name(blob->values, const_cast<char*>(name));

  // Vector layout with stable external mapping
  Laik_Data_Parameters* params = (Laik_Data_Parameters*)malloc(sizeof(*params));
  params->prefix_row_data = 0;
  params->vector_local_indices = reinterpret_cast<const int64_t*>(A.localToGlobalMap.data());
  params->vector_local_count = (uint64_t)A.localNumberOfRows;
  params->vector_external_indices = exchangesValues ? reinterpret_cast<const int64_t*>(A.externalLocalToGlobal.data()) : 0;
  params->vector_external_count = exchangesValues ? (uint64_t)A.numberOfExternalValues : 0;
  laik_data_attach_params(blob->values, params);
  laik_data_set_layout_factory(blob->values, laik_new_layout_vector);

  // Initialize with external partitioning to pre-allocate, then switch to local
  if (exchangesValues && extP)
    laik_switchto_partitioning(blob->values, extP, LAIK_DF_None, LAIK_RO_None);
  if (localP)
    laik_switchto_partitioning(blob->values, localP, LAIK_DF_None, LAIK_RO_None);

  return blob;
}

Laik_Blob* init_blob(const SparseMatrix& A, bool exchangesValues, const char* name)
{
  return init_blob(A, exchangesValues, name, A.local, A.ext);
}

void init_partition_data(SparseMatrix& A, partition_d* local, partition_d* ext)
{
  ext->size = A.totalNumberOfRows;
  ext->geom = A.geom;
  ext->localToGlobalMap = &A.localToGlobalMap;
  ext->externalLocalToGlobal = A.externalLocalToGlobal.data();
  ext->numberOfExternalValues = A.numberOfExternalValues;
  ext->halo = true;
  ext->receiveList.clear();

  // Build receiveList (owner -> needed globals)
  for (local_int_t i = 0; i < ext->numberOfExternalValues; ++i) {
    global_int_t g = ext->externalLocalToGlobal[i];
    int owner = ComputeRankOfMatrixRow(*ext->geom, g);
    ext->receiveList[owner].insert(g);
  }

  // Use halo metadata even when MPI is disabled, since LAIK still needs it.
  ext->numberOfNeighbours = A.numberOfSendNeighbors;
  ext->neighbors = A.neighbors;
  ext->elementsToSend = A.elementsToSend;
  ext->receiveLength = A.receiveLength;

  const char* dbg = std::getenv("HPCG_LAIK_EXT_DEBUG");
  if (dbg && dbg[0] != '\0') {
    fprintf(stderr, "[rank %d] ext nExt=%d nNbr=%d\n",
            ext->geom->rank, (int)ext->numberOfExternalValues, ext->numberOfNeighbours);
    fprintf(stderr, "[rank %d] ext externalLocalToGlobal:", ext->geom->rank);
    for (local_int_t i = 0; i < ext->numberOfExternalValues; ++i) {
      fprintf(stderr, " %lld", (long long)ext->externalLocalToGlobal[i]);
    }
    fprintf(stderr, "\n");

    for (auto it = ext->receiveList.begin(); it != ext->receiveList.end(); ++it) {
      fprintf(stderr, "[rank %d] ext receiveList from %d:", ext->geom->rank, it->first);
      for (auto sit = it->second.begin(); sit != it->second.end(); ++sit) {
        fprintf(stderr, " %lld", (long long)*sit);
      }
      fprintf(stderr, "\n");
    }

    if (ext->numberOfNeighbours > 0 && ext->neighbors && ext->receiveLength && ext->elementsToSend) {
      local_int_t sendIdx = 0;
      for (int nb = 0; nb < ext->numberOfNeighbours; ++nb) {
        fprintf(stderr, "[rank %d] ext neighbor %d recvLen=%d sendIdxStart=%d\n",
                ext->geom->rank, ext->neighbors[nb], (int)ext->receiveLength[nb], (int)sendIdx);
        for (local_int_t i = 0; i < ext->receiveLength[nb]; ++i) {
          fprintf(stderr, "[rank %d] ext elementsToSend[%d]=%d\n",
                  ext->geom->rank, (int)sendIdx, (int)ext->elementsToSend[sendIdx]);
          ++sendIdx;
        }
      }
    }
  }

  local->size = ext->size;
  local->geom = ext->geom;
  local->halo = false;
  local->numberOfNeighbours = 0;
  local->neighbors = NULL;
  local->localToGlobalMap = NULL;
  local->elementsToSend = NULL;
  local->receiveLength = NULL;
  local->externalLocalToGlobal = NULL;
  local->numberOfExternalValues = 0;
}

void init_partitionings(SparseMatrix& A, partition_d* local, partition_d* ext)
{
  init_partitionings(A, local, ext, world);
}

void init_partitionings(SparseMatrix& A, partition_d* local, partition_d* ext, Laik_Group* world)
{
  Laik_Partitioner* x_localPR = laik_new_partitioner(
      "x_localPR", partitioner_alg_for_x_vector, (void*)local, LAIK_PF_None);
  A.local = laik_new_partitioning(x_localPR, world, A.space, NULL);
  if (laik_size(world) == 1) {
    A.ext = A.local;
    return;
  }

  Laik_Partitioner* x_extPR = laik_new_partitioner(
      "x_extPR", partitioner_alg_for_x_vector, (void*)ext, LAIK_PF_None);
  A.ext = laik_new_partitioning(x_extPR, world, A.space, NULL);
}

void ZeroLaikVector(Laik_Blob* x)
{
  assert(x);

  double* base;
  uint64_t count;

  laik_get_map_1d(x->values, 0, (void**)&base, &count);

  for (uint64_t i = 0; i < x->localLength; i++)
    base[i] = 0.0;
}

void ScaleLaikVectorValue(Laik_Blob* v, local_int_t index, double value)
{
  assert(v);
  assert(index >= 0 && index < v->localLength);

  double* vv;
  laik_get_map_1d(v->values, 0, (void**)&vv, 0);
  vv[index] *= value;
}

void CopyLaikVectorToLaikVector(Laik_Blob* x, Laik_Blob* y)
{
  assert(x->localLength == y->localLength);

  double* xv;
  double* yv;

  laik_get_map_1d(x->values, 0, (void**)&xv, 0);
  laik_get_map_1d(y->values, 0, (void**)&yv, 0);

  for (uint64_t i = 0; i < x->localLength; i++)
    yv[i] = xv[i];
}

void fillRandomLaikVector(Laik_Blob* x)
{
  assert(x);

  double* xv;
  uint64_t count;
  laik_get_map_1d(x->values, 0, (void**)&xv, &count);
  for (uint64_t i = 0; i < x->localLength; i++)
    xv[i] = i + 1.0;
}

void CopyVectorToLaikVector(Vector& v, Laik_Blob* x)
{
  assert(v.localLength >= x->localLength);

  double* xv;
  uint64_t count;
  laik_get_map_1d(x->values, 0, (void**)&xv, &count);

  const double* vv = v.values;

  for (uint64_t i = 0; i < x->localLength; i++)
    xv[i] = vv[i];
}

void CopyLaikVectorToVector(const Laik_Blob* x, Vector& v)
{
  assert(x->localLength == v.localLength);

  double* xv;
  uint64_t count;
  laik_get_map_1d(x->values, 0, (void**)&xv, &count);

  double* vv = v.values;

  for (uint64_t i = 0; i < x->localLength; i++)
    vv[i] = xv[i];
}

void CopyLaikVectorToVector(Laik_Blob* x, Vector& v)
{
  const Laik_Blob* x_const = x;
  CopyLaikVectorToVector(x_const, v);
}

void DeleteLaikVector(Laik_Blob* x)
{
  x->localLength = 0;
  if (x->reservation) {
    laik_reservation_free(x->reservation);
    x->reservation = 0;
  }
  laik_free(x->values);
  x->values = NULL;
}