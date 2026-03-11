#!/usr/bin/env bash
set -euo pipefail

PROCS=${PROCS:-2}
export LAIK_BACKEND=${LAIK_BACKEND:-tcp2}
export LAIK_SIZE=${LAIK_SIZE:-${PROCS}}

# build LAIK first
cd /home/dbudak/laik
make

# build HPCG with LAIK setup
cd /home/dbudak/hpcg
make arch=Linux_LAIK

# run (no MPI)
cd /home/dbudak/hpcg/bin
rm -f hpcg*.txt

# ./xhpcg
