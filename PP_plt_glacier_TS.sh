#!/usr/bin/env bash

echo "Starting job with the following parameters:"
echo "$@"

echo $SLURM_CPUS_ON_NODE

export PATH="<PATH_TO_ANACONDA>/anaconda3/bin:$PATH"
python -u "$@"

