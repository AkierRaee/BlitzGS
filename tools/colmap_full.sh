# Copyright (C) 2024 Denso IT Laboratory, Inc.
# All Rights Reserved

COLMAP_RESULTS_DIR=$1
DATASET_ROOT=$2


bash tools/triangulate_colmap.sh $COLMAP_RESULTS_DIR $DATASET_ROOT/train
