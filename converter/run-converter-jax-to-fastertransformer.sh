
#!/bin/bash

# Copyright 2023 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Usage
# 
# run-converter-jax-to-fastertransformer.sh \
#   model_name
#   gcs_jax_checkpoint_location
#   gcs_ft_checkpoint_location
#   tensor-parallelism (1,2,4,8)

# Set up a global error handler
err_handler() {
    echo "Error on line: $1"
    echo "Caused by: $2"
    echo "That returned exit status: $3"
    echo "Aborting..."
    exit $3
}

trap 'err_handler "$LINENO" "$BASH_COMMAND" "$?"' ERR

TIMESTAMP=`date "+%Y-%m-%d %H:%M:%S"`

echo "NVIDIA Driver version"
nvidia-smi

# Set variables
MODEL_NAME=$1
echo "MODEL_NAME = ${MODEL_NAME}"

GCS_JAX_CHECKPOINT=$2
echo "GCS_JAX_CHECKPOINT = ${GCS_JAX_CHECKPOINT}"

GCS_FT_CHECKPOINT=$3
echo "GCS_FT_CHECKPOINT = ${GCS_FT_CHECKPOINT}"

if [[ -z $4 ]];
then 
    TENSOR_PARALLELISM=1
else
    TENSOR_PARALLELISM=$4
fi
echo "TENSOR_PARALLELISM = ${TENSOR_PARALLELISM}"

# Copy JAX checkpoint to local directory
LOCAL_JAX_CHECKPOINT="/models/"$(basename $GCS_JAX_CHECKPOINT)
mkdir -p $LOCAL_JAX_CHECKPOINT
echo "[INFO] ${TIMESTAMP} Copying JAX checkpoint from ${GCS_JAX_CHECKPOINT} to local ${LOCAL_JAX_CHECKPOINT}"
SECONDS=0
gcloud storage cp --quiet --recursive $GCS_JAX_CHECKPOINT /models/
echo "[INFO] Completed copying JAX checkpoint locally in ${SECONDS}s"

# Creating local directories for saving FasterTransformer model
LOCAL_FT_CHECKPOINT="/models/ft"
mkdir -p $LOCAL_FT_CHECKPOINT

# Run JAX to FasterTransformer 
echo "[INFO] ${TIMESTAMP} Converting JAX checkpoint to FasterTransformer"
SECONDS=0
python3 /FasterTransformer/examples/tensorflow/t5/utils/jax_t5_ckpt_convert.py \
   $LOCAL_JAX_CHECKPOINT \
   $LOCAL_FT_CHECKPOINT \
   --tensor-parallelism $TENSOR_PARALLELISM
echo "[INFO] ${TIMESTAMP} Completed converting JAX checkpoint to FasterTransformer in ${SECONDS}s"

# Organize model repository for Triton serving
echo "[INFO] ${TIMESTAMP} Organizing model repository for serving"
cd $LOCAL_FT_CHECKPOINT
mkdir -p $LOCAL_FT_CHECKPOINT/${MODEL_NAME}/1
mv $LOCAL_FT_CHECKPOINT/${TENSOR_PARALLELISM}-gpu $LOCAL_FT_CHECKPOINT/${MODEL_NAME}/1/

# Format Triton config
cp /triton/config.pbtxt $LOCAL_FT_CHECKPOINT/${MODEL_NAME}/config.pbtxt
sed -i -e 's!@@MODEL_NAME@@!'$MODEL_NAME'!g' $LOCAL_FT_CHECKPOINT/${MODEL_NAME}/config.pbtxt 
sed -i -e 's!@@TENSOR_PARA_SIZE@@!'$TENSOR_PARALLELISM'!g' $LOCAL_FT_CHECKPOINT/${MODEL_NAME}/config.pbtxt 
sed -i -e 's!@@PIPELINE_PARA_SIZE@@!'$TENSOR_PARALLELISM'!g;' $LOCAL_FT_CHECKPOINT/${MODEL_NAME}/config.pbtxt 

# Uploaded FasterTransformer checkpoint to Cloud Storage bucket
echo "[INFO] ${TIMESTAMP} Copying FasterTransformer model from local ${LOCAL_FT_CHECKPOINT} to ${GCS_FT_CHECKPOINT}"
SECONDS=0
gcloud storage cp --recursive $LOCAL_FT_CHECKPOINT $GCS_FT_CHECKPOINT
echo "[INFO] Completed copying FasterTransformer model to Cloud Storage in ${SECONDS}s"