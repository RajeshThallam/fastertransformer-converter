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

FROM nvcr.io/nvidia/pytorch:22.07-py3

# Install gcloud SDK
RUN apt-get install apt-transport-https ca-certificates gnupg
RUN echo "deb [signed-by=/usr/share/keyrings/cloud.google.gpg] http://packages.cloud.google.com/apt cloud-sdk main" | tee -a /etc/apt/sources.list.d/google-cloud-sdk.list && curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | apt-key --keyring /usr/share/keyrings/cloud.google.gpg  add - && apt-get update -y && apt-get install google-cloud-cli -y

# Clone FasterTransformer repo
WORKDIR /
RUN git clone --branch=main https://github.com/NVIDIA/FasterTransformer.git

# Build FasterTransformer
# Specify SM version as 80 for A100 GPUs; and 70 for V100
WORKDIR /FasterTransformer
RUN mkdir build && \
    cd build && \
    cmake -DSM=80 -DCMAKE_BUILD_TYPE=Release -DBUILD_PYT=ON -DBUILD_MULTI_GPU=ON .. && \
    make -j12

# Install other required packages
WORKDIR /FasterTransformer
RUN pip install -r /FasterTransformer/examples/pytorch/t5/requirement.txt
RUN pip install transformers==4.20.1 zarr
RUN pip install --upgrade "jax[cuda]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html

# Copy conversion script
COPY run-converter-jax-to-fastertransformer.sh /run-converter-jax-to-fastertransformer.sh

# Copy Triton related config
RUN mkdir -p /triton
COPY config.pbtxt /triton/config.pbtxt
