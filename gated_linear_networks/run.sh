#!/bin/sh
# Copyright 2020 DeepMind Technologies Limited.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
set -e

python3.6 -m venv gln_venv
source gln_venv/bin/activate
python3.6 -m pip install --upgrade pip
python3.6 -m pip install --upgrade setuptools wheel
python3.6 -m pip install  --upgrade jax jaxlib
python3.6 -m pip install  --upgrade tensorflow==2.3.1
python3.6 -m pip install -r gated_linear_networks/requirements.txt
python3.6 -m pip install --upgrade jax jaxlib==0.1.57+cuda102 -f https://storage.googleapis.com/jax-releases/jax_releases.html

# Run MNIST example with Bernoulli GLN
#export XLA_PYTHON_CLIENT_MEM_FRACTION=.50
#export XLA_PYTHON_CLIENT_ALLOCATOR=platform
#XLA_PYTHON_CLIENT_PREALLOCATE=false
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:/usr/local/cuda-10.2/lib64:/usr/local/cuda/extras/CUPTI/lib64:$LD_LIBRARY_PATH
python3.6 -m gated_linear_networks.examples.bernoulli_mnist \
  --num_layers=2 \
  --neurons_per_layer=100 \
  --context_dim=1
