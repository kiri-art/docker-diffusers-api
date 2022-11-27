#!/bin/sh

# This entire file is no longer used but kept around for reference.

if [ "$FLASH_ATTENTION" == "1" ]; then

  echo "Building with flash attention"
  git clone https://github.com/HazyResearch/flash-attention.git
  cd flash-attention
  git checkout cutlass
  git submodule init
  git submodule update
  python setup.py install

  cd ..
  git clone https://github.com/HazyResearch/diffusers.git
  pip install -e diffusers

else

  echo "Building without flash attention"
  git clone https://github.com/huggingface/diffusers
  cd diffusers
  git checkout v0.9.0
  # 2022-11-21 [Community Pipelines] K-Diffusion Pipeline 
  # git checkout 182eb959e5efc8c77fa31394ca55376331c0ed25
  # 2022-11-24 v_prediction (for SD 2.0)
  # git checkout 30f6f4410487b6c1cf5be2da6c7e8fc844fb9a44
  cd ..
  pip install -e diffusers

fi

