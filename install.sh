#!/bin/sh

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
  # git checkout v0.7.0
  # 2022-11-21 [Community Pipelines] K-Diffusion Pipeline 
  git checkout 182eb959e5efc8c77fa31394ca55376331c0ed25
  cd ..
  pip install -e diffusers

fi

