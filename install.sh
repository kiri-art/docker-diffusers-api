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
  # .components, lpw fixes
  git checkout v0.5.1
  cd ..
  pip install -e diffusers

fi

