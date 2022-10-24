#!/bin/sh

if [ "$USE_PATCHMATCH" != "1" ]; then
  echo "Skipping PyPatchMatch install because USE_PATCHMATCH=$USE_PATCHMATCH"
  mkdir PyPatchMatch
  touch PyPatchMatch/patch_match.py
  exit
fi

echo "Installing PyPatchMatch because USE_PATCHMATCH=$USE_PATCHMATCH"
apt-get install -yqq libopencv-dev python3-opencv > /dev/null
git clone https://github.com/lkwq007/PyPatchMatch
cd PyPatchMatch
git checkout 0ae9b8bbdc83f84214405376f13a2056568897fb
sed -i '0,/if os.name!="nt":/s//if False:/' patch_match.py
make
