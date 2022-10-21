#!/usr/bin/env bash

# Run this in banana-sd-base's PARENT directory.
# Modify the below first per your preferences

# Requires `yq` from https://github.com/mikefarah/yq
# Note, there are two yqs.  In Archlinux the package is "go-yq".

if [ -z "$1" ]; then 
  echo "Using 'scripts/permutations.yaml' as default INFILE"
  echo "You can also run: permutate.sh MY_INFILE"
  INFILE='scripts/permutations.yaml'
else
  INFILE=$1
fi

if [ -z "$TARGET_REPO_BASE" ]; then
  TARGET_REPO_BASE="git@github.com:kiri-art"
  echo 'No TARGET_REPO_BASE found, using "$TARGET_REPO_BASE"'
fi

permutations=$(yq e -o=j -I=0 '.list[]' $INFILE)

# Needed for ! expansion in cp command further down.
shopt -s extglob
# Include dot files in expansion for .git .gitignore
shopt -s dotglob

COUNTER=0
declare -A vars

mkdir -p permutations

while IFS="=" read permutation; do
  # e.g. Permutation #1: banana-sd-txt2img
  NAME=$(echo "$permutation" | yq e '.name')
  COUNTER=$[$COUNTER + 1]
  echo
  echo "Permutation #$COUNTER: $NAME"

  while IFS="=" read -r key value
  do
    if [ "$key" != "name" ]; then
      if [ "${value:0:1}" == "$" ]; then
        # For e.g. "$HF_AUTH_TOKEN", expand from environment
        value="${value:1}"
        vars[$key]=${!value}
      else
        vars[$key]=$value;
      fi
    fi
  done < <(echo $permutation | yq e 'to_entries | .[] | (.key + "=" + .value)')

  if [ -d "permutations/$NAME" ]; then 
    echo "./permutations/$NAME already exists, skipping..."
    echo "Run 'rm -rf permutations/$NAME' first to remake this permutation"
    echo "In a later release, we'll merge updates in this case."
    continue
  fi

  # echo "mkdir permutations/$NAME"
  mkdir permutations/$NAME
  # echo 'cp -a ./!(permutations|scripts|root-cache) permutations/$NAME'
  cp -a ./!(permutations|scripts|root-cache) permutations/$NAME
  # echo cd permutations/$NAME
  cd permutations/$NAME

  echo "Substituting variables in Dockerfile"
  for key in "${!vars[@]}"; do
    value="${vars[$key]}"
    sed -i "s@^ARG $key.*\$@ARG $key=\"$value\"@" Dockerfile
  done

  diffusers=${vars[diffusers]}
  if [ "$diffusers" ]; then
    echo "Replacing diffusers with $diffusers"
    echo "!!! NOT DONE YET !!!"
  fi

  mkdir root-cache
  touch root-cache/non-empty-directory
  git add root-cache

  git remote rm origin
  git remote add upstream git@github.com:kiri-art/docker-diffusers-api.git
  git remote add origin $TARGET_REPO_BASE/$NAME.git

  echo git commit -a -m "$NAME permutation variables"
  git commit -a -m "$NAME permutation variables"

  # echo "cd ../.."
  cd ../..
  echo
done <<EOF
$permutations
EOF
