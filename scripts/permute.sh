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

  for file in DOWNLOAD_VARS.py APP_VARS.py ; do
    echo "Substiting variables in $file"
    for key in "${!vars[@]}"
    do
      value="${vars[$key]}"
      # echo "key $key value $value"
      # echo sed -i "s@^$key = .*\$@$key = \"$value\"@" $file
      sed -i "s@^$key = .*\$@$key = \"$value\"@" $file
    done
  done

  # Hopefully soon we'll get build vars through Banana
  echo "Substituting variables HF_AUTH_TOKEN in Dockerfile"
  sed -i 's/ARG HF_AUTH_TOKEN/# ARG HF_AUTH_TOKEN/' Dockerfile
  sed -i "s/ENV HF_AUTH_TOKEN=\${HF_AUTH_TOKEN}/ENV HF_AUTH_TOKEN=\"${HF_AUTH_TOKEN}\"/" Dockerfile

  diffusers=${vars[diffusers]}
  if [ "$diffusers" ]; then
    echo "Replacing diffusers with $diffusers"
    echo "!!! NOT DONE YET !!!"
  fi

  git remote rm origin
  git remote add upstream git@github.com:kiri-art/docker-diffusers-api.git
  git remote add origin git@github.com:gadicc/$NAME.git

  echo git commit -a -m "$NAME variables"
  git commit -a -m "$NAME variables"

  # echo "cd ../.."
  cd ../..
  echo
done <<EOF
$permutations
EOF
