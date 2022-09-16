#!/usr/bin/env bash

# Run this in banana-sd-base's PARENT directory.
# Modify the below first per your preferences

# Requires `yq` from https://github.com/mikefarah/yq
# Note, there are two yqs.  In Archlinux the package is "go-yq".

if [ -z "$1" ]; then 
  INFILE='scripts/permutations.yaml'
else 
  INFILE=$1
fi

permutations=$(yq e -o=j -I=0 '.list[]' $INFILE)

# Needed for ! expansion in cp command further down.
shopt -s extglob

COUNTER=0
declare -A vars

echo "rm -rf permutations"
rm -rf permutations
echo "mkdir permutations"
mkdir permutations

while IFS="=" read permutation; do
  # e.g. Permutation #1: banana-sd-txt2img
  NAME=$(echo "$permutation" | yq e '.name')
  COUNTER=$[$COUNTER + 1]
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

  # echo "mkdir permutations/$NAME"
  mkdir permutations/$NAME
  # echo 'cp -a ./!(permutations) permutations/$NAME'
  cp -a ./!(permutations) permutations/$NAME
  # echo cd permutations/$NAME
  cd permutations/$NAME

  for file in DOWNLOAD_VARS.py APP_VARS.py ; do
    echo "Substiting variables in $file"
    for key in "${!vars[@]}"
    do
      value="${vars[$key]}"
      # echo "key $key value $value"
      echo sed -i "s@^$key = .*\$@$key = \"$value\"@" $file
      sed -i "s@^$key = .*\$@$key = \"$value\"@" $file
    done
  done

  diffusers=${vars[diffusers]}
  if [ "$diffusers" ]; then
    echo "Replacing diffusers with $diffusers"
    echo "!!! NOT DONE YET !!!"
  fi

  git remote rm origin
  git remote add upstream git@github.com/gadicc/banana-sd-base.git
  rm remote add origin git@github.com:gadicc/$NAME.git

  echo git commit -a -m "$NAME variables"
  git commit -a -m "$NAME variables"

  # echo "cd ../.."
  cd ../..
  echo
done <<EOF
$permutations
EOF
