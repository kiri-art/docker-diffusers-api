#!/bin/bash

PAYLOAD_FILE="/tmp/request.json"

if [ -z "$LAMBDA_API_KEY" ]; then
  echo "No LAMBDA_API_KEY set"
  exit 1
fi 

SSH_KEY_FILE="$HOME/.ssh/diffusers-api-test.pem"
if [ ! -f "$SSH_KEY_FILE" ]; then
  curl -L $DDA_TEST_PEM > $SSH_KEY_FILE
  chmod 600 $SSH_KEY_FILE
fi

#curl -u $LAMBDA_API_KEY: https://cloud.lambdalabs.com/api/v1/instances

# TODO, find an available instance
# https://cloud.lambdalabs.com/api/v1/instance-types

lambda_run() {
  # $1 = lambda instance-operation
  if [ -z "$2" ] ; then
    RESULT=$(
      curl -su ${LAMBDA_API_KEY}: \
        https://cloud.lambdalabs.com/api/v1/$1 \
        -H "Content-Type: application/json"
    )
  else
    RESULT=$(
      curl -su ${LAMBDA_API_KEY}: \
        https://cloud.lambdalabs.com/api/v1/$1 \
        -d @$2 -H "Content-Type: application/json"
    )
  fi

  if [ $? -eq 1 ]; then
    echo "curl failed"
    exit 1
  fi

  if [ "$RESULT" != "" ]; then
    echo $RESULT | jq -e .error >& /dev/null
    if [ $? -eq 0 ]; then
      echo "lambda error"
      echo $RESULT
      exit 1
    fi
  fi
}

instance_create() {
  echo -n "Creating instance..."
  local RESULT=""
  cat > $PAYLOAD_FILE << __END__
  {
    "region_name": "us-west-2",
    "instance_type_name": "gpu_1x_a100_sxm4",
    "ssh_key_names": [
      "diffusers-api-test"
    ],
    "file_system_names": [],
    "quantity": 1
  }
__END__

  lambda_run "instance-operations/launch" $PAYLOAD_FILE
  # echo $RESULT
  INSTANCE_ID=$(echo $RESULT | jq -re '.data.instance_ids[0]')
  echo "$INSTANCE_ID"
  if [ $? -eq 1 ]; then
    echo "jq failed"
    exit 1
  fi
}

instance_terminate() {
  # $1 = INSTANCE_ID
  echo "Terminating instance $1"
  cat > $PAYLOAD_FILE << __END__
  {
    "instance_ids": [
      "$1"
    ]
  }
__END__
  lambda_run "instance-operations/terminate" $PAYLOAD_FILE
  echo $RESULT
}

declare -A IPS
instance_wait() {
  INSTANCE_ID="$1"
  echo -n "Waiting for $INSTANCE_ID"
  STATUS=""
  LAST_STATUS=""
  while [ "$STATUS" != "active" ] ; do
    echo -n "."
    lambda_run "instances/$INSTANCE_ID"
    STATUS=$(echo $RESULT | jq -r '.data.status')
    if [ "$STATUS" != "$LAST_STATUS" ]; then
      # echo $RESULT
      # echo STATUS $STATUS
      LAST_STATUS=$STATUS
    fi
    sleep 1
  done
  echo

  IP=$(echo $RESULT | jq -r '.data.ip')
  echo STATUS $STATUS
  echo IP $IP
  IPS["$INSTANCE_ID"]=$IP
}

instance_run_script() {
  INSTANCE_ID="$1"
  SCRIPT="$2"
  DIRECTORY="${3:-'.'}"
  IP=${IPS["$INSTANCE_ID"]}

  echo "instance_run_script $1 $2 $3"
  ssh -i $SSH_KEY_FILE ubuntu@$IP "cd $DIRECTORY && bash -s" < $SCRIPT
  return $?
}

instance_run_command() {
  INSTANCE_ID="$1"
  CMD="$2"
  DIRECTORY="${3:-'.'}"
  IP=${IPS["$INSTANCE_ID"]}

  echo "instance_run_command $1 $2"
  ssh -i $SSH_KEY_FILE -o StrictHostKeyChecking=accept-new ubuntu@$IP "cd $DIRECTORY && $CMD"
  return $?
}

instance_rsync() {
  INSTANCE_ID="$1"
  SOURCE="$2"
  DEST="$3"
  IP=${IPS["$INSTANCE_ID"]}

  echo "instance_rsync $1 $2 $3"
  rsync -avzPe "ssh -i $SSH_KEY_FILE -o StrictHostKeyChecking=accept-new" --filter=':- .gitignore' --exclude=".git" $SOURCE ubuntu@$IP:$DEST
  return $?
}

# Image Method 3, preparation (TODO, arg to specify which method)
docker build -t gadicc/diffusers-api:test .
docker push gadicc/diffusers-api:test

instance_create
# INSTANCE_ID="913e06f669bf4e799c6223801eb82f40"

instance_wait $INSTANCE_ID

commands() {
  instance_run_command $INSTANCE_ID "echo 'export HF_AUTH_TOKEN=\"$HF_AUTH_TOKEN\"' >> ~/.bashrc"

  # Whether to build or just for test scripts, lets transfer this checkout.
  instance_rsync $INSTANCE_ID . docker-diffusers-api

  instance_run_command $INSTANCE_ID "sudo apt-get update"
  if [ $? -eq 1 ]; then return 1 ; fi
  instance_run_command $INSTANCE_ID "sudo apt install -yqq python3.9"
  if [ $? -eq 1 ]; then return 1 ; fi
  instance_run_command $INSTANCE_ID "python3.9 -m pip install -r docker-diffusers-api/tests/integration/requirements.txt"
  if [ $? -eq 1 ]; then return 1 ; fi
  instance_run_command $INSTANCE_ID "sudo usermod -aG docker ubuntu"
  if [ $? -eq 1 ]; then return 1 ; fi

  # Image Method 1: Transfer entire image
  # This turned out to be way too slow, quicker to rebuild on lambda
  # Longer term, I guess we need our own container registry.
  # echo "Saving and transferring docker image to Lambda..."
  # IP=${IPS["$INSTANCE_ID"]}
  # docker save gadicc/diffusers-api:latest \
  #   | xz \
  #   | pv \
  #   | ssh -i $SSH_KEY_FILE ubuntu@$IP docker load
  # if [ $? -eq 1 ]; then return 1 ; fi

  # Image Method 2: Build on LambdaLabs
  #if [ $? -eq 1 ]; then return 1 ; fi
  #instance_run_command $INSTANCE_ID "docker build -t gadicc/diffusers-api ." docker-diffusers-api

  # Image Method 3: Just upload new layers; Lambda has fast downloads from registry
  # At start of script we have docker build/push.  Now let's pull:
  instance_run_command $INSTANCE_ID "docker pull gadicc/diffusers-api:test"

  # instance_run_script $INSTANCE_ID run_integration_tests.sh docker-diffusers-api
  instance_run_command $INSTANCE_ID "export HF_AUTH_TOKEN=\"$HF_AUTH_TOKEN\" && python3.9 -m pytest -s tests/integration" docker-diffusers-api
}

commands
RETURN_VALUE=$?

instance_terminate $INSTANCE_ID

exit $RETURN_VALUE