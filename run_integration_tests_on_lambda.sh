#!/bin/sh

PAYLOAD_FILE="/tmp/request.json"

if [ -z "$LAMBDA_API_KEY" ]; then
  echo "No LAMBDA_API_KEY set"
  exit 1
fi 

SSH_KEY_FILE="$HOME/.ssh/id_rsa.pub"
if [ ! -f "$SSH_KEY_FILE" ]; then
  echo "No ssh key"
  exit 1
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
  local RESULT=""
  cat > $PAYLOAD_FILE << __END__
  {
    "region_name": "us-west-2",
    "instance_type_name": "gpu_1x_a100_sxm4",
    "ssh_key_names": [
      "Gadi Default"
    ],
    "file_system_names": [],
    "quantity": 1
  }
__END__

  lambda_run "instance-operations/launch" $PAYLOAD_FILE
  echo $RESULT
  INSTANCE_ID=$(echo $RESULT | jq -re '.data.instance_ids[0]')
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

instance_wait() {
  # $1 = INSTANCE_ID
  echo -n "Waiting for $1"
  STATUS=""
  LAST_STATUS=""
  while [ "$STATUS" != "active" ] ; do
    echo -n "."
    lambda_run "instances/$1"
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
}

instance_create
echo INSTANCE_ID $INSTANCE_ID 
instance_wait $INSTANCE_ID
instance_terminate $INSTANCE_ID
