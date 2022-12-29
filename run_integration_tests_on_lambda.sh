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

cat > $PAYLOAD_FILE << __END__
{
  "region_name": "us-east-1",
  "instance_type_name": "gpu_1x_a100_sxm4",
  "ssh_key_names": [
    "Gadi Default"
  ],
  "file_system_names": [],
  "quantity": 1
}
__END__

RESULT=$(
  curl -u ${LAMBDA_API_KEY}: \
    https://cloud.lambdalabs.com/api/v1/instance-operations/launch \
    -d @${PAYLOAD_FILE} -H "Content-Type: application/json"
)
if [ $? -eq 1 ]; then
  echo "curl failed"
  exit 1
fi

echo $RESULT | jq -e .error
if [ $? -eq 0 ]; then
  echo "lambda error"
  exit 1
fi

echo $RESULT
INSTANCE_ID=$(echo $RESULT | jq -re '.data.instance_ids[0]')
if [ $? -eq 1 ]; then
  echo "jq failed"
  exit 1
fi

echo INSTANCE_ID=$INSTANCE_ID

echo "Terminating instance $INSTANCE_ID"
cat > $PAYLOAD_FILE << __END__
{
  "instance_ids": [
    "$INSTANCE_ID"
  ]
}
__END__

RESULT=$(
  curl -u ${LAMBDA_API_KEY}: \
    https://cloud.lambdalabs.com/api/v1/instance-operations/terminate \
    -d @${PAYLOAD_FILE} -H "Content-Type: application/json"
)
if [ $? -eq 1 ]; then
  echo "curl failed"
  exit 1
fi

echo $RESULT
echo $RESULT | jq -e .error
if [ $? -eq 0 ]; then
  echo "lambda error"
  exit 1
fi
