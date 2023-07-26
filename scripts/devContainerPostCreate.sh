#!/bin/bash

# devcontainer.json postCreateCommand

echo
echo Initialize conda bindings for bash
conda init bash

echo Activating
source /opt/conda/bin/activate base

echo Installing dev dependencies
pip install watchdog
