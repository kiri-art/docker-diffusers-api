#!/bin/bash

source /opt/conda/bin/activate base

ln -sf /api/diffusers .

watchmedo auto-restart --recursive -d api python api/server.py