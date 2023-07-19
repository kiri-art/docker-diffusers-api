#!/bin/sh

rsync -avzPe "ssh -p $1" api/ $2:/api/
