#!/bin/bash

[ -n "$PARCS_ARGS" ] && ./generate_config.sh $PARCS_ARGS > parcs.config 

python3 start.py -config parcs.config
