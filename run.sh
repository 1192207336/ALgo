#!/bin/bash

# show ${RUNTIME_SCRIPT_DIR}
echo ${RUNTIME_SCRIPT_DIR}
# enter train workspace
cd ${RUNTIME_SCRIPT_DIR}

# write your code below
#python -u main.py

python -u main.py --mm_emb_id 81 82 83 84 86 --norm_first