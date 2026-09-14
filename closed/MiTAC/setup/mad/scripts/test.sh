#!/bin/bash

export MAD_SECRETS_HFTOKEN=<your_hf_token_goes_here>
export MAD_SYSTEM_GPU_ARCHITECTURE=$(rocminfo | grep "Name:" | grep "gfx" | awk 'NR==1' | awk '{print $2}')
export MAD_DATAHOME=<local_model_path_goes_here>

if [ -z "$1" ]; then
  echo "Usage: $0 <script_to_run>"
  exit 1
fi

bash $1
