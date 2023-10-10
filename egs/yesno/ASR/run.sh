#!/usr/bin/env bash

set -eou pipefail

stage=-1
stop_stage=100
. path.sh
. utils/parse_options.sh || exit 1

log() {
  # This function is from espnet
  local fname=${BASH_SOURCE[1]##*/}
  echo -e "$(date '+%Y-%m-%d %H:%M:%S') (${fname}:${BASH_LINENO[0]}:${FUNCNAME[1]}) $*"
}


if [ ${stage} -le 0 ] && [ ${stop_stage} -ge 0 ];then
   log "prepred data"
      
   ./prepare.sh --stage 2 --stop-stage 5
fi

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ];then
  log "training transducer network for yesno dataset"
  CUDA_VISIBLE_DEVICES=4 ./transducer/train.py \
     --world-size 1
fi

if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ];then
   log "decoding yesno testset"
   ./transducer/decode.py 
      
fi
