#!/bin/bash

set -e
set -x

export LD_LIBRARY_PATH=${HOME}/work/ai/whisper.cpp/build-install/lib/:$LD_LIBRARY_PATH
#export DYLD_LIBRARY_PATH=${HOME}/work/ai/whisper.cpp/build-install/lib/:$LD_LIBRARY_PATH

model="${HOME}/work/ai/whisper.cpp/models/ggml-parakeet-tdt-0.6b-v3-f32.bin"
ls ${model}
#sample="${HOME}/work/ai/whisper.cpp/samples/jfk.wav"
sample="${HOME}/work/ai/whisper.cpp/samples/gb1.wav"

echo "Running parakeet on ${sample} with model ${model}"

./ffmpeg -i ${sample} \
    -loglevel quiet \
    -af "parakeet=model=${model}:use_gpu=1:destination=-" \
    -f null -
