#!/bin/bash

set -e

export PKG_CONFIG_PATH="${HOME}/work/ai/whisper.cpp/build-install/lib/pkgconfig${PKG_CONFIG_PATH:+:$PKG_CONFIG_PATH}"

./configure --prefix=/usr --enable-version3 --disable-shared --enable-gpl \
  --enable-nonfree --enable-static --enable-pthreads --enable-filters \
  --enable-openssl --enable-runtime-cpudetect --enable-libvpx --enable-libx264 \
  --enable-libx265 --enable-libspeex --enable-libfreetype --enable-fontconfig \
  --enable-libzimg --enable-libvorbis --enable-libwebp --enable-libfribidi \
  --enable-libharfbuzz --enable-libass --enable-whisper --enable-parakeet
make

# Don't forget to the library path
# export LD_LIBRARY_PATH=${HOME}/work/ai/whisper.cpp/build-install/lib/:$LD_LIBRARY_PATH
# export DYLD_LIBRARY_PATH=${HOME}/work/ai/whisper.cpp/build-install/lib/:$LD_LIBRARY_PATH
