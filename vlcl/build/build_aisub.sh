#!/bin/bash
set -e
BASE="/c/RAHUL/PROJECTS _OF_Rahul/Offline Subtitles"
cd "$BASE"
echo "=== Building aisub VLC plugin ==="
/mingw64/bin/gcc -shared \
  -o vlcl/build/libaisub_plugin.dll \
  -I vlcl/build \
  -I vlcl/vlc/include \
  -DHAVE_CONFIG_H \
  -D__PLUGIN__ \
  -DMODULE_STRING='"aisub"' \
  -D_WIN32_WINNT=0x0601 \
  -DWIN32_LEAN_AND_MEAN \
  -O2 -w \
  vlcl/vlc/modules/spu/aisub.c \
  -L vlcl \
  -lvlccore \
  -Wl,--no-undefined
echo "=== BUILD SUCCESS ==="
ls -la vlcl/build/libaisub_plugin.dll
