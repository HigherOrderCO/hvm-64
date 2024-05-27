#!/usr/bin/env bash

echo "Compiling all .hvm2 files"

shopt -s globstar

for hvml in **/*.hvm2; do
  hvm64="${hvml%.hvm2}.hvm"
  echo "> $hvml"
  hvml compile "$hvml" > "$hvm64"
done
