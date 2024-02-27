#!/usr/bin/env bash

echo "Compiling all .hvm2 files"

shopt -s globstar

for hvml in **/*.hvm2; do
  hvmc="${hvml%.hvm2}.hvmc"
  echo "> $hvml"
  hvml compile "$hvml" > "$hvmc"
done
