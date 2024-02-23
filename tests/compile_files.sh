#!/bin/bash
echo "Compiling all .hvm2 files at" $@
find $@ | awk '$1 ~ /\.hvm2$/ {print $1}' | sed 's/.hvm2$/.hvm/g' |  while IFS= read -r file; do
    # Do something with each file, for example, print its name
	echo ">" hvml compile "$file"2 ">" "$file"c
    hvml compile "$file"2 > "$file"c
	
done