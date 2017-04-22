#!/bin/bash
seq 0 11 | awk '{ print 512 " " 8192 " " 2**$0 }' | \
    valgrind -q --tool=callgrind --instr-atstart=no \
    ./part_conv.bin

find -E . -regex "\./callgrind\.out\.[0-9]+\.[0-9]+" | \
    awk 'BEGIN {FS="."} { print "echo \"" $5 " " "$( callgrind_annotate " $0 " | grep TOTALS | tr -d \",\" )\"" }' | \
    bash | awk '{ printf("%2d 10^%f\n", $1,log($2)/log(10)) }'

find -E . -regex "\./callgrind\.out\.[0-9]+\.[0-9]+" -delete

find -E . -regex "\./callgrind\.out\.[0-9]+" -delete
