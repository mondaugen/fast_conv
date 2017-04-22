#!/bin/bash
seq 1 10 | awk '{ print 200 " " 4096 " " 2**$0 }' | \
    valgrind --tool=callgrind --instr-atstart=no \
    ./part_conv.bin
#find . -name "callgrind.out*" | awk 'BEGIN {FS="."} { print $5 }'
find -E . -regex "\./callgrind\.out\.[0-9]+\.[0-9]+" | awk 'BEGIN {FS="."} { print "echo \"" $5 " " "$( callgrind_annotate " $0 " | grep TOTALS | tr -d \",\" )\"" }' | bash | awk '{ print $1 " " $2 }'

