#!/bin/bash
seq 1 10 | awk '{ print 200 " " 4096 " " 2**$0 }' | \
    valgrind --tool=callgrind --instr-atstart=no \
    ./part_conv.bin
find . -name "callgrind.out*" | awk 'BEGIN {FS="."} { print $5 }'

