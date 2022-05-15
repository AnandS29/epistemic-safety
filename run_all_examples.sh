#!/bin/sh

for i in 0 ... 24
do
    echo "Running example $i"
    python3 run_example.py -e $i > logs/log_$i.txt 2>&1 &
done