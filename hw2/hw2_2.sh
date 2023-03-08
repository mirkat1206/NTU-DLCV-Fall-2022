#!/bin/bash

# TODO - run your inference Python3 code
for i in 0 50
do
    python3 hw2_2_test.py $1 $i
done

wait
