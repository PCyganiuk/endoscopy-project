#!/bin/bash

#bash run_all.sh

#base
nohup bash run_tests_fisheye.sh > master_run_fisheye.log 2>&1 &

#galar train
nohup bash run_tests_fisheye.sh > master_run_fisheye.log 2>&1 &

#mix
nohup bash run_tests_fisheye.sh > master_run_fisheye.log 2>&1 &

echo "all running!"