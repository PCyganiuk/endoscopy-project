#!/bin/bash

#bash run_all.sh

#base
nohup bash run_tests_fisheye_base.sh > master_run_fisheye_base.log 2>&1 &

#galar train
nohup bash run_tests_fisheye_galar.sh > master_run_fisheye_galar.log 2>&1 &

#mix
nohup bash run_tests_fisheye_mix.sh > master_run_fisheye_mix.log 2>&1 &

echo "all running!"