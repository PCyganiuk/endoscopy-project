
#!/bin/bash

# How to run:
# chmod +x run_tests.sh <- run this
# nohup bash run_tests.sh > master_run_fisheye.log 2>&1 & <- then this

# ps aux | grep python <- check if still running

# tail -f master_run.log <- get the output to console

# kill <PID> <- kill if nessesary

# Create directory for logs
mkdir -p logs

# Loop over type-num and model-size combinations
for type in {0..2}; do 
  for size in {0..0}; do
    timestamp=$(date +"%Y-%m-%d_%H-%M-%S")
    log_file="logs/test_type${type}_size${size}_${timestamp}_fisheye.log"

    echo "Starting training: type-num=${type}, model-size=${size}"
    echo "Logging to ${log_file}"
    export CUDA_VISIBLE_DEVICES=$((type + 3))
    #export CUDA_VISIBLE_DEVICES="0,1,2,3"
    nohup python3 main.py \
      --ers-path /local_storage/gwo/public/gastro/galar/ers_jpg/ \
      --galar-path /local_storage/gwo/public/gastro/galar/galar_jpg/ \
      --type-num "${type}" \
      --epochs 15 \
      --k-folds 20 \
      --model-size "${size}" \
      --binary 1 \
      --verbose 1 \
      --fisheye 1 \
      > "${log_file}" 2>&1 &

    echo "Finished training: type-num=${type}, model-size=${size}"
    echo "---------------------------------------------"
    sleep 1200
  done
done

echo "All trainings completed successfully!"

# nohup python3 main.py --ers-path /mnt/d/ERS/ers_jpg/ --galar-path /mnt/e/galar_jpg/ --type-num 0 --epochs 1 --k-folds 2 --model-size 0 --binary 1 --verbose 2 --fisheye 1 > logs/baseline.log 2>&1 &
