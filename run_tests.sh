#!/bin/bash

# How to run:
# chmod +x run_tests.sh <- run this
# nohup ./run_tests.sh > master_run.log 2>&1 & <- then this

# ps aux | grep python <- check if still running

# tail -f master_run.log <- get the output to console

# kill <PID> <- kill if nessesary

# Create directory for logs
mkdir -p logs

# Loop over type-num and model-size combinations
for type in {0..3}; do
  for size in {0..0}; do
    timestamp=$(date +"%Y-%m-%d_%H-%M-%S")
    log_file="logs/baseline_type${type}_size${size}_${timestamp}.log"

    echo "Starting training: type-num=${type}, model-size=${size}"
    echo "Logging to ${log_file}"

    nohup python3 main.py \
      --ers-path /local_storage/gwo/public/gastro/ers/ \
      --galar-path /local_storage/gwo/public/gastro/galar/galar_jpg/ \
      --type-num "${type}" \
      --epochs 10 \
      --k-folds 20 \
      --model-size "${size}" \
      --binary 1 \
      --verbose 2 \
      > "${log_file}" 2>&1

    echo "Finished training: type-num=${type}, model-size=${size}"
    echo "---------------------------------------------"
  done
done

echo "All trainings completed successfully!"

# nohup python3 main.py --ers-path /mnt/d/ERS/ers_jpg/ --galar-path /mnt/e/galar_jpg/ --type-num 0 --epochs 1 --k-folds 2 --model-size 0 --binary 1 --verbose 2 > logs/baseline.log 2>&1 &
