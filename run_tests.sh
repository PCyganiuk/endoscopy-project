#python3 main.py --ers-path /mnt/e/ERS/ers_jpg/ --galar-path /mnt/e/galar/ --type-num 0 --epochs 1 --k-folds 2 --model-size 0

# ps aux | grep python <- check if still running

# tail -f output.log <- get the output to console

# kill <PID> <- kill if nessesary

nohup python3 main.py --ers-path /mnt/e/ERS/ers_jpg/ --galar-path /mnt/e/galar/ --type-num 0 --epochs 1 --k-folds 2 --model-size 0 --verbose 1 > logs/output.log 2>&1 &