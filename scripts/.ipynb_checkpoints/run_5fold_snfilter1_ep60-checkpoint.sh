accelerate launch --num_processes=1 ../src/train.py --fold=0 --model_name='squeeze-192-12-4-snfilter1-ep60-fold0-seed2023'
accelerate launch --num_processes=1 ../src/train.py --fold=1 --model_name='squeeze-192-12-4-snfilter1-ep60-fold1-seed2023'
accelerate launch --num_processes=1 ../src/train.py --fold=2 --model_name='squeeze-192-12-4-snfilter1-ep60-fold2-seed2023'
accelerate launch --num_processes=1 ../src/train.py --fold=3 --model_name='squeeze-192-12-4-snfilter1-ep60-fold3-seed2023'
accelerate launch --num_processes=1 ../src/train.py --fold=4 --model_name='squeeze-192-12-4-snfilter1-ep60-fold4-seed2023'