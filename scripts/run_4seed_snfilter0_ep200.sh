accelerate launch --num_processes=1 ../src/train.py --use_all_train=True --train_signal_to_noise_filter=0 --seed=111 --epoch=200 --model_name='squeeze-192-12-4-snfilter0-ep200-foldall-seed111'
accelerate launch --num_processes=1 ../src/train.py --use_all_train=True --train_signal_to_noise_filter=0 --seed=222 --epoch=200 --model_name='squeeze-192-12-4-snfilter0-ep200-foldall-seed222'
accelerate launch --num_processes=1 ../src/train.py --use_all_train=True --train_signal_to_noise_filter=0 --seed=333 --epoch=200 --model_name='squeeze-192-12-4-snfilter0-ep200-foldall-seed333'
accelerate launch --num_processes=1 ../src/train.py --use_all_train=True --train_signal_to_noise_filter=0 --seed=444 --epoch=200 --model_name='squeeze-192-12-4-snfilter0-ep200-foldall-seed444'