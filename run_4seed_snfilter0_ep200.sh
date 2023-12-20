torchrun --nproc_per_node=1 train.py --world_size=1 --use_all_train=True --train_signal_to_noise_filter=0 --seed=111 --epoch=200 --warmup=1
torchrun --nproc_per_node=1 train.py --world_size=1 --use_all_train=True --train_signal_to_noise_filter=0 --seed=222 --epoch=200 --warmup=1
torchrun --nproc_per_node=1 train.py --world_size=1 --use_all_train=True --train_signal_to_noise_filter=0 --seed=333 --epoch=200 --warmup=1
torchrun --nproc_per_node=1 train.py --world_size=1 --use_all_train=True --train_signal_to_noise_filter=0 --seed=444 --epoch=200 --warmup=1
