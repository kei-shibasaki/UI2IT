CUDA_VISIBLE_DEVICES=7 python -m train_codes.train_idt -c config/config_lptn.json
CUDA_VISIBLE_DEVICES=7 python -m train_codes.train_cycle -c config/config_lptn_cycle.json
CUDA_VISIBLE_DEVICES=7 python -m train_codes.train_cycle -c config/config_cycle.json
CUDA_VISIBLE_DEVICES=7 python -m train_codes.train_mspc -c config/config_mspc.json
CUDA_VISIBLE_DEVICES=7 python -m train_codes.train_mspc_wgan -c config/config_mspc.json


CUDA_VISIBLE_DEVICES=7 python temp.py 