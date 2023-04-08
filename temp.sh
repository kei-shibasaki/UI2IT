CUDA_VISIBLE_DEVICES=7 python -m train_codes.train_idt -c config/config_lptn.json
CUDA_VISIBLE_DEVICES=7 python -m train_codes.train_cycle_up -c config/config_cycle.json
CUDA_VISIBLE_DEVICES=7 python -m train_codes.train_cycle_idt -c config/config_cycle_anime.json
CUDA_VISIBLE_DEVICES=4 python -m train_codes.train_mspc_paper -c config/config_mspc_anime.json
CUDA_VISIBLE_DEVICES=7 python -m train_codes.train_mspc -c config/config_mspc2.json
CUDA_VISIBLE_DEVICES=5 python -m train_codes.train_mspc_wgan -c config/config_mspc_wgan.json

CUDA_VISIBLE_DEVICES=1 python -m train_codes.train_mspc_mod_mod -c config/config_mspc_anime_mod_mod.json
CUDA_VISIBLE_DEVICES=5 python -m train_codes.train_mspc_mod_mod -c config/config_mspc_mod_mod.json

CUDA_VISIBLE_DEVICES=0 python -m train_codes.train_cycle_up -c config/config_cycle.json
CUDA_VISIBLE_DEVICES=7 python -m train_codes.train_cycle_idt -c config/config_cycle.json

CUDA_VISIBLE_DEVICES=0,1,2,3 python -m train_codes.train_cycle_up_dp -c config/config_cycle.json
CUDA_VISIBLE_DEVICES=1,2 python -m train_codes.train_cycle_idt_dp -c config/config_cycle.json
CUDA_VISIBLE_DEVICES=3,5 python -m train_codes.train_cycle_idt_dp -c config/config_cycle2.json

CUDA_VISIBLE_DEVICES=7 python -m train_codes.train_mspc_paper -c config/config_mspc.json
CUDA_VISIBLE_DEVICES=7 python -m train_codes.train_mspc_lsgan -c config/config_mspc.json

CUDA_VISIBLE_DEVICES=5 python -m train_codes.train_mspc_lsgan -c config/config_mspc2.json
CUDA_VISIBLE_DEVICES=1 python -m train_codes.train_mspc_lsgan -c config/config_mspc.json
CUDA_VISIBLE_DEVICES=2 python -m train_codes.train_mspc_lsgan -c config/config_mspc.json
CUDA_VISIBLE_DEVICES=2 python -m train_codes.train_mspc_lsgan_clip_later -c config/config_mspc_clip_later.json

CUDA_VISIBLE_DEVICES=0 python -m train_codes.train_mspc_one -c config/config_mspc_one.json
CUDA_VISIBLE_DEVICES=2 python -m train_codes.train_mspc -c config/config_mspc.json



CUDA_VISIBLE_DEVICES=5 python temp.py 