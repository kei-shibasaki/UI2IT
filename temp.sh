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

CUDA_VISIBLE_DEVICES=2 python -m train_codes.train_mspc_paper -c config/config_mspc_anime.json
CUDA_VISIBLE_DEVICES=7 python -m train_codes.train_mspc_lsgan -c config/config_mspc.json

CUDA_VISIBLE_DEVICES=5 python -m train_codes.train_mspc_lsgan -c config/config_mspc2.json
CUDA_VISIBLE_DEVICES=1 python -m train_codes.train_mspc_lsgan -c config/config_mspc.json
CUDA_VISIBLE_DEVICES=2 python -m train_codes.train_mspc_lsgan -c config/config_mspc.json
CUDA_VISIBLE_DEVICES=2 python -m train_codes.train_mspc_lsgan_clip_later -c config/config_mspc_clip_later.json

CUDA_VISIBLE_DEVICES=7 python -m train_codes.train_mspc_one_sal -c config/config_mspc_one_sal_anime.json
CUDA_VISIBLE_DEVICES=7 python -m train_codes.train_mspc_one_sal2 -c config/config_mspc_one_sal_2.json
CUDA_VISIBLE_DEVICES=3 python -m train_codes.train_mspc -c config/config_mspc.json
CUDA_VISIBLE_DEVICES=3 python -m train_codes.train_mspc -c config/config_mspc.json

CUDA_VISIBLE_DEVICES=6 python -m train_codes.train_ours -c config/config_ours.json
CUDA_VISIBLE_DEVICES=7 python -m train_codes.train_ours -c config/config_ours2.json

CUDA_VISIBLE_DEVICES=3 python -m train_codes.train_ours -c config/config_ours_anime.json
CUDA_VISIBLE_DEVICES=2 python -m train_codes.train_ours -c config/config_ours_anime2.json
CUDA_VISIBLE_DEVICES=7 python -m train_codes.train_ours -c config/config_ours_anime3.json
CUDA_VISIBLE_DEVICES=6 python -m train_codes.train_ours -c config/config_ours_anime2.json
CUDA_VISIBLE_DEVICES=2 python -m train_codes.train_ours_2 -c config/config_ours_anime4.json


CUDA_VISIBLE_DEVICES=7 python -m train_codes.train_ours -c config/config_ours_anime2.json

CUDA_VISIBLE_DEVICES=2 python -m train_codes.train_ours -c config/config_ours_anime2.json
CUDA_VISIBLE_DEVICES=2 python -m train_codes.train_ours_2 -c config/config_ours_anime3.json
CUDA_VISIBLE_DEVICES=3 python -m train_codes.train_ours_idt -c config/config_ours_anime4.json
CUDA_VISIBLE_DEVICES=3 python -m train_codes.train_ours_mask -c config/config_ours_anime_mask.json
CUDA_VISIBLE_DEVICES=7 python -m train_codes.train_ours_premask -c config/config_ours_anime_premask.json
CUDA_VISIBLE_DEVICES=2 python -m train_codes.train_ours_premask -c config/config_ours_anime_premask2.json
CUDA_VISIBLE_DEVICES=3 python -m train_codes.train_ours_premask_idt -c config/config_ours_anime_premask2.json

CUDA_VISIBLE_DEVICES=6 python -m train_codes.train_mspc_paper -c config/config_mspc_front.json
CUDA_VISIBLE_DEVICES=6 python -m train_codes.train_mspc_lsgan -c config/config_mspc_front.json

CUDA_VISIBLE_DEVICES=3 python -m train_codes.train_ours_idt -c config/config_ours_anime3.json

CUDA_VISIBLE_DEVICES=6 python -m train_codes.train_mspc_paper -c config/config_mspc.json
CUDA_VISIBLE_DEVICES=7 python -m train_codes.train_mspc_paper -c config/config_mspc_anime.json
CUDA_VISIBLE_DEVICES=1 python -m train_codes.train_mspc_paper -c config/config_mspc_cat2dog.json
CUDA_VISIBLE_DEVICES=2 python -m train_codes.train_mspc_paper -c config/config_mspc_apple2orange.json

CUDA_VISIBLE_DEVICES=3 python -m train_codes.train_ours_idt -c config/config_ours2.json
CUDA_VISIBLE_DEVICES=2 python -m train_codes.train_ours_idt -c config/config_ours_anime.json
CUDA_VISIBLE_DEVICES=2 python -m train_codes.train_ours_adv_idt -c config/config_ours.json
CUDA_VISIBLE_DEVICES=6 python -m train_codes.train_ours_adv_idt -c config/config_ours_anime3.json


CUDA_VISIBLE_DEVICES=6 python -m train_codes.train_ours_idt -c config/config_ours_anime2.json
CUDA_VISIBLE_DEVICES=7 python -m train_codes.train_ours_adv_idt -c config/config_ours_anime.json
CUDA_VISIBLE_DEVICES=1 python -m train_codes.train_ours_idt_foreonly -c config/config_ours_anime3.json
CUDA_VISIBLE_DEVICES=3 python -m train_codes.train_ours_idt_foreonly_resnet -c config/config_ours4.json

CUDA_VISIBLE_DEVICES=7 python -m train_codes.train_ours_idt_foreonly3 -c config/config_ours3.json


CUDA_VISIBLE_DEVICES=3 python temp.py 

tar -cvf res_ours_ab.tar experiments/ours_horse2zebra_lr_sep_idt_foreonly_multires/generated/140000 experiments/ours_horse2zebra_lr_sep_idt_foreonly3/generated/180000