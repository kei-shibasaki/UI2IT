dirs=("experiments/ours_anime_premask" "experiments/ours_anime_premask_nosep" "experiments/ours_anime_premask_nosep_idt" "experiments/ours_apple2orange_lr_sep_recons_idt" "experiments/ours_apple2orange_saladv_lr_sep_recons_idt" "experiments/ours_edges2handbags" "experiments/ours_edges2handbags_oneside_b4" "experiments/ours_edges2handbags_oneside_b4_2" "experiments/ours_edges2handbags_oneside_idt10" "experiments/ours_edges2shoes" "experiments/ours_edges2shoes_oneside_b4" "experiments/ours_edges2shoes_oneside_idt10" "experiments/ours_edges2shoes2" "experiments/ours_horse2zebra_lr_sep_idt_foreonly_mod" "experiments/ours_horse2zebra_lr_sep_recons_idt" "experiments/ours_horse2zebra_lr_sep_recons_idt10" "experiments/ours_horse2zebra_lr_sep_recons10_idt" "experiments/ours_horse2zebra_lr_sep_recons10_idt10" "experiments/ours_horse2zebra_saladv_lr_sep_recons_idt" "experiments/ours_selfie2anime_p3m10k" "experiments/ours_selfie2anime_p3m10k_foreonly" "experiments/ours_selfie2anime_p3m10k_foreonly_binary" "experiments/ours_selfie2anime_p3m10k_saladv")

# Loop through all the directories in the source directory
for dir in ${dirs[@]}; do
    echo $dir
    rm -rf $dir
done