if [ $# -lt 1 ]; then
    echo "Usage: $0 gpu "
    exit 1
fi
gpu="$1"

for track_id in "alley_1" "ambush_5" "bamboo_2" "bandage_1" "cave_2" "market_6" "shaman_2" "sleeping_1" "temple_2" 
do
    bash ./experiments/davis/train_sequence.sh ${gpu} --track_id ${track_id} --path_suffix _sintel \
    --backbone robust_CVD/Sintel/googckpt_3epoch/${track_id}/R0-100_hierarchical2_midas2/StD100.0_StR1.0_SmD0_SmR0.0/checkpoints/0003.pt
done