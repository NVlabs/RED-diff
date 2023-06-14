# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved

root=<root>

ETA=1.0
STEPS=1000
MODEL=imagenet256
GRAD_WEIGHT=1.0


samples_root=$root/_exp/samples     #where to save data
save_deg=False
save_ori=False
overwrite=True
smoke_test=1e5
batch_size=20  #50
num_steps=1000  #$STEPS


for DEG in sr4 in2_20ff; do
for ALGO in ddrm pgdm reddiff dps; do

IDX=${ETA}_${STEPS}_${GRAD_WEIGHT}
TARGET=$MODEL/$ALGO/$DEG/$IDX   #debug

# val=`expr $gpu_idx + 1`
# echo "$gpu_idx + 1 : $val"

echo $ETA
echo $DEG
echo $ALGO
echo $GRAD_WEIGHT

#sample
python   main.py   exp.overwrite=$overwrite   algo=$ALGO   algo.deg=$DEG    algo.eta=$ETA    exp.num_steps=$num_steps    algo.sigma_y=0.0   loader.batch_size=$batch_size    exp.seed=3    loader=imagenet256_ddrmpp    dist.num_processes_per_node=1   exp.name=$TARGET  exp.save_ori=$save_ori  exp.save_deg=$save_deg  exp.smoke_test=$smoke_test  exp.samples_root=$samples_root   algo.grad_term_weight=$GRAD_WEIGHT # algo.awd=True 

done
done
