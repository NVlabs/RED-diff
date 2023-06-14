# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved

root=/lustre/fsw/nvresearch/mmardani/output/latent-diffusion-sampling/pgdm
code=$root
dataset=$SHARE_DATA/imagenet-root
subset_txt=./misc/dgp_top1k.txt
#subset_txt=./misc/dgp_top10.txt
#subset_txt=./misc/dgp_top100.txt

GPUS=1

EXPDIR=$root/_exp
SOURCE_FID=$EXPDIR/fid_stats/imagenet256_train_mean_std.npz
SOURCE_KID=$EXPDIR/fid_stats/imagenet256_val_dgp_top1k.npy

MEAN_STD_STATS=False
SPLIT=custom


declare -a models=("imagenet256_uncond")  #("imagenet256_uncond" "imagenet256_cond")

# declare -a degs=("sr4" "in2_20ff") # ("deblur_gauss" "deblur_uni")  ("deblur_nl" "hdr" "phase_retrieval")
# declare -a algs=("ddrmppvarinf" "ddrm" "ddrmpp" "dps")   #"ddrmppvarinf_parallel"

# declare -a degs=("deblur_gauss" "deblur_uni")
# declare -a algs=("ddrmppvarinf" "ddrm" "ddrmpp" "dps")   #"ddrmppvarinf_parallel"

declare -a degs=("deblur_nl" "hdr" "phase_retrieval")
declare -a algs=("reddiff" "dps")


for MODEL in ${models[@]}; do
for DEG in ${degs[@]}; do
for ALGO in ${algs[@]}; do   #mcg

TARGET_DIR=$MODEL/$ALGO/$DEG
arr=($EXPDIR/samples/$TARGET_DIR/*)    # This creates an array of the full paths to all subdirs
arr=("${arr[@]##*/}") 


for DIR in ${arr[@]}; do

DIR_PARSE=(${DIR//_/ })

ETA=${DIR_PARSE[0]}
STEPS=${DIR_PARSE[1]}
GRAD_WEIGHT=${DIR_PARSE[2]}
GRAD_TYPE=${DIR_PARSE[3]}


IDX=$DIR
TARGET=$TARGET_DIR/$IDX

echo "--------------------------------"
echo $GRAD_WEIGHT
echo $GRAD_TYPE
echo $ETA
echo $ALGO
echo $DEG
echo $IDX
echo $TARGET
echo $STEPS  


cd $code/eval

#FID
#FID STATS
python fid_stats.py mean_std_stats=True dist.num_processes_per_node=$GPUS save_path=$TARGET/fid exp.root=$EXPDIR dataset.root=$EXPDIR/samples/$TARGET  dataset.meta_root=$dataset  dataset.split=$SPLIT  dataset.subset_txt=$subset_txt
#FID - MEAN_STD_STATS=True
python fid.py path1=$EXPDIR/fid_stats/$TARGET/fid_mean_std.npz  path2=$SOURCE_FID results=$TARGET exp.root=$EXPDIR

#KID
#KID STATS
python fid_stats.py mean_std_stats=False dist.num_processes_per_node=$GPUS save_path=$TARGET/kid_gen_dgp exp.root=$EXPDIR dataset.root=$EXPDIR/samples/$TARGET  dataset.meta_root=$dataset  dataset.split=$SPLIT  dataset.subset_txt=$subset_txt
#python fid_stats.py mean_std_stats=False dist.num_processes_per_node=$GPUS save_path=$SOURCE_KID exp.root=$EXPDIR dataset.root=$dataset/imagenet/val  dataset.meta_root=$dataset  dataset.split=$SPLIT  dataset.subset_txt=$subset_txt
#KID - MEAN_STD_STATS=False
python fid.py path1=$EXPDIR/fid_stats/$TARGET/kid_gen_dgp.npy path2=$SOURCE_KID results=$TARGET exp.root=$EXPDIR

#PSNR
python psnr.py dist.num_processes_per_node=$GPUS exp.root=$EXPDIR  dataset1.root=$dataset  dataset2.root=$EXPDIR/samples/$TARGET  dataset2.split=custom dataset1.meta_root=$dataset  dataset2.meta_root=$dataset results=$TARGET  dataset1.subset_txt=$subset_txt  dataset2.subset_txt=$subset_txt

#top1 accuracy
python ca.py  dataset.transform=ca_cropped  dataset.root=$EXPDIR/samples/$TARGET   dataset.meta_root=$dataset  dataset.split=${SPLIT}   exp.root=$EXPDIR   results=$TARGET  dataset.subset_txt=$subset_txt


done
done
done
done