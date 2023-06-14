# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved

samples_root=/home/mmardani/research/stable-diffusion-sampling-gitlab/reddiff/_exp
save_deg=True
save_ori=True
overwrite=True
smoke_test=1
batch_size=1
num_steps=10


#noisy inpaint + reddiff + adam
python   main.py   exp.overwrite=True   algo=reddiff  exp.seed=1  algo.sigma_x0=0.0   algo.awd=True    algo.deg=in2_20ff     algo.lr=0.25   exp.num_steps=$num_steps    algo.sigma_y=0.1   loader.batch_size=$batch_size    loader=imagenet256_ddrmpp    dist.num_processes_per_node=1   exp.name=debug  exp.save_ori=$save_ori  exp.save_deg=$save_deg  exp.smoke_test=$smoke_test   exp.samples_root=$samples_root      exp.save_evolution=True     algo.grad_term_weight=1.0

#noisy inpaint + dps
#tune eta 0.0 0.5 1.0
#grad_term_weight 0.1 0.25 0.5 1.0 2.0
#python   main.py   exp.overwrite=True   algo=dps    algo.deg=in2_20ff    algo.eta=0.5    exp.num_steps=$num_steps    algo.sigma_y=0.1   loader.batch_size=$batch_size    exp.seed=3    loader=imagenet256_ddrmpp    dist.num_processes_per_node=1   exp.name=debug  exp.save_ori=$save_ori  exp.save_deg=$save_deg  exp.smoke_test=$smoke_test   exp.samples_root=$samples_root   algo.grad_term_weight=0.5

#noisy inpaint + pgdm
#python   main.py   exp.overwrite=True   algo=pgdm   algo.awd=True    algo.deg=in2_20ff    algo.eta=1.0    exp.num_steps=100    algo.sigma_y=0.1   loader.batch_size=$batch_size    exp.seed=3    loader=imagenet256_ddrmpp    dist.num_processes_per_node=1   exp.name=debug  exp.save_ori=$save_ori  exp.save_deg=$save_deg  exp.smoke_test=$smoke_test   exp.samples_root=$samples_root   

#inpaint + ddrm
#python   main.py   exp.overwrite=True   algo=ddrm    algo.deg=in2_20ff    algo.eta=1.0    exp.num_steps=20    algo.sigma_y=0.1   loader.batch_size=$batch_size    exp.seed=3    loader=imagenet256_ddrmpp    dist.num_processes_per_node=1   exp.name=debug  exp.save_ori=$save_ori  exp.save_deg=$save_deg  exp.smoke_test=$smoke_test   exp.samples_root=$samples_root  





# #JPEG restoration
# #JPEG + reddiff + adam
# python   main.py   exp.overwrite=True   algo=reddiff   algo.awd=True    algo.deg=jpeg20    algo.eta=1.0    exp.num_steps=$num_steps    algo.sigma_y=0.0   loader.batch_size=$batch_size    exp.seed=3    loader=imagenet256_ddrmpp    dist.num_processes_per_node=1   exp.name=debug  exp.save_ori=$save_ori  exp.save_deg=$save_deg  exp.smoke_test=$smoke_test  exp.samples_root=$samples_root  algo.grad_term_weight=0.25   exp.save_evolution=True   algo.lr=0.5  exp.start_step=1000   exp.end_step=0




#Nonlinear HDR
#HDR + reddiff + adam
#python   main.py   exp.overwrite=True   algo=reddiff   algo.awd=True    algo.deg=hdr    algo.eta=1.0    exp.num_steps=$num_steps    algo.sigma_y=0.0   loader.batch_size=$batch_size    exp.seed=3    loader=imagenet256_ddrmpp    dist.num_processes_per_node=1   exp.name=debug  exp.save_ori=$save_ori  exp.save_deg=$save_deg  exp.smoke_test=$smoke_test  exp.samples_root=$samples_root  algo.grad_term_weight=0.25   exp.save_evolution=True   algo.lr=0.5  exp.start_step=1000   exp.end_step=0

#HDR + dps
#python   main.py   exp.overwrite=True   algo=dps   algo.awd=True    algo.deg=hdr    algo.eta=0.5    exp.num_steps=$num_steps    algo.sigma_y=0.0   loader.batch_size=$batch_size    exp.seed=3    loader=imagenet256_ddrmpp    dist.num_processes_per_node=1   exp.name=debug  exp.save_ori=$save_ori  exp.save_deg=$save_deg  exp.smoke_test=$smoke_test  exp.samples_root=$samples_root  algo.grad_term_weight=0.1

#deblur + pgdm + pgdm
#python   main.py   exp.overwrite=True   algo=pgdm   algo.awd=True    algo.deg=hdr    algo.eta=1.0    exp.num_steps=100    algo.sigma_y=0.0   loader.batch_size=$batch_size    exp.seed=3    loader=imagenet256_ddrmpp    dist.num_processes_per_node=1   exp.name=debug  exp.save_ori=$save_ori  exp.save_deg=$save_deg  exp.smoke_test=$smoke_test  exp.samples_root=$samples_root




#Nonlinear phase retrieval
#phase retrieval + reddiff + adam
#python   main.py   exp.overwrite=True   algo=reddiff   algo.awd=True    algo.deg=phase_retrieval    algo.eta=1.0    exp.num_steps=$num_steps    algo.sigma_y=0.0   loader.batch_size=$batch_size    exp.seed=3    loader=imagenet256_ddrmpp    dist.num_processes_per_node=1   exp.name=debug  exp.save_ori=$save_ori  exp.save_deg=$save_deg  exp.smoke_test=$smoke_test  exp.samples_root=$samples_root  algo.grad_term_weight=0.25   exp.save_evolution=True   algo.lr=0.5  exp.start_step=1000   exp.end_step=0

#phase retrieval + dps
#python   main.py   exp.overwrite=True   algo=dps   algo.awd=True    algo.deg=phase_retrieval    algo.eta=1.0    exp.num_steps=$num_steps    algo.sigma_y=0.0   loader.batch_size=$batch_size    exp.seed=3    loader=imagenet256_ddrmpp    dist.num_processes_per_node=1   exp.name=debug  exp.save_ori=$save_ori  exp.save_deg=$save_deg  exp.smoke_test=$smoke_test  exp.samples_root=$samples_root  algo.grad_term_weight=0.4






#Nonlinear deblurring
#deblur + reddiff + adam
#python   main.py   exp.overwrite=True   algo=reddiff   algo.awd=True    algo.deg=deblur_nl    algo.eta=1.0    exp.num_steps=$num_steps    algo.sigma_y=0.0   loader.batch_size=$batch_size    exp.seed=3    loader=imagenet256_ddrmpp    dist.num_processes_per_node=1   exp.name=debug  exp.save_ori=$save_ori  exp.save_deg=$save_deg  exp.smoke_test=$smoke_test  exp.samples_root=$samples_root  algo.grad_term_weight=0.25   exp.save_evolution=False   algo.lr=0.5  exp.start_step=1000   exp.end_step=0

#python   main.py   exp.overwrite=True   algo=dps   algo.awd=True    algo.deg=deblur_nl    algo.eta=0.5    exp.num_steps=$num_steps    algo.sigma_y=0.0   loader.batch_size=$batch_size    exp.seed=3    loader=imagenet256_ddrmpp    dist.num_processes_per_node=1   exp.name=debug  exp.save_ori=$save_ori  exp.save_deg=$save_deg  exp.smoke_test=$smoke_test  exp.samples_root=$samples_root  algo.grad_term_weight=0.05





#Gauss deblurring
#deblur + reddiff + adam
#python   main.py   exp.overwrite=True   algo=reddiff   algo.awd=True    algo.deg=deblur_gauss    algo.eta=1.0    exp.num_steps=$num_steps    algo.sigma_y=0.0   loader.batch_size=$batch_size    exp.seed=3    loader=imagenet256_ddrmpp    dist.num_processes_per_node=1   exp.name=debug  exp.save_ori=$save_ori  exp.save_deg=$save_deg  exp.smoke_test=$smoke_test  exp.samples_root=$samples_root  algo.grad_term_weight=0.25   exp.save_evolution=True   algo.lr=0.5  exp.start_step=1000   exp.end_step=0

#deblur + pgdm + pgdm
#python   main.py   exp.overwrite=True   algo=pgdm   algo.awd=True    algo.deg=deblur_gauss    algo.eta=1.0    exp.num_steps=$num_steps    algo.sigma_y=0.0   loader.batch_size=$batch_size    exp.seed=3    loader=imagenet256_ddrmpp    dist.num_processes_per_node=1   exp.name=debug  exp.save_ori=$save_ori  exp.save_deg=$save_deg  exp.smoke_test=$smoke_test  exp.samples_root=$samples_root

#deblur + dps
#python   main.py   exp.overwrite=True   algo=dps   algo.awd=True    algo.deg=deblur_gauss    algo.eta=1.0    exp.num_steps=$num_steps    algo.sigma_y=0.0   loader.batch_size=$batch_size    exp.seed=3    loader=imagenet256_ddrmpp    dist.num_processes_per_node=1   exp.name=debug  exp.save_ori=$save_ori  exp.save_deg=$save_deg  exp.smoke_test=$smoke_test  exp.samples_root=$samples_root  algo.grad_term_weight=2.0

#deblur + ddrm
#python   main.py   exp.overwrite=True   algo=ddrm    algo.deg=deblur_gauss    algo.eta=1.0    exp.num_steps=$num_steps    algo.sigma_y=0.0   loader.batch_size=$batch_size    exp.seed=3    loader=imagenet256_ddrmpp    dist.num_processes_per_node=1   exp.name=debug  exp.save_ori=$save_ori  exp.save_deg=$save_deg  exp.smoke_test=$smoke_test   exp.samples_root=$samples_root -cn imagenet256_cond   #model.ckpt=imagenet256_cond



#SR
#sr + reddiff + adam + parallel
#python   main.py   exp.overwrite=True   algo=reddiff_parallel   algo.awd=True    algo.deg=sr4    algo.eta=1.0    exp.num_steps=$num_steps    algo.sigma_y=0.0   loader.batch_size=$batch_size    exp.seed=3    loader=imagenet256_ddrmpp    dist.num_processes_per_node=1   exp.name=debug  exp.save_ori=$save_ori  exp.save_deg=$save_deg  exp.smoke_test=$smoke_test   exp.samples_root=$samples_root

#sr + reddiff + adam
#python   main.py   exp.overwrite=True   algo=reddiff   algo.awd=True    algo.deg=sr4    algo.eta=1.0    exp.num_steps=$num_steps    algo.sigma_y=0.0   loader.batch_size=$batch_size    exp.seed=3    loader=imagenet256_ddrmpp    dist.num_processes_per_node=1   exp.name=debug  exp.save_ori=$save_ori  exp.save_deg=$save_deg  exp.smoke_test=$smoke_test  exp.samples_root=$samples_root

#sr + pgdm + pgdm
#python   main.py   exp.overwrite=True   algo=pgdm   algo.awd=True    algo.deg=sr4    algo.eta=1.0    exp.num_steps=$num_steps    algo.sigma_y=0.0   loader.batch_size=$batch_size    exp.seed=3    loader=imagenet256_ddrmpp    dist.num_processes_per_node=1   exp.name=debug  exp.save_ori=$save_ori  exp.save_deg=$save_deg  exp.smoke_test=$smoke_test  exp.samples_root=$samples_root

#sr + dps
#choose eta
#grad_term_weight 0.1 0.25 0.5 1.0 2.0
#python   main.py   exp.overwrite=$overwrite   algo=dps    algo.deg=sr4    algo.eta=0.0    exp.num_steps=$num_steps    algo.sigma_y=0.0   loader.batch_size=$batch_size    exp.seed=3    loader=imagenet256_ddrmpp    dist.num_processes_per_node=1   exp.name=debug  exp.save_ori=$save_ori  exp.save_deg=$save_deg  exp.smoke_test=$smoke_test   exp.samples_root=$samples_root   algo.grad_term_weight=1.0

#sr + ddrm
#python   main.py   exp.overwrite=True   algo=ddrm    algo.deg=sr4    algo.eta=1.0    exp.num_steps=$num_steps    algo.sigma_y=0.0   loader.batch_size=$batch_size    exp.seed=3    loader=imagenet256_ddrmpp    dist.num_processes_per_node=1   exp.name=debug  exp.save_ori=$save_ori  exp.save_deg=$save_deg  exp.smoke_test=$smoke_test   exp.samples_root=$samples_root -cn imagenet256_cond   #model.ckpt=imagenet256_cond




#INPAINT
#inpaint + reddiff + adam + parallel
#python   main.py   exp.overwrite=True   algo=reddiff_parallel   algo.grad_term_weight=0.5    algo.lr=0.25  algo.awd=True    algo.deg=in2_20ff    algo.eta=1.0    exp.num_steps=$num_steps    algo.sigma_y=0.0   loader.batch_size=$batch_size    exp.seed=3    loader=imagenet256_ddrmpp    dist.num_processes_per_node=1   exp.name=debug  exp.save_ori=$save_ori  exp.save_deg=$save_deg  exp.smoke_test=$smoke_test   exp.samples_root=$samples_root

#inpaint + reddiff + adam
#python   main.py   exp.overwrite=True   algo=reddiff  exp.seed=1  algo.sigma_x0=0.0001   algo.lr=0.25   algo.awd=True    algo.deg=in2_20ff    algo.eta=1.0    exp.num_steps=$num_steps    algo.sigma_y=0.0   loader.batch_size=$batch_size    loader=imagenet256_ddrmpp    dist.num_processes_per_node=1   exp.name=debug  exp.save_ori=$save_ori  exp.save_deg=$save_deg  exp.smoke_test=$smoke_test   exp.samples_root=$samples_root

#inpaint + pgdm
#python   main.py   exp.overwrite=True   algo=pgdm   algo.awd=True    algo.deg=in2_20ff    algo.eta=1.0    exp.num_steps=$num_steps    algo.sigma_y=0.0   loader.batch_size=$batch_size    exp.seed=3    loader=imagenet256_ddrmpp    dist.num_processes_per_node=1   exp.name=debug  exp.save_ori=$save_ori  exp.save_deg=$save_deg  exp.smoke_test=$smoke_test   exp.samples_root=$samples_root -cn imagenet256_cond   #model.ckpt=imagenet256_cond

#inpaint + mcg
#gives nans
#grad_term_weight needs to be tuned properly
#choose eta
#grad_term_weight ??
#python   main.py   exp.overwrite=True   algo=mcg    algo.deg=in2_20ff    algo.eta=0.0    exp.num_steps=$num_steps    algo.sigma_y=0.0   loader.batch_size=$batch_size    exp.seed=3    loader=imagenet256_ddrmpp    dist.num_processes_per_node=1   exp.name=debug  exp.save_ori=$save_ori  exp.save_deg=$save_deg  exp.smoke_test=$smoke_test   exp.samples_root=$samples_root

#inpaint + dps
#choose eta
#grad_term_weight 0.1 0.25 0.5 1.0 2.0
#python   main.py   exp.overwrite=True   algo=dps    algo.deg=in2_20ff    algo.eta=0.0    exp.num_steps=$num_steps    algo.sigma_y=0.0   loader.batch_size=$batch_size    exp.seed=3    loader=imagenet256_ddrmpp    dist.num_processes_per_node=1   exp.name=debug  exp.save_ori=$save_ori  exp.save_deg=$save_deg  exp.smoke_test=$smoke_test   exp.samples_root=$samples_root   algo.grad_term_weight=0.5

#inpaint + ddrm
#python   main.py   exp.overwrite=True   algo=ddrm    algo.deg=in2_20ff    algo.eta=1.0    exp.num_steps=$num_steps    algo.sigma_y=0.0   loader.batch_size=$batch_size    exp.seed=3    loader=imagenet256_ddrmpp    dist.num_processes_per_node=1   exp.name=debug  exp.save_ori=$save_ori  exp.save_deg=$save_deg  exp.smoke_test=$smoke_test   exp.samples_root=$samples_root -cn imagenet256_cond   #model.ckpt=imagenet256_cond
