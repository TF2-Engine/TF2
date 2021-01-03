source /home/fpga_env_191.sh
source recompile.sh
make clean
# make RESNET50=1
make RESNET50=1 DEBUG=1
# make IMAGENET=1 RESNET50=1 DEBUG=1
# make SQUEEZENET=1 DEBUG=1
#make IMAGENET=1 SQUEEZENET=1 DEBUG=1
# source recompile.sh
#source run.sh > Verification_$time_.log
