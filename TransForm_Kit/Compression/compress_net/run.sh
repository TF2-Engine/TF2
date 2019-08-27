#! /bin/bash
percentage_array=(0.0 0.3 0.6 0.8 0.9 1.0) # quan percentage,you can change by yourself
train_val_imgpath="/mnt/imagenet2012/" #img path 
snapshot_modelpath="./model/part" #compress_modelname
network="resnet50" #networkname 
lr=0.0001 #learing rate
maxepoch=30 #max epoch
#first compress 0.0->0.3 need pretrained model
python ./examples/imagenet/main.py /mnt/imagenet2012/ --pretrained --pre_percentage=${percentage_array[0]} --cur_percentage=${percentage_array[1]} --snapshotmodelname=""$snapshot_modelpath"1" --lr 0.0001 --epochs=$maxepoch --a=$network

#next compress after some step,full model will be compressed
length=${#percentage_array[@]}
curpart=2
while(($curpart<$length))
do
    if [ $curpart == `expr $length - 1` ]
    then
        maxepoch=1
    fi
    prepart=`expr $curpart - 1`
    python ./examples/imagenet/main.py $train_val_imgpath --resume=""$snapshot_modelpath""$prepart"_model_best.pth.tar" --pre_percentage=${percentage_array[$prepart]} --cur_percentage=${percentage_array[$curpart]} --snapshotmodelname=""$snapshot_modelpath""$curpart"" --lr=$lr --epochs=$maxepoch --a=$network  
    let "curpart++"    
done                                                     
