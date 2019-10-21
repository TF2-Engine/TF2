#train dns-cp 
#args :/mnt/imagenet2012 dataset path
#      --pretrained  pretrained model used
#      --channel_select channel select 
#      --lr learning rate --kept_ratio param kept ratio 
#      --snapshotmodelname trained model save path
#      --channel_select_resume  prunde model path(output from channel_select)

#first channel select
python main.py /mnt/imagenet2012  --pretrained --channel_select --lr=0.001 --kept_ratio=0.5 --batch-size=128 --snapshotmodelname=../pytorchmodel/

#second finetune
python main.py /mnt/imagenet2012 --batch-size=128 --lr=0.01 --channel_select_resume=../pytorchmodel/20000_64.868_0.4756_checkpoint.pth.tar --snapshotmodelname=../pytorchmodel/


