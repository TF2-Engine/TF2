#nohup ./bin/host ../host/model/googlenet/param.bin ../host/model/googlenet_Q ../host/test_images/googlenet_data_label_391.bin ../host/model/googlenet/loss3_classifier.bin 1 &
#./bin/host ../host/model/googlenet/param.bin ../host/model/googlenet_Q ../host/test_images/googlenet_data_label_391.bin ../host/model/googlenet/loss3_classifier.bin 50
#env CL_CONTEXT_EMULATOR_DEVICE_ALTERA=1 ./bin/host ../host/model/googlenet/param.bin ../host/model/googlenet_Q ../host/test_images/googlenet_data_label_391.bin ../host/verify/googlenet_fc1000_label_391.bin 3
#./bin/host ../host/model/googlenet/param.bin ../host/model/googlenet_Q ../host/test_images/googlenet_data_label_391.bin ../host/verify/googlenet_fc1000_label_391.bin 16
#env CL_CONTEXT_EMULATOR_DEVICE_ALTERA=1 ./bin/host ../host/model/resnet50/param.bin ../host/model/resnet50_Q ../host/test_images/resnet50_data_label_100.bin ../host/verify/resnet50_fc1000_label_100.bin 2
#./bin/host ../host/model/pytorch_resnet50/fpgamodel.bin ../host/model/pytorch_resnet50_q ../imagenet_test_images/ILSVRC2012_val_00000001.JPEG ../host/model/pytorch_resnet50/fc1000.bin 1 
#nohup ./bin/host ../host/model/pytorch_resnet50/fpgamodel.bin ../host/model/pytorch_resnet50_q ../ILSVRC2012_val_00000001.JPEG ../host/model/pytorch_resnet50/fc1000.bin 1 &
#nohup ./bin/host ../host/model/pytorch_resnet50/fpgamodel.bin ../host/model/pytorch_resnet50_q ../valimages/ILSVRC2012_val_00000293.JPEG ../host/model/pytorch_resnet50/fc1000.bin 1 &
#./bin/host ../host/model/pytorch_resnet50/fpgamodel.bin ../host/model/pytorch_resnet50_q ../valimages/ILSVRC2012_val_00000293.JPEG ../host/model/pytorch_resnet50/fc1000.bin 50
nohup ./bin/host ../host/model/pytorch_resnet50/fpgamodel.bin ../host/model/pytorch_resnet50_q ../valimages/ILSVRC2012_val_00000293.JPEG ../host/model/pytorch_resnet50/fc1000.bin 50000 &
#nohup ./bin/host ../host/model/pytorch_resnet50/fpgamodel.bin ../host/model/pytorch_resnet50_q ../host/model/pytorch_resnet50/data.bin ../host/model/pytorch_resnet50/fc1000.bin 1 &
#./bin/host ../host/model/pytorch_resnet50/fpgamodel.bin ../host/model/pytorch_resnet50_q ../host/model/pytorch_resnet50/data.bin ../host/model/pytorch_resnet50/fc1000.bin 500 
#nohup ./bin/host ../host/model/pytorch_resnet50/fpgamodel.bin ../host/model/pytorch_resnet50_q ../host/model/pytorch_resnet50/data.bin ../host/model/pytorch_resnet50/fc1000.bin 50000 &
#./bin/host ../host/model/resnet50/param.bin ../host/model/resnet50_Q ../host/test_images/resnet50_data_label_100.bin ../host/verify/resnet50_fc1000_label_100.bin 50000
#nohup ./bin/host ../host/model/resnet50/param.bin ../host/model/resnet50_Q ../host/test_images/resnet50_data_label_100.bin ../host/verify/resnet50_fc1000_label_100.bin 50000 &
#./bin/host ../host/model/resnet50_pruned/param.bin ../host/model/resnet50_pruned_Q ../host/test_images/ILSVRC2012_val_00000001.JPEG ../host/verify/resnet50_fc1000_label_100.bin 1
