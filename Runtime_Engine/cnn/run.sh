#./bin/host ../host/model/googlenet/param.bin ../host/model/googlenet_Q ../host/test_images/googlenet_data_label_391.bin ../host/model/googlenet/loss3_classifier.bin 1
#env CL_CONTEXT_EMULATOR_DEVICE_ALTERA=1 ./bin/host ../host/model/googlenet/param.bin ../host/model/googlenet_Q ../host/test_images/googlenet_data_label_391.bin ../host/verify/googlenet_fc1000_label_391.bin 3
#./bin/host ../host/model/googlenet/param.bin ../host/model/googlenet_Q ../host/test_images/googlenet_data_label_391.bin ../host/verify/googlenet_fc1000_label_391.bin 16
#env CL_CONTEXT_EMULATOR_DEVICE_ALTERA=1 ./bin/host ../host/model/googlenet/param.bin ../host/model/googlenet_Q ../host/test_images/googlenet_data_label_391.bin ../host/verify/googlenet_fc1000_label_391.bin 1
#env CL_CONTEXT_EMULATOR_DEVICE_ALTERA=1 ./bin/host ../host/model/resnet50/param.bin ../host/model/resnet50_Q ../host/test_images/resnet50_data_label_100.bin ../host/verify/resnet50_fc1000_label_100.bin 2
#./bin/host ../host/model/resnet50/param.bin ../host/model/resnet50_Q ../host/test_images/resnet50_data_label_100.bin ../host/verify/resnet50_fc1000_label_100.bin 16
#nohup env CL_CONTEXT_EMULATOR_DEVICE_ALTERA=1 ./bin/host ../host/model/resnet50/param.bin ../host/model/resnet50_Q ../host/model/resnet50/data.bin ../host/model/resnet50/res3b_branch2a-scale.bin 2 &
#nohup env CL_CONTEXT_EMULATOR_DEVICE_ALTERA=1 ./bin/host ../host/model/resnet50/param.bin ../host/model/resnet50_Q ../host/model/resnet50/data.bin ../host/model/resnet50/res2c.bin 2 &
#nohup env CL_CONTEXT_EMULATOR_DEVICE_ALTERA=1 ./bin/host ../host/model/resnet50/param.bin ../host/model/resnet50_Q ../host/model/resnet50/data.bin ../host/model/resnet50/pool1.bin 2 &
#nohup env CL_CONTEXT_EMULATOR_DEVICE_ALTERA=1 ./bin/host ../host/model/resnet50/param.bin ../host/model/resnet50_Q ../host/model/resnet50/data.bin ../host/model/resnet50/res2a_branch1-scale.bin 2 &
nohup env CL_CONTEXT_EMULATOR_DEVICE_ALTERA=1 ./bin/host ../host/model/resnet50/param.bin ../host/model/resnet50_Q ../host/model/resnet50/data.bin ../host/model/resnet50/res2a_branch2a-scale.bin 2 &
#nohup env CL_CONTEXT_EMULATOR_DEVICE_ALTERA=1 ./bin/host ../host/model/resnet50/param.bin ../host/model/resnet50_Q ../host/model/resnet50/data.bin ../host/model/resnet50/res2a_branch2b-scale.bin 2 &
#nohup env CL_CONTEXT_EMULATOR_DEVICE_ALTERA=1 ./bin/host ../host/model/resnet50/param.bin ../host/model/resnet50_Q ../host/model/resnet50/data.bin ../host/model/resnet50/res2a.bin 2 &
#nohup env CL_CONTEXT_EMULATOR_DEVICE_ALTERA=1 ./bin/host ../host/model/resnet50/param.bin ../host/model/resnet50_Q ../host/model/resnet50/data.bin ../host/model/resnet50/res2b_branch2a-scale.bin 2 &
#nohup env CL_CONTEXT_EMULATOR_DEVICE_ALTERA=1 ./bin/host ../host/model/resnet50/param.bin ../host/model/resnet50_Q ../host/model/resnet50/data.bin ../host/model/resnet50/res2b_branch2b-scale.bin 2 &
#nohup env CL_CONTEXT_EMULATOR_DEVICE_ALTERA=1 ./bin/host ../host/model/resnet50/param.bin ../host/model/resnet50_Q ../host/model/resnet50/data.bin ../host/model/resnet50/res2b.bin 2 &
#nohup env CL_CONTEXT_EMULATOR_DEVICE_ALTERA=1 ./bin/host ../host/model/resnet50/param.bin ../host/model/resnet50_Q ../host/model/resnet50/data.bin ../host/model/resnet50/res2c_branch2a-scale.bin 2 &
#nohup env CL_CONTEXT_EMULATOR_DEVICE_ALTERA=1 ./bin/host ../host/model/resnet50/param.bin ../host/model/resnet50_Q ../host/model/resnet50/data.bin ../host/model/resnet50/res2c_branch2b-scale.bin 2 &
#nohup env CL_CONTEXT_EMULATOR_DEVICE_ALTERA=1 ./bin/host ../host/model/resnet50/param.bin ../host/model/resnet50_Q ../host/model/resnet50/data.bin ../host/model/resnet50/res3a_branch1-scale.bin 2 &
#nohup env CL_CONTEXT_EMULATOR_DEVICE_ALTERA=1 ./bin/host ../host/model/resnet50/param.bin ../host/model/resnet50_Q ../host/model/resnet50/data.bin ../host/model/resnet50/res3a_branch2a-scale.bin 2 &
#nohup env CL_CONTEXT_EMULATOR_DEVICE_ALTERA=1 ./bin/host ../host/model/resnet50/param.bin ../host/model/resnet50_Q ../host/model/resnet50/data.bin ../host/model/resnet50/res3a_branch2b-scale.bin 2 &
#nohup env CL_CONTEXT_EMULATOR_DEVICE_ALTERA=1 ./bin/host ../host/model/resnet50/param.bin ../host/model/resnet50_Q ../host/model/resnet50/data.bin ../host/model/resnet50/res3a.bin 2 &
