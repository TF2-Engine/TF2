#aoc -march=emulator device/src/cnn.cl -o device/googlenet/cnn.aocx -v
#aoc -march=emulator -DRESNET50 device/src/cnn.cl -o device/resnet50/cnn.aocx -v
aoc -march=emulator -DRESNET50 device/src/cnn.cl -o device/resnet50/cnn.aocx -I device/src/lib -L device/src -l device/src/opencl_lib.aoclib -board=inspur_2bank_a10 -v
#aoc -march=emulator -DRESNET50_PRUNED device/src/cnn.cl -o device/resnet50_pruned/cnn.aocx -v
