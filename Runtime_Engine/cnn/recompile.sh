rm -r device/googlenet/cnn
aoc -march=emulator device/src/cnn.cl -o device/googlenet/cnn.aocx -I device/src/lib -L device/src -l device/src/opencl_lib.aoclib -board=inspur_2bank_a10 -v
make clean
make
