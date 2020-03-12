#nohup aoc -v --report src/cnn.cl &
nohup aoc -v --report src/cnn.cl -I src/lib -L src -l opencl_lib.aoclib -board=inspur_2bank_a10 &
#nohup aoc -rtl -v --report src/cnn.cl -I src/lib -L src -l opencl_lib.aoclib -board=inspur_2bank_a10 &
#nohup aoc -v --report -DRESNET50 src/cnn.cl &
#nohup aoc -rtl -v --report -DRESNET50 src/cnn.cl &
#nohup aoc -v -c --report -DRESNET50 src/cnn.cl &
