#nohup aoc -v -report src/cnn.cl &
nohup aoc -v -report -DRESNET50 src/cnn.cl -I src/lib -L src -l src/opencl_lib.aoclib -board=inspur_2bank_a10 -v -report &
#nohup aoc -v -report -DRESNET50 src/cnn.cl -dont-error-if-large-area-est &
#nohup aoc -v -report -DRESNET50_PRUNED src/cnn.cl &
#nohup aoc -v -c -report -DRESNET50 src/cnn.cl &
