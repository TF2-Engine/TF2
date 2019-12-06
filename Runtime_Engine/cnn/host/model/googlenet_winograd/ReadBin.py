import struct
count = 0
with open("GoogLeNetWinogradParam.bin","rb") as read_bin:
   #""" 
    while(read_bin.read(4)):
        count = count + 1
        if count == 6623962: #6624962:
            break
    #"""
    for i in range(1000):
        data = read_bin.read(4)
        count = count + 1
        data_float = struct.unpack("f",data)[0]
        print(data_float)
        #feat_layer.append(data_float)
    #"""
print(count)
