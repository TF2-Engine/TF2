import caffe_pb2 as caffe_pb2


def analysis_model(src_binary_model_layers):#,dst_text_net,dst_binary_model):
  layers = src_binary_model_layers
  layer_id = -1
  layerid_bottoms_list = []
  #step1:find eltwise or concat rules: layer have multi bottoms(input)
  for layer in layers:
    layer_id = layer_id + 1
    if len(layer.bottom) >1:
      layerid_bottoms_list.append(layer_id)
  ###########step2:find split layer####################
  #find topname
  topname_list = []
  for layer in layers:
    if len(layer.top)>0:
      if len(layer.bottom) ==1 and len(layer.top)==1 and layer.bottom == layer.top:
        continue
      else:
        if len(topname_list)>0 and layer.top[0] in topname_list:
          continue
        else:
          topname_list.append(layer.top[0])
  
  #find split layer
  splitlayer_toplayers = {}
  for topname in topname_list:
    lianjielayerid = []
    layer_id = -1
    for layer in layers:
      layer_id+=1
      if len(layer.top) ==1 and layer.top[0] == topname:
        splitlayerid = layer_id
      if len(layer.bottom) ==1 and len(layer.top)==1 and layer.bottom == layer.top:
        continue
      else:
        if len(layer.bottom)>0:
          for bottom in layer.bottom:
            if bottom == topname:
              #print("lianjie:",layers[splitlayerid].name,layers[layer_id].name)
              lianjielayerid.append(layer_id)
    if len(lianjielayerid) > 1:
      splitlayer_toplayers[splitlayerid] = lianjielayerid
      #print("split:",layers[splitlayerid].name,lianjielayerid)
  ###only for print
  '''splitlayernum = -1
  for splitlayerid,toplayerids in splitlayer_toplayers.items():
    splitlayernum += 1
    print(splitlayernum,layers[splitlayerid].name)
    for toplayerid in toplayerids:
      print(layers[toplayerid].name)'''
  ###########step3:generate new layer paixu ####################
  if len(splitlayer_toplayers) != len(layerid_bottoms_list):
    print("error")
  splitlayernum = -1
  splitlayer_opnums = {}
  for splitlayerid,toplayerids in splitlayer_toplayers.items():
    splitlayernum += 1
    opnums_list = []
    for topindex in list(range(len(toplayerids)-1)):
      opnums_list.append(toplayerids[topindex+1] - toplayerids[topindex])
    opnums_list.append(layerid_bottoms_list[splitlayernum] - toplayerids[len(toplayerids)-1])
    #print(opnums_list)
    splitlayer_opnums[splitlayerid] = opnums_list
  caffe_net = []
  multiinput_layerbottoms=[]
  layer_id = 0
  for splitlayerid,toplayerids in splitlayer_toplayers.items():
    if layer_id == splitlayerid-1:  #one:change eltwise bottom shunxu copy eltwise(concat) layers
      caffe_net.append(layers[layer_id])
      print(caffe_net[layer_id].bottom)
      for bottomId in range(len(layers[layer_id].bottom)):
        caffe_net[layer_id].bottom[bottomId] = multiinput_layerbottoms[bottomId]
      print(caffe_net[layer_id].bottom)
      layer_id+=1

    if layer_id < toplayerids[0]: #two:copy layers except one three four 
      for index in list(range(layer_id,toplayerids[0])):
        caffe_net.append(layers[index])
        layer_id +=1
    #three:copy layers that between split layer and eltwise layer
    zip_toplayer_opnums = zip(splitlayer_opnums[splitlayerid],toplayerids)
    zip_toplayer_opnums = sorted(zip_toplayer_opnums,reverse=False)
    print(zip_toplayer_opnums)
    multiinput_layerbottoms=[]
    for opnums,toplayerid in zip_toplayer_opnums:
        for index in list(range(toplayerid,toplayerid+opnums)):
          caffe_net.append(layers[index])
        if opnums == 0:
          index = layer_id
        multiinput_layerbottoms.append(layers[index].bottom[0])
    layer_id +=sum(splitlayer_opnums[splitlayerid])
  
    #print(layer_id,splitlayerid,caffe_net[layer_id-1].name,layers[layer_id].name)
  #four: copy last eltwise 
  if len(layers[layer_id].bottom) == len(multiinput_layerbottoms):  #one:change eltwise bottom shunxu copy eltwise(concat) layers
      caffe_net.append(layers[layer_id])
      print(caffe_net[layer_id].bottom)
      for bottomId in range(len(layers[layer_id].bottom)):
        caffe_net[layer_id].bottom[bottomId] = multiinput_layerbottoms[bottomId]
      print(caffe_net[layer_id].bottom)
      layer_id+=1
  #five: copy layers after last eltwise
  if layer_id < len(layers):
    for index in list(range(layer_id,len(layers))):
        caffe_net.append(layers[index])
  return caffe_net