## TF2_auto_config widget 

TF2_auto_config widget as one of utilities of TF2 platform, help users generate TF2 configuration header file automatically.

### Compile and installation

With following steps, tf2_auto_config widget could be ready

```shell
cd /<yourpath>/TF2_auto_config
mkdir build
cd build
cmake ..
make
```

Screenshot below shows us tf2_auto_config widget in your build folder

![TF2_auto_config_content](../../TF2_auto_config_content.png)

### Generate TF2 configuration header file

Execute command to generate your tf2 configuration header file

```shell
./tf2_auto_config /<yourpath>/<yournetworkstructfilename>.bin <yournetworkname>
```

Users could get their TF2 configuration header file in *<yourpath>/TF2_auto_config/**generated_config_file**/*

#### Example

Take CNN network Resnet50 for instance, whose network struct file is already stored in  <yourpath>/TF2_auto_config/examples/resnet50/

```shell
./tf2_auto_config ../examples/resnet50/fpganetwork.bin resnet50
```

After executing command above, corresponding configuration header file is generated in <yourpath>/TF2_auto_config/generated_config_file/ automatically

![TF2_auto_config_result](../../TF2_auto_config_result.png)

