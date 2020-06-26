# Run example

Steps to run the example

## Download Matterport3D data for aihabitat

Follow instructions [here](../README.md#datasets)

``` shellsession
$ export MATTERPORT3D_PATH_DIR=<location of downloaded Matter port dataset>
```


## Download DEDUCE pretrained models

Follow instructions [here](https://github.com/anwesanpal/DEDUCE)

Or directly download `resnet18_best_home.pth.tar` from [here](https://drive.google.com/open?id=1EVnOGJXBn4wo5V5eez4JsCxFs08fQUU_) and put it in the `deduce/models/` directory.

## Run script

``` shellsession
$ export MATTERPORT3D_PATH_DIR=<location of downloaded Matter port dataset>
$ python examples/example.py --scene $MATTERPORT3D_PATH_DIR/v1/tasks/mp3d/gTV8FGcVJC9/gTV8FGcVJC9.glb --save_png
```
