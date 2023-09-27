# acRNN
This folder contains an implementation of acRNN for the CMU motion database written in Pytorch.


### Preparation

```
conda create --name acrnn python3.8
conda activate acrnn
pip install -r requirements.txt
```

### Data

Download the motion data from the CMU motion database in the form of bvh files [here](https://sites.google.com/a/cgspeed.com/cgspeed/motion-capture/the-motionbuilder-friendly-bvh-conversion-release-of-cmus-motion-capture-database) or [here](http://mocap.cs.cmu.edu/). Transform to training data by

```
python3 position2dual.py source_bvh_folder
```

### Training

```
python3 train_model.py converted_data_folder
```
