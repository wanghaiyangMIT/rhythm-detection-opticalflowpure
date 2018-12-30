
## Rhythm-detection-noshift
Rhythm-detection-noshift is a sample version for video rhythm of our CV-Class project. It's based on optical flow, which as a feature inputed to our model.We have ever tried some features such as **pose** from [Alpha Pose](http://www.mvig.org/research/alphapose.html), **scene change** from [Pyscenedetectlib](https://github.com/Breakthrough/PySceneDetect),**optical flow** from [gpu-flow](https://github.com/feichtenhofer/gpu_flow)based on opencv-c++ .**Our dataset** is a MV with a strong rhythm that was manually downloaded from the music station. As for the **label** ,we use [librosa](https://github.com/librosa/librosa) to generate rhythm strength.  
## Rhythm-deteciton-shiftlayer
our code provide shiftlayer version with attention in modelsshift.py ,if you wangt to use it ,you need change it in train.py , open it with --shift_with_attention, and import **modelsshift** ,and change the models in main function with modelsshift, i think it will be ok.

## Contents
- [Rhythm-detection-noshift](rhythm-detection-noshift)
- [Rhythm-detection-shiftlayer](#rhythm-detection-shiftlayer)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Output](#output)
- [Contributors](#contributors)
- [Citation](#citation)
- [License](#license)

## Installation
1. Get the code and build related modules.
  ```Shell
  git clone https://github.com/wanghaiyangMIT/rhythm-detection-opticalflowpure.git
  ```
2. Install [Torch](https://github.com/torch/distro)(verson = 0.4.0).If you complete the label and feature extraction,you can installed the lib.
  ```Shell
  python 3.6
  opencv 3.4.1.15
  numpy  1.13.1
  scipy
  json   
  os
  random
  tqdm
  ```
## Quick Start
- **Rhythm-detection-noshiftlayer**:  Run Rhythm-detection-shiftlayer for optical flow feature extracted by our code (extract frames 4fps and each has a optical flow 224x224x3) for features in a folder ,but the feature organized not fixed, so you need to read the dataloader to organize your own dataset,there we provide a video with no shift basic version, you can organized your optical flow , scene change, pose feature dataset.If you want to use shiftlayer version, you can change the import models whith modelsshift,main function also need change the models as modelsshift.Basic version as follows: 
```
python train.py --audio_dir --video_dir --save_dir --device  
```
you can see the options in train.py, if you meet some problem ,please contact me ,because the data preparetion is complexed.


## Output
Output each frame whether a click

## Contributors
Authored by [Yu-Tong Xie](https://github.com/xxxxxyt/), [Hai-Yang Wang](https://github.com/wanghaiyangMIT/) and [Yan Hao](https://github.com/honeyhaoyan/), [Zi-Hao Xu](https://github.com/shsjxzh/) Currently, it is developed and maintained by [Hai-Yang Wang](https://github.com/wanghaiyangMIT/)

## Citation
Please cite these papers in your publications if it helps your research:

## License
rhythm-detection is freely available for free non-commercial use, and may be redistributed under these conditions. For commercial queries, contact [Hai-Yang Wang](https://github.com/wanghaiyangMIT/)
