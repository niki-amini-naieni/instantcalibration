# 💥Instant Uncertainty Calibration of NeRFs Using a Meta-Calibrator (ECCV 2024)

Niki Amini-Naieni, Tomas Jakab, Andrea Vedaldi, Ronald Clark

Official PyTorch implementation for Instant Uncertainty Calibration of NeRFs Using a Meta-Calibrator. Details can be found in the paper, [[Paper]](https://arxiv.org/abs/2312.02350) [[Project page]](https://niki-amini-naieni.github.io/instantcalibration.github.io/).

<img src=img/teaser.png width="50%"/>

## Contents
* [Preparation](#preparation)
* [Construct Calibration Curves [Optional]](#construct-calibration-curves-optional)
* [Train Meta-Calibrator](#train-meta-calibrator)
* [Calibrate Uncertainty and Calculate Final Metrics](#calibrate-uncertainty-and-calculate-final-metrics)
* [Citation](#citation)
* [Acknowledgements](#acknowledgements)

## Preparation

### 1. Clone Repository

```
git clone https://github.com/niki-amini-naieni/instantcalibration.git
```

### 2. Download Dataset

Please use [this download link](https://drive.google.com/file/d/1tVCVQTNO0CKRhs9wh1h8siAbIvM2EIhY/view?usp=sharing) for downloading the LLFF dataset from the [NeRF repository](https://github.com/bmild/nerf). Unzip the dataset folder (named ```llff-data.zip```) into the ```instantcalibration``` folder, so that your directory looks like the one below.

```
instantcalibration
--> data
-->--> nerf_llff_data
-->-->--> fern
-->-->--> flower
-->-->--> fortress
...
```

### 3. Set Up Anaconda Environment:

The following commands will create a suitable Anaconda environment for running the code. To produce the results here, we used [Anaconda version 2022.10](https://repo.anaconda.com/archive/Anaconda3-2022.10-Linux-x86_64.sh).

```
conda create -n instantcalibration python=3.7
conda activate instantcalibration
cd instantcalibration
pip install jax==0.2.16
pip install jaxlib==0.1.68+cuda110 -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
pip install -r requirements.txt
pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu117
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
```

### 4. Download Pre-Trained Weights

* Make the ```checkpoints``` directory inside the ```instantcalibration``` folder.

  ```
  mkdir checkpoints
  ```

* Download the zip folder (named ```llff3.zip```) with the pre-trained FlipNeRF checkpoints for 3-view LLFF from [here](https://drive.google.com/file/d/1oD7QpGq0_JawCzf_WTXK4hLxUKXLTErV/view?usp=sharing) and place them into the ```checkpoints``` directory.

  ```
  unzip llff3.zip
  mv llff3 checkpoints
  ```

  Your directory should look like the one below.
  
  ```
  instantcalibration
  --> data
  --> checkpoints
  -->--> llff3
  -->-->--> fern
  -->-->--> flower
  -->-->--> fortress
  ...
  ```

## Construct Calibration Curves [Optional]

To train the meta-calibrator, we used calibration data from 30 scenes. We provide this data in the ```scenes``` folder. We provide example code and instructions on how to generate data for the LLFF scenes below. The calibration curves constructed with this code may not exactly match the ones in the ```scenes``` folder due to the fact that the calibration data for some scenes was sub-sampled for efficiency purposes.

* Make the ```scenes``` directory inside the ```instantcalibration``` repository.

  ```
  mkdir scenes
  ```

* Run the following command to generate data for the ```flower``` scene in LLFF. 

  ```
  python get_metacal_data.py --scene flower --gin_config configs/llff_flower.gin --test_model_dir checkpoints/llff3/flower --output_dir scenes/flower
  ```

* To generate data for the other scenes in LLFF, use the following command, but replace [scene_name] with the name of the scene you would like to generate data for.

  ```
  python get_metacal_data.py --scene [scene_name] --gin_config configs/llff_[scene_name].gin --test_model_dir checkpoints/llff3/[scene_name] --output_dir scenes/[scene_name]
  ```

## Train Meta-Calibrator

* Run the following code to extract the DinoV2 features of the uncalibrated uncertainty maps and predicted NeRF images for all the scenes.

  ```
  python extract_features.py
  ```


* For each scene you would like to calibrate, run the following code, replacing [scene_name] with the name of the scene.

  ```
  python train_calibrator_all_scenes.py --test_scene [scene_name]
  ```

## Calibrate Uncertainty and Calculate Final Metrics
To get the final metrics in the paper, for each scene you would like to test, run the following code, replacing [scene_name] with the name of the scene.

```
python get_final_metrics.py --scene [scene_name] --gin_config configs/llff_[scene_name].gin --test_model_dir checkpoints/llff3/[scene_name]
```

Below, we have provided a table showing the results for each scene that you can compare to. We reproduced the results from the main paper below after refactoring our code. These results are slightly different from the ones in the main paper because of the random sampling in the meta-calibrator training code. You will likely find slightly different (but very close) results to the ones below and the ones in the main paper if you have followed all previous steps correctly.

| Scene    | PSNR  | LPIPS | Cal Err (Uncal) RGB Avg. | Cal Err (Meta-Cal) RGB Avg. | NLL (Uncal) | NLL (Meta-Cal) |
|----------|-------|-------|--------------------------|-----------------------------|-------------|----------------|
| Flower   | 20.25 | 0.216 | 0.0112                   | 0.0012                      | -0.10       | -0.09          |
| Room     | 20.19 | 0.222 | 0.0245                   | 0.0140                      | 0.52        | 0.12           |
| Orchids  | 16.10 | 0.225 | 0.0070                   | 0.0004                      | -0.23       | -0.44          |
| Trex     | 20.39 | 0.185 | 0.0097                   | 0.0005                      | -0.33       | -0.48          |
| Leaves   | 16.19 | 0.217 | 0.0024                   | 0.0042                      | -0.72       | -0.68          |
| Horns    | 17.79 | 0.304 | 0.0085                   | 0.0017                      | -0.59       | -0.84          |
| Fortress | 23.19 | 0.236 | 0.0013                   | 0.0039                      | -1.30       | -1.33          |
| Fern     | 20.59 | 0.277 | 0.0041                   | 0.0004                      | -1.34       | -1.39          |
| **Average** | **19.38** | **0.235** | **0.0086**                   | **0.0033**                      | **-0.51**       | **-0.64**          |


## Citation

```
@inproceedings{AminiNaieni24,
    title={Instant Uncertainty Calibration of {NeRFs} Using a Meta-Calibrator},
    author={Niki Amini-Naieni and Tomas Jakab and Andrea Vedaldi and Ronald Clark},
    booktitle={ECCV},
    year={2024}
}
```

### Acknowledgements

This repository uses code from the [FlipNeRF repository](https://github.com/shawn615/FlipNeRF). 

