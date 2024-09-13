# 💥Instant Uncertainty Calibration of NeRFs Using a Meta-Calibrator (ECCV 2024)

Niki Amini-Naieni, Tomas Jakab, Andrea Vedaldi, Ronald Clark

Official PyTorch implementation for Instant Uncertainty Calibration of NeRFs Using a Meta-Calibrator. Details can be found in the paper, [[Paper]](https://arxiv.org/abs/2312.02350) [[Project page]](https://niki-amini-naieni.github.io/instantcalibration.github.io/).

<img src=img/teaser.png width="50%"/>

## Instructions

- ```get_metacal_data.py``` provides code to get training data for meta-calibrator
- ```extract_features.py``` provides code to extract DINOv2 features from inferred images and uncalibrated uncertainty maps
- ```train_calibrator_all_scenes.py``` provides code to train meta-calibrator on calibration curves from training scenes and predict the calibration curve of a test scene
- ```get_final_metrics.py``` provides code to evaluate the uncertainties and image quality of the uncalibrated and calibrated NeRF

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

