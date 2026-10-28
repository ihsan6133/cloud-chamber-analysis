# cloud-chamber-analysis

A deep learning application to analyse and extract features from cloud chamber footage.

The objective is to detect the particle trails and categorize their types and perform quantitative analysis.



## Cloud chamber video preprocessing.

Pipeline per frame:
  1. Grayscale (+ optional crop to the chamber area)
  2. Subtract a static background (median of sampled frames) -> removes
     fixed clutter: glare, edges, scratches, lamp gradients
  3. Light Gaussian blur -> suppresses sensor noise
  4. White top-hat -> removes broad slowly-varying haze / fog glow
  5. Hysteresis threshold + connected-component filtering -> removes specks
     and keeps only elongated, track-like structures
  6. Contrast normalisation -> clean, consistent grayscale for the ML model

## Outputs:
  `out/cleaned.mp4`   grayscale cleaned footage (track-enhanced)
  
  `out/mask.mp4`      binary track mask
  
  `out/preview.png`   original | cleaned | mask comparison of a sample frame
  
  `out/frames/*.png`  (optional, --save-frames) lossless frames for training


## Installation  

```sh
pip install opencv-python numpy
```

## Usage:  

### Preprocessing
```sh
python preprocess_chamber.py video.webm --out out
```

"""