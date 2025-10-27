# STEM Distortion Correction

This project provides tools to simulate and correct scan distortions in micrographs obtained via scanning transmission electron microscopy (STEM).
Measured images are supposed to look like this

<img width="1600" height="1600" alt="grafik" src="https://github.com/user-attachments/assets/98036707-7315-4a57-846e-e90f22027a86" />

## Features
- Create a grid MRC image for visual inspection
- Apply three types of scan distortions:
  1. Anisotropy in x/y
  2. Logarithmic distortion in the left quarter
  3. Progressive y-direction scaling (inflation towards right)
- Read and write MRC files

## Requirements
Install dependencies with:
```
pip install -r requirements.txt
```

## Usage
### Create a grid image
```
python stem_distortion.py --output_grid test_grid.mrc
```

### Apply distortions
#### Anisotropy only
```
python stem_distortion.py --input_mrc test_grid.mrc --output_mrc anisotropic.mrc --x_scale 1.0 --y_scale 1.02 --log_amplitude 0 --y_scale_factor 0
```

#### Logarithmic distortion only
```
python stem_distortion.py --input_mrc test_grid.mrc --output_mrc logarithmic.mrc --x_scale 1.0 --y_scale 1.0 --log_amplitude 5 --log_decay 0.3 --y_scale_factor 0
```

#### Progressive y-scaling only
```
python stem_distortion.py --input_mrc test_grid.mrc --output_mrc y_scaling.mrc --x_scale 1.0 --y_scale 1.0 --log_amplitude 0 --y_scale_factor 0.0005
```

#### Combine distortions
```
python stem_distortion.py --input_mrc test_grid.mrc --output_mrc combined.mrc --x_scale 1.0 --y_scale 1.02 --log_amplitude 5 --log_decay 0.3 --y_scale_factor 0.0005
```

## Parameters
- `--x_scale`, `--y_scale`: Anisotropic scaling factors
- `--log_amplitude`, `--log_decay`: Logarithmic distortion parameters
- `--y_scale_factor`: Progressive y-scaling factor

## Output
All images are saved in MRC format for easy inspection.
