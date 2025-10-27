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
- Batch processing of multiple MRC files
- Automatic descriptive output filenames
- Read and write MRC files

## Requirements
Install dependencies with:
```
pip install -r requirements.txt
```

## Usage
### Create a grid image
```bash
python stem_distortion.py --output_grid test_grid.mrc
```

### Process a directory of MRC files
The script can process all MRC files in a directory, automatically generating output filenames that include the distortion parameters:

```bash
python stem_distortion.py --input_dir ./your_directory \
    --x_scale 1.02 \
    --y_scale 1.0 \
    --log_amplitude 5 \
    --log_decay 0.3 \
    --y_scale_factor 2.0
```

Output files will be created in the same directory with descriptive suffixes, for example:
- `input_aniso_x1.02_y1.00.mrc` (for anisotropic scaling)
- `input_log_a5.0_d0.3.mrc` (for logarithmic distortion)
- `input_yscale_2.0000.mrc` (for y-scaling)
- `input_aniso_x1.02_y1.00_log_a5.0_d0.3_yscale_2.0000.mrc` (for combined distortions)

### Example Distortion Combinations

#### Anisotropy only
```bash
python stem_distortion.py --input_dir . \
    --x_scale 1.0 --y_scale 1.02
```

#### Logarithmic distortion only
```bash
python stem_distortion.py --input_dir . \
    --log_amplitude 5 --log_decay 0.3
```

#### Progressive y-scaling only
```bash
python stem_distortion.py --input_dir . \
    --y_scale_factor 2.0  # Right edge shifts 2 pixels outward
```

## Parameters
- `--input_dir`: Directory containing MRC files to process
- `--output_grid`: Output path for generating a test grid
- `--x_scale`, `--y_scale`: Anisotropic scaling factors (unitless, default: 1.0 = no scaling)
  - Values > 1.0 stretch the image content
  - Values < 1.0 compress the image content
- `--log_amplitude`: Maximum pixel displacement for logarithmic distortion (in pixels, default: 0)
  - Controls how far scan lines are shifted in the left quarter of the image
- `--log_decay`: Decay rate of logarithmic distortion (default: 0.5)
  - Controls how quickly the distortion diminishes towards the center
- `--y_scale_factor`: Progressive y-scaling magnitude (in pixels, default: 0)
  - Positive values: inflate the right side (e.g., 2.0 = right edge shifts 2 pixels outward)
  - Negative values: compress the right side (e.g., -2.0 = right edge shifts 2 pixels inward)
  - Effect scales linearly from center to right edge

## Output
All images are saved in MRC format with descriptive filenames indicating the applied distortions and their parameters. Original files are preserved, and new files are created with appropriate suffixes.
