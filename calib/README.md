Place real stereo calibration files in this directory before enabling
`calibration.enabled` in `config.json`.

Expected default file names:

- `left.yaml`: left camera intrinsics and distortion coefficients.
- `right.yaml`: right camera intrinsics and distortion coefficients.
- `stereo.yaml`: stereo `R` and `T`, optionally rectification matrices.

The loader accepts OpenCV YAML/XML or JSON files. Do not use placeholder
matrices for production capture; incorrect calibration can make rectified
preview and epipolar checks misleading.
