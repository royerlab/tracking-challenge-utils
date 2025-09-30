# Tracking Challenge Utils

Utilities for the whole embryo tracking challenge, providing tools for visualizing and evaluating cell tracking datasets.

## Requirements

- Python >=3.11
- [uv](https://docs.astral.sh/uv/) for dependency management

## Installation

Clone the repository and install dependencies:

```bash
git clone git@github.com:royerlab/tracking-challenge-utils.git
cd tracking-challenge-utils
uv sync
```

## Examples

IMPORTANT: The example requires you to manually change the `DATA_DIR` variable to the path of the dataset you want to use.

### Basic Dataset Visualization

View embryo images and ground truth tracks:

```bash
uv run examples/basic.py
```

This example:
- Loads a tracking dataset
- Displays the image data in napari
- Shows ground truth tracks as both connected tracks and individual points

### Naive Tracking Pipeline

IMPORTANT: This is a proof of concept but not a competitive baseline for the challenge.

Run a complete cell detection and tracking pipeline:

```bash
uv run examples/naive_tracking.py
```

This example demonstrates:
- Naive cell detection using Gaussian filtering and peak detection
- Distance-based cell linking between frames
- Track evaluation against ground truth
- Visualization of detected cells, solution tracks, and ground truth
