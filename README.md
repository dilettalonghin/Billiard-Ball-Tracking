# Billiard Ball Tracking

Billiard Ball Tracking is a C++ and OpenCV project for analysing billiard-shot videos. It detects the playing field and balls in the first and last frames of a clip, classifies the balls, produces segmentation masks, and tracks the detected balls throughout the video.

## Features

- playing-field detection using edge detection and the Hough transform;
- ball detection using colour preprocessing and the Hough circle transform;
- classification of cue, eight, solid-colour, and striped balls;
- coloured semantic-mask generation;
- ball tracking across video frames;
- evaluation with mAP at IoU 0.5 and mean IoU.

## Requirements

- a C++11-compatible compiler;
- CMake 3.10 or newer;
- OpenCV with the tracking module available.

On macOS with Homebrew, OpenCV and CMake can be installed with:

```bash
brew install cmake opencv
```

## Build

From the project root:

```bash
cmake -S . -B build
cmake --build build
```

This creates two executables:

- `build/BilliardVision`, which runs the vision and tracking pipeline;
- `build/Metrics`, which evaluates the generated results against the annotations.

## Dataset layout

The dataset is intentionally not included in the repository. Each clip directory passed to the programs must have the following structure:

```text
data/
└── gameX_clipY/
    ├── gameX_clipY.mp4
    ├── frames/
    │   ├── frame_first.png
    │   └── frame_last.png
    ├── bounding_boxes/
    │   └── frame_last_bbox.txt
    ├── masks/
    │   └── frame_last.png
    └── output/
```

Bounding-box annotation rows use the format:

```text
x y width height category_id
```

Category IDs are:

| ID | Category |
|---:|---|
| 0 | Background |
| 1 | Cue ball |
| 2 | Eight ball |
| 3 | Solid-colour ball |
| 4 | Striped ball |
| 5 | Playing field |

## Usage

Run the analysis pipeline by passing a clip directory:

```bash
./build/BilliardVision data/game1_clip1
```

The program writes bounding boxes, visualisations, segmentation masks, and selected tracking frames to the clip's `output/` directory. It also opens result windows, so a graphical desktop session is required.

After generating the outputs, calculate the evaluation metrics with:

```bash
./build/Metrics data/game1_clip1
```

## Project structure

```text
.
├── CMakeLists.txt
├── include/        # Public headers and data structures
└── src/            # Detection, classification, segmentation, tracking, and metrics
```

The local `data/` directory and `Copia di REPORT.pdf` are excluded through `.gitignore` because they are large, local project assets.
