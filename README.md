# Video-Based Human Fall Detection Using Deep Learning

A computer vision project that detects potential human falls from recorded video using YOLOv8m, geometric analysis, motion tracking, and a multi-level risk scoring mechanism.

## Project Overview

Falls can pose serious safety risks, particularly for elderly individuals and people with mobility impairments. This project explores a camera-based fall detection approach that does not require wearable sensors.

The system combines YOLOv8m object detection with bounding-box aspect ratio analysis and frame-to-frame motion analysis to distinguish potential falls from normal human activities. A risk score ranging from 0 to 3 provides an additional indication of event severity.

## Objectives

- Develop a video-based human fall detection pipeline using deep learning.
- Build a custom dataset from publicly available YouTube videos.
- Analyze bounding-box geometry and vertical displacement across consecutive frames.
- Reduce false positives by combining visual and motion-based cues.
- Assign risk levels to detected events for severity-aware monitoring.

## Technologies Used

- **Language:** Python
- **Environment:** Google Colab
- **Deep Learning:** YOLOv8m
- **Computer Vision:** OpenCV
- **Data Processing:** NumPy and other libraries used in the notebook

## System Architecture

The system follows a three-layer pipeline:

1. **Input layer:** Load recorded video and extract frames at fixed intervals.
2. **Processing layer:** Preprocess frames, perform YOLOv8m detection, and analyze bounding-box geometry and motion.
3. **Output layer:** Display the predicted activity, detection information, and associated risk score.

### Workflow

```text
Recorded Video
      |
      v
Frame Extraction
      |
      v
Preprocessing
      |
      v
YOLOv8m Detection
      |
      v
Bounding-Box and Motion Analysis
      |
      v
Fall Assessment and Risk Scoring
      |
      v
Annotated Detection Output
```

## Dataset

A custom dataset was developed using video clips collected from publicly available YouTube videos.

The dataset includes fall events and normal activities, such as walking, sitting, bending, and lying down. Frames were extracted and annotated using the YOLO format.

**Dataset characteristics:**
- Source: YouTube video clips
- Classes: Fall and Normal
- Annotation format: YOLO
- Environments: Indoor and outdoor scenes with varying lighting and camera angles
- Preparation: Frame extraction, resizing, annotation, and dataset splitting

The dataset and its annotations are not included in this repository by default. Access to the data may be required to reproduce training.

## Model and Training

The project uses **YOLOv8m**, a single-stage object detection model, fine-tuned on the custom fall detection dataset.

A two-stage training strategy was used:

| Parameter | Stage 1 | Stage 2 |
|---|---:|---:|
| Epochs | 40 | 20 |
| Image size | 640 × 640 | 960 × 960 |
| Batch size | 8 | 4 |
| Learning rate | 0.001 | 0.001 |
| Purpose | Initial feature learning | Higher-resolution fine-tuning |

The Adam optimizer was used, and validation mAP was monitored for early stopping, as described in the project report.

## Fall Detection Methodology

### 1. YOLOv8m Detection
Each video frame is passed through the trained model to obtain detection information, including bounding boxes and confidence scores.

### 2. Bounding-Box Aspect Ratio
The bounding-box height-to-width ratio is examined as a geometric cue. A more horizontally oriented bounding box can indicate a potential fall, although lying down and other normal activities can produce similar shapes.

### 3. Motion Analysis
The vertical displacement of the bounding-box center is tracked across consecutive frames. Sudden movement can provide additional evidence of a potential fall.

### 4. Combined Assessment
The geometric and motion-based signals are combined with detection information to distinguish potential falls from normal activities.

## Risk Scoring

The system uses a risk score from 0 to 3.

| Score | Risk level | Interpretation |
|---:|---|---|
| 0 | No Risk | Normal human activity |
| 1 | Low Risk | Minor posture deviation |
| 2 | Medium Risk | Suspicious motion or potential partial fall |
| 3 | High Risk | High-confidence fall indication |

The report describes the score as a weighted combination of detection confidence, bounding-box aspect ratio deviation, and vertical motion speed. Exact weights and thresholds should be taken from the implementation.

## Results and Discussion

The project report describes the following qualitative findings:

- YOLOv8m detected people under varied lighting conditions and camera angles in the evaluated examples.
- Aspect-ratio analysis helped identify horizontally oriented postures but could produce false positives for people lying on furniture.
- Motion analysis provided additional information for distinguishing sudden falls from gradual movements.
- Higher-resolution fine-tuning was intended to improve the detection of distant subjects.
- The risk score provided multiple levels of indication rather than a single binary output.

The report also identifies limitations involving severe occlusion, overlapping bounding boxes, multi-person scenes, and dataset size.

Numerical accuracy, precision, recall, mAP, and inference speed should be added only when supported by the actual evaluation results.

## Applications

Potential application areas include:
- Elderly care facilities and hospitals
- Smart-home safety monitoring
- Rehabilitation environments
- Public-space safety monitoring

These are potential applications; the project is an experimental system and is not a clinically validated safety solution.

## Limitations

- Sensitivity to camera angle and bounding-box geometry
- Reduced reliability under severe occlusion
- Difficulty handling multiple people and overlapping detections
- Limited dataset scale and coverage of edge cases
- Recorded-video evaluation in the current project version

## Future Work

- Expand the dataset with more diverse fall and normal-activity examples.
- Integrate pose estimation for improved posture analysis.
- Improve multi-person tracking.
- Explore real-time alerting through SMS or IoT integrations.
- Investigate deployment on edge devices such as Raspberry Pi or NVIDIA Jetson.

## How to Run

1. Open `fall_detection.ipynb` in Google Colab.
2. Review the installation and configuration cells.
3. Configure access to the required dataset and trained model weights.
4. Run the notebook cells in order.
5. Inspect the annotated video or frame outputs and available evaluation results.

The notebook may require changes to file paths or access settings depending on where the dataset and model weights are stored.

## Team

- Deepali Banka
- Soumistha Bhuyan
- Shruti Sameeksha Pradhan

## References

1. J. Redmon et al., “You Only Look Once: Unified, Real-Time Object Detection,” CVPR, 2016.
2. Ultralytics, [YOLOv8 Documentation](https://docs.ultralytics.com/).
3. S. Zhang et al., “Vision-Based Fall Detection Using Deep Learning,” IEEE Access, 2020.
4. M. Mubashir et al., “A Survey on Fall Detection: Principles and Approaches,” Neurocomputing, 2013.
5. N. Noury et al., “Fall Detection — Principles and Methods,” Engineering in Medicine and Biology Society, 2007.

## Disclaimer

This project was developed for academic and research purposes. Its predictions should not be treated as a substitute for professional medical assessment or a validated emergency monitoring system.
