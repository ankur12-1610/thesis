# Self-Prompting Polyp Segmentation in Colonoscopy Using Hybrid YOLO-SAM 2 Model

## Introduction
Colorectal Cancer (CRC) is a leading cause of cancer-related deaths globally. Early detection and removal of polyps during colonoscopy can significantly reduce CRC incidence and mortality. However, challenges such as variability in polyp characteristics and artifacts in colonoscopy images/videos hinder accurate segmentation.

### Objective
This research proposes a novel polyp segmentation method by combining YOLOv8 and SAM 2 models:
- **YOLOv8**: Detects polyps using bounding boxes.
- **SAM 2**: Performs precise segmentation using YOLOv8-generated bounding box prompts.

The method reduces manual annotation efforts by leveraging self-prompting mechanisms.

---

## Methodology
### Model Architecture
1. **YOLOv8**:
   - Detects polyps and generates bounding box predictions.
   - Fine-tuned using bounding box annotations.

2. **SAM 2**:
   - Performs segmentation based on YOLOv8 bounding box prompts.
   - Components:
     - **Image Encoder**: Extracts high-level features from images/videos.
     - **Prompt Encoder**: Processes bounding box prompts for segmentation.
     - **Memory Mechanism**: Tracks objects across video frames for consistent segmentation.
     - **Mask Decoder**: Generates final segmentation masks.

### Key Features
- **Training Strategy**:
  - Only YOLOv8 is fine-tuned; SAM 2 uses frozen weights for zero-shot capabilities.
- **Annotation Efficiency**:
  - Uses bounding box data instead of detailed segmentation masks, reducing annotation time.
- **Real-Time Processing**:
  - Lightweight architecture enables real-time video segmentation.

---

## Datasets
The model was evaluated on five image datasets and two video datasets:

### Image Datasets
1. **Kvasir-SEG**: 1,000 polyp images with ground truth annotations.
2. **CVC-ClinicDB**: 612 images from colonoscopy videos.
3. **CVC-ColonDB**: 380 polyp images from 15 videos.
4. **ETIS**: 196 high-resolution polyp images.
5. **CVC-300**: 60 polyp images.

### Video Datasets
1. **PolypGen**: Includes positive/negative frames and video sequences from multiple medical centers.
2. **SUN-SEG**: Contains over 158,000 frames from colonoscopy videos with detailed annotations.

---

## Results
### Quantitative Evaluation
The model achieved state-of-the-art performance across multiple datasets:
| Dataset       | Metric | YOLO-SAM | YOLO-SAM 2 | Improvement |
|---------------|--------|----------|------------|-------------|
| CVC-ColonDB   | mIoU   | 0.808    | **0.848**  | +9.8%       |
| PolypGen      | mIoU   | 0.678    | **0.904**  | +20.7%      |
| SUN-SEG Easy  | Dice   | 0.945    | **0.958**  | +1.3%       |

### Qualitative Evaluation
Visual results show that YOLO-SAM 2 produces segmentation masks closely matching ground truth annotations, outperforming previous methods.

---

## Comparison with State-of-the-Art Models
| Model          | mIoU (CVC-ColonDB) | mDice (ETIS) | Training Annotations |
|-----------------|--------------------|--------------|-----------------------|
| UNet           | 0.436              | 0.398        | Detailed masks        |
| Polyp-PVT      | 0.727              | 0.787        | Detailed masks        |
| YOLO-SAM       | 0.808              | 0.933        | Bounding boxes        |
| YOLO-SAM 2     | **0.848**          | **0.949**    | Bounding boxes        |

---

## Conclusions
The proposed YOLO-SAM 2 model demonstrates superior performance in polyp segmentation tasks with reduced annotation requirements.

### Key Contributions:
1. Integration of YOLOv8 bounding box predictions with SAM 2's zero-shot segmentation capabilities.
2. Real-time processing capability suitable for clinical deployment.

### Future Directions:
- Optimize the model further for clinical applications.
- Explore its application in other medical imaging tasks.

