# Experiment 2 - Stage 1 (Updated with Experiment 1 Data Ingestion)

## 🎯 What's New?

Experiment 2 Stage 1 now uses **Experiment 1's superior data ingestion process**:
- ✅ Loads videos directly from MP4 files (not pre-extracted frames)
- ✅ Uses OpenCV for efficient video loading
- ✅ Includes attention mechanism to identify crime-relevant frames
- ✅ Gradient clipping for stable training
- ✅ Automatically prepares data for Stage 2

---

## 📋 Prerequisites

### 1. Dependencies
```bash
pip install torch torchvision opencv-python pillow numpy tqdm matplotlib
```

### 2. Data Structure
Your video directory should look like this:

```
ucf_crime_v2/
├── Train/
│   ├── NormalVideos/
│   │   ├── Normal_Videos001_x264.mp4
│   │   ├── Normal_Videos002_x264.mp4
│   │   └── ...
│   ├── Abuse/
│   │   ├── Abuse001_x264.mp4
│   │   └── ...
│   ├── Arrest/
│   ├── Arson/
│   ├── Assault/
│   ├── Burglary/
│   ├── Explosion/
│   ├── Fighting/
│   ├── RoadAccidents/
│   ├── Robbery/
│   ├── Shooting/
│   ├── Shoplifting/
│   ├── Stealing/
│   └── Vandalism/
├── Test/
│   └── (same structure)
└── Validation/
    └── (same structure)
```

**Important**: The dataset expects `.mp4` files, not directories of frames!

---

## 🚀 How to Run

### Step 1: Configure Paths

Open `experiment_2_frfr.ipynb` and update Cell 4:

```python
# UPDATE THESE PATHS TO YOUR VIDEO DIRECTORIES
train_dir = r"YOUR_PATH/ucf_crime_v2/Train"
test_dir  = r"YOUR_PATH/ucf_crime_v2/Test"
```

### Step 2: Run Training

Execute cells in order:
1. **Cell 0**: Overview (markdown)
2. **Cell 1**: Import libraries
3. **Cell 2**: Dataset class definition
4. **Cell 3**: Model with attention mechanism
5. **Cell 4**: Load data and create dataloaders
6. **Cell 5-6**: Verify data distribution (optional but recommended)
7. **Cell 7-8**: Train model with gradient clipping

**Expected Output**:
```
Epoch 1 | Loss: 0.6XXX | Accuracy: XX.XX% | LR: 0.000100
Epoch 2 | Loss: 0.5XXX | Accuracy: XX.XX% | LR: 0.000XXX
...
✓ Model checkpoint saved to: .../binary_stage1_with_attention.pt
```

### Step 3: Run Inference (Stage 1 Output)

Execute cells:
- **Cell 9-10**: Extract anomalies with attention weights

**Expected Output**:
```
Stage 1 Inference: 100%|██████████| 145/145 [XX:XX<XX:XX]

============================================================
✓ Stage 1 Complete!
  - Detected XXX crime clips
  - Anomaly records saved to: ./stage1_output/anomalies.json
  - Attention data saved to: ./stage1_output/attention_data/stage1_attention_data.pt
  - Ready for Stage 2 training
============================================================
```

### Step 4: Visualize Attention (Optional)

Execute cells:
- **Cell 11-12**: Generate attention weight visualizations

**Output**: `./stage1_output/attention_visualization.png`

---

## 📊 Output Files

After running the notebook, you'll have:

### 1. Model Checkpoint
**Location**: `C:\Users\rayaa\Downloads\ucf_crime_v2\checkpoints\binary_stage1_with_attention.pt`

**Contents**:
```python
{
    'epoch': 5,
    'model_state_dict': ...,
    'optimizer_state_dict': ...,
    'scheduler_state_dict': ...
}
```

### 2. Anomaly Detection Results
**Location**: `./stage1_output/anomalies.json`

**Format**:
```json
[
  {
    "video_path": "path/to/video.mp4",
    "confidence": 0.987,
    "attention_weights": [0.05, 0.12, ..., 0.89],
    "top_attention_indices": [12, 15, 8, 10, 14]
  },
  ...
]
```

### 3. Stage 2 Training Data
**Location**: `./stage1_output/attention_data/stage1_attention_data.pt`

**Contents**: PyTorch tensor file with:
- Video paths
- Confidence scores
- Full attention weight arrays
- Clip tensors (for Stage 2 training)

### 4. Attention Visualization
**Location**: `./stage1_output/attention_visualization.png`

Shows bar plots of attention weights for detected crime clips.

---

## 🔧 Configuration Options

### Adjust Hyperparameters

In Cell 8, you can modify:

```python
EPOCHS = 5                    # Number of training epochs
CLIP_LEN = 16                 # Number of frames per video
FRAME_SIZE = 112              # Frame resolution (HxW)
GRADIENT_CLIP_VALUE = 1.0     # Max gradient norm
batch_size = 4                # Training batch size (in loader)
learning_rate = 1e-4          # Initial learning rate
```

### Customize Transforms

In Cell 4, modify the transform pipeline:

```python
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                        std=[0.229, 0.224, 0.225])
])
```

---

## 🐛 Troubleshooting

### Problem: "Could not open video"
**Solution**: Ensure videos are in `.mp4` format and not corrupted. Test with:
```python
import cv2
cap = cv2.VideoCapture("your_video.mp4")
print(f"Opened: {cap.isOpened()}")
```

### Problem: "CUDA out of memory"
**Solution**: Reduce batch size in Cell 4:
```python
train_loader = DataLoader(train_data, batch_size=2, ...)  # Reduce from 4 to 2
test_loader = DataLoader(test_data, batch_size=1, ...)    # Reduce from 2 to 1
```

### Problem: "No module named 'cv2'"
**Solution**: Install OpenCV:
```bash
pip install opencv-python
```

### Problem: Slow training on CPU
**Solution**: Training on GPU is highly recommended. Check:
```python
print(torch.cuda.is_available())  # Should print: True
print(torch.cuda.get_device_name(0))  # Should show your GPU
```

---

## 📈 Performance Expectations

### Training Time (per epoch):
- **GPU (RTX 3090)**: ~3-5 minutes
- **GPU (GTX 1080)**: ~8-12 minutes
- **CPU**: ~45-60 minutes (not recommended)

### Memory Usage:
- **GPU Memory**: ~4-6 GB
- **RAM**: ~8-12 GB

### Expected Accuracy:
- **After 5 epochs**: 60-75%
- **After 10 epochs**: 75-85%
- **After 20 epochs**: 85-92%

---

## 🎓 Understanding the Model

### Model Architecture

```
Input: (Batch, 3, 16, 112, 112)
  ↓
3D Convolutions (spatial + temporal)
  ↓
Attention Mechanism (identifies important frames)
  ↓
Weighted Features
  ↓
Classifier (2 classes: Normal=0, Crime=1)
  ↓
Output: Logits (B, 2) + Attention Weights (B, T)
```

### Attention Mechanism

The attention module learns to assign importance weights to each frame:
- **High attention weight** (close to 1.0): Frame is important for crime detection
- **Low attention weight** (close to 0.0): Frame is less relevant

These weights are used to:
1. Improve classification accuracy
2. Identify which frames to extract for Stage 2
3. Provide model interpretability

---

## 🔄 Next Steps: Stage 2

With Stage 1 complete, you can now proceed to Stage 2:

1. **Load Stage 1 attention data**:
   ```python
   attention_data = torch.load('./stage1_output/attention_data/stage1_attention_data.pt')
   ```

2. **Extract high-attention frames**: Use the attention weights to identify the most relevant frames from crime videos

3. **Train Stage 2 model**: Detailed anomaly type classification (13 crime categories)

4. **Use same data ingestion**: Continue using the OpenCV-based video loading approach

---

## 📚 Additional Resources

- **Original Experiment 1**: See `models/sandbox/experiment_1/brandon_improved_experiment1.ipynb`
- **Stage 2 Reference**: See `models/sandbox/experiment_2/rayaan/stage1.ipynb`
- **UCF Crime Dataset**: [Link to dataset info](../../info/dataset.txt)

---

## ✅ Checklist

Before running the notebook:
- [ ] Videos are in `.mp4` format
- [ ] Directory structure matches expected format
- [ ] Paths in Cell 4 are updated to your data location
- [ ] GPU is available (check `torch.cuda.is_available()`)
- [ ] All dependencies are installed

After running the notebook:
- [ ] Training completed successfully (5 epochs)
- [ ] Model checkpoint saved
- [ ] Anomalies detected and saved
- [ ] Attention data ready for Stage 2
- [ ] Visualization generated (optional)

---

## 💬 Questions?

If you encounter issues:
1. Check the troubleshooting section above
2. Verify your data structure matches the expected format
3. Ensure all paths are correct
4. Review the comparison document: `QUICK_COMPARISON.md`

**Happy training! 🚀**

