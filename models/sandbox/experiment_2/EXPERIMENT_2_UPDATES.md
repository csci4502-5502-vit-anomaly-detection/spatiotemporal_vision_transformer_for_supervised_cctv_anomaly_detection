# Experiment 2 - Stage 1 Updates

## Summary of Changes

This document describes the updates made to Experiment 2's Stage 1 notebook (`experiment_2_frfr.ipynb`) to use the superior data ingestion process from Experiment 1.

---

## Key Changes Made

### 1. **Data Ingestion - Now Uses Experiment 1's Approach**

#### Before:
- Expected pre-extracted frames stored as PNG files in directories
- Used `PIL.Image.open()` to load individual frame images
- Required separate frame extraction preprocessing step

#### After:
- **Loads videos directly from MP4 files** (like Experiment 1)
- Uses **OpenCV (`cv2.VideoCapture`)** for efficient video loading
- Extracts frames on-the-fly during training
- Uniform temporal sampling across video duration

**Benefits:**
- No need for pre-extracted frame directories
- More flexible and memory efficient
- Consistent with Experiment 1's proven approach
- Works on beefier machines with better performance

---

### 2. **Enhanced Dataset Class**

```python
class UCFCrimeBinaryDataset(Dataset):
    - Added frame_size parameter for consistent resizing
    - Added _load_video_cv2() method for OpenCV video loading
    - Samples frames uniformly using np.linspace()
    - Handles video loading errors gracefully with dummy tensors
    - Automatic padding for short videos
```

**Key Features:**
- Direct MP4 loading from directory structure
- Automatic BGR → RGB conversion
- On-the-fly resizing to target frame size
- Error handling for corrupted videos

---

### 3. **Attention Mechanism Added**

The model now includes an attention mechanism to identify which frames are most important for crime detection (as per Stage 1 goals).

```python
class BinaryCrimeDetector(nn.Module):
    - Added attention module (Conv3d layers + Sigmoid)
    - Attention weights range from 0 to 1
    - Returns both predictions and attention weights
    - Weighted features for better classification
```

**Purpose:**
- Identify crime-relevant frames
- Extract high-attention frames for Stage 2
- Interpretable model outputs

---

### 4. **Training Improvements**

#### Gradient Clipping
```python
GRADIENT_CLIP_VALUE = 1.0
torch.nn.utils.clip_grad_norm_(model.parameters(), GRADIENT_CLIP_VALUE)
```
- Keeps weights stable during training
- Prevents exploding gradients
- Ensures attention weights stay in valid range [0, 1]

#### Other Enhancements
- **AdamW optimizer** with weight decay (1e-4)
- **CosineAnnealingLR scheduler** for learning rate decay
- **Accuracy tracking** during training
- **Better checkpoint saving** with full training state

---

### 5. **Stage 1 Inference - Attention Extraction**

The inference phase now extracts and saves:
1. **Anomaly predictions** (crime vs. normal)
2. **Confidence scores** for each prediction
3. **Attention weights** for all detected crime clips
4. **Top-5 frame indices** with highest attention
5. **Clip tensors** for Stage 2 training

**Output Files:**
- `./stage1_output/anomalies.json` - Detection results with metadata
- `./stage1_output/attention_data/stage1_attention_data.pt` - Full attention data for Stage 2
- `./stage1_output/attention_visualization.png` - Attention weight plots

---

### 6. **Visualization Added**

New visualization function to inspect attention weights:
- Bar plots showing attention per frame
- Highlights top-attention frames
- Shows model confidence
- Saves visualization images

---

## Configuration Changes

### Updated Paths
```python
# OLD (frame directories):
train_dir = r"C:\Users\rayaa\Downloads\ucf_crime\Train"

# NEW (video directories):
train_dir = r"C:\Users\rayaa\Downloads\ucf_crime_v2\Train"
```

**Note:** Make sure your directories contain `.mp4` files, not frame directories!

### Updated Transforms
```python
# Resize now handled in dataset via OpenCV
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                        std=[0.229, 0.224, 0.225])
])
```

---

## Expected Directory Structure

```
ucf_crime_v2/
├── Train/
│   ├── NormalVideos/
│   │   ├── video001.mp4
│   │   ├── video002.mp4
│   │   └── ...
│   ├── Abuse/
│   │   ├── video001.mp4
│   │   └── ...
│   ├── Arrest/
│   └── ... (other crime categories)
├── Test/
│   └── (same structure)
└── Validation/
    └── (same structure)
```

---

## Stage 1 Output

After running the updated notebook, you'll have:

1. **Trained Model**
   - Location: `C:\Users\rayaa\Downloads\ucf_crime_v2\checkpoints\binary_stage1_with_attention.pt`
   - Contains: model weights, optimizer state, scheduler state

2. **Anomaly Detection Results**
   - JSON file with all detected crime clips
   - Confidence scores
   - Attention weights for each clip
   - Top frame indices

3. **Stage 2 Training Data**
   - Pre-processed attention data
   - Ready for Stage 2 model training
   - Includes high-attention frames and tensors

---

## How to Use

### Step 1: Update Paths
Edit cell 4 to point to your video directories:
```python
train_dir = r"YOUR_PATH_TO_VIDEOS/Train"
test_dir  = r"YOUR_PATH_TO_VIDEOS/Test"
```

### Step 2: Run Training
Execute cells 1-8 to train the model with attention mechanism.

### Step 3: Run Inference
Execute cells 9-10 to:
- Detect crime clips in test set
- Extract attention weights
- Save data for Stage 2

### Step 4: Visualize (Optional)
Execute cells 11-12 to visualize attention weights.

---

## Next Steps: Stage 2

With the updated Stage 1 complete, you can now:

1. Load the attention data: `./stage1_output/attention_data/stage1_attention_data.pt`
2. Extract high-attention frames from crime videos
3. Train Stage 2 model for detailed anomaly classification
4. Use the same data ingestion approach for consistency

---

## Summary

✅ **Data Ingestion**: Now matches Experiment 1's efficient MP4 loading approach  
✅ **Attention Mechanism**: Identifies crime-relevant frames  
✅ **Gradient Clipping**: Stable training with bounded weights  
✅ **Stage 1 Complete**: Binary classification with frame importance  
✅ **Ready for Stage 2**: Attention data extracted and saved  

The notebook is now ready to run with the improved data pipeline!

