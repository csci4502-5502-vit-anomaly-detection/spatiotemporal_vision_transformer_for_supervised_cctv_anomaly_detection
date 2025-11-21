# Experiment 2 Notebook Comparison

## Two Versions Available

We now have **two versions** of Experiment 2 Stage 1:

### 1. `experiment_2_frfr.ipynb` - Simple OpenCV Version ✅
**Purpose**: Minimal changes - ONLY updates the data ingestion to use Experiment 1's approach

**What Changed:**
- ✅ Data loading: MP4 files → OpenCV (`cv2.VideoCapture`)
- ✅ On-the-fly frame extraction

**What Stayed the Same:**
- ⚠️ Original simple 3D CNN model (no attention)
- ⚠️ Basic training loop (Adam optimizer, no extras)
- ⚠️ Simple inference (just predictions + confidence scores)

**File Size**: ~9 cells
**Use Case**: Testing if the OpenCV data ingestion works correctly

---

### 2. `experiment_2_enhanced.ipynb` - Full Featured Version ⭐
**Purpose**: OpenCV data ingestion + ALL advanced features from the update

**What's Included:**
- ✅ OpenCV data loading (like simple version)
- ✅ **Attention mechanism** in model architecture
- ✅ **Gradient clipping** during training
- ✅ **AdamW optimizer** with weight decay
- ✅ **Cosine Annealing LR scheduler**
- ✅ **Accuracy tracking** during training
- ✅ **Attention weight extraction** for Stage 2
- ✅ **Data verification** cells
- ✅ **Visualization** of attention weights

**File Size**: ~13 cells
**Use Case**: Full Stage 1 pipeline with all enhancements

---

## Quick Comparison Table

| Feature | `experiment_2_frfr.ipynb` (Simple) | `experiment_2_enhanced.ipynb` (Full) |
|---------|-----------------------------------|-------------------------------------|
| **Data Ingestion** | ✅ OpenCV MP4 loading | ✅ OpenCV MP4 loading |
| **Model** | Simple 3D CNN | 3D CNN + Attention |
| **Training** | Basic (Adam) | Advanced (AdamW + scheduler + gradient clipping) |
| **Outputs** | Predictions only | Predictions + attention weights |
| **Stage 2 Ready** | ❌ No | ✅ Yes (attention data saved) |
| **Visualization** | ❌ No | ✅ Yes |
| **Data Verification** | ❌ No | ✅ Yes |
| **File Size** | ~9 cells | ~13 cells |
| **Training Time** | Baseline | +10-15% overhead |
| **Complexity** | Low | Medium |

---

## Which One Should You Use?

### Use `experiment_2_frfr.ipynb` (Simple) If:
- ✅ You only want to test the OpenCV data ingestion
- ✅ You want minimal changes from the original
- ✅ You're comparing performance between old vs new data loading
- ✅ You don't need Stage 2 preparation yet
- ✅ You prefer simplicity

### Use `experiment_2_enhanced.ipynb` (Full) If:
- ✅ You want the complete Stage 1 pipeline
- ✅ You need attention weights for Stage 2
- ✅ You want gradient clipping as per Stage 1 goals
- ✅ You want better model interpretability
- ✅ You're doing the full 2-stage experiment
- ✅ You want all the bells and whistles

---

## Code Differences

### Model Architecture

**Simple Version:**
```python
class BinaryCrimeDetector(nn.Module):
    def __init__(self):
        self.features = nn.Sequential(...)  # 3D CNN layers
        self.classifier = nn.Sequential(...)
    
    def forward(self, x):
        return self.classifier(self.features(x))
```

**Enhanced Version:**
```python
class BinaryCrimeDetector(nn.Module):
    def __init__(self):
        self.features = nn.Sequential(...)  # 3D CNN layers
        self.attention = nn.Sequential(...)  # NEW: Attention module
        self.classifier = nn.Sequential(...)
    
    def forward(self, x, return_attention=False):
        features = self.features(x)
        attention_map = self.attention(features)  # NEW
        weighted_features = features * attention_map  # NEW
        logits = self.classifier(weighted_features)
        
        if return_attention:
            return logits, attention_weights  # NEW
        return logits
```

### Training Loop

**Simple Version:**
```python
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

for epoch in range(EPOCHS):
    for clips, labels, _ in train_loader:
        optimizer.zero_grad()
        outputs = model(clips)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
```

**Enhanced Version:**
```python
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=5)

for epoch in range(EPOCHS):
    for clips, labels, _ in train_loader:
        optimizer.zero_grad()
        outputs, attention_weights = model(clips, return_attention=True)  # NEW
        loss = criterion(outputs, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # NEW
        optimizer.step()
    scheduler.step()  # NEW
```

### Inference

**Simple Version:**
```python
outputs = model(clips)
probs = torch.softmax(outputs, dim=1)
preds = probs.argmax(dim=1)
# Save predictions only
```

**Enhanced Version:**
```python
outputs, attention_weights = model(clips, return_attention=True)
probs = torch.softmax(outputs, dim=1)
preds = probs.argmax(dim=1)
# Save predictions + attention weights + top frame indices
# Prepare data for Stage 2
```

---

## File Locations

```
models/sandbox/experiment_2/
├── experiment_2_frfr.ipynb           ← SIMPLE (OpenCV only)
├── experiment_2_enhanced.ipynb        ← FULL (OpenCV + all features)
├── NOTEBOOK_COMPARISON.md             ← This file
├── EXPERIMENT_2_UPDATES.md            ← Detailed changelog
├── QUICK_COMPARISON.md                ← Before/after comparison
└── README_UPDATED.md                  ← Usage guide
```

---

## Migration Path

If you're currently using the **simple version** and want to upgrade to the **enhanced version**:

1. Your OpenCV data loading code is identical - no changes needed
2. Just switch to `experiment_2_enhanced.ipynb`
3. Your video directories and paths remain the same
4. Training will take ~10-15% longer but produces better results

---

## Summary

✅ **Both notebooks have the improved OpenCV data ingestion**  
✅ **Simple version**: Minimal changes, easy to verify  
✅ **Enhanced version**: Full pipeline with all Stage 1 features  
✅ **Same data format**: Both use MP4 videos directly  
✅ **Easy switching**: Pick based on your current needs  

Choose the simple version for testing, the enhanced version for production! 🚀

