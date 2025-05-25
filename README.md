# PointNet on Toronto-3D Outdoor LiDAR Dataset

This project implements the PointNet architecture to perform **semantic segmentation** on the **Toronto-3D outdoor LiDAR dataset**. It includes data loading, preprocessing, training, and visualization of 3D point cloud predictions.

## 🚀 Overview

- **Dataset**: Toronto-3D — large-scale outdoor LiDAR point cloud dataset with 9 labeled classes
- **Model**: PointNet (adapted from Keras Examples)
- **Task**: Per-point classification (semantic segmentation)
- **Platform**: Google Colab (Python, TensorFlow/Keras)

## 📁 Directory Structure

```
project/
├── Toronto_3D/                    # Extracted dataset folder
│   ├── L001.ply
│   ├── L002.ply
│   ├── Colors.xml
│   └── Mavericks_classes_9.txt
├── pointnet_model.py              # Model architecture
├── data_generator.py              # Block sampling & generator
├── train.py                       # Training script
└── README.md
```

## 📦 Requirements

Install the required dependencies:

```bash
pip install numpy matplotlib laspy open3d plyfile tensorflow
```

### Dependencies

- **numpy**: Numerical computations
- **matplotlib**: 2D plotting and visualization
- **laspy**: LAS/LAZ file handling
- **open3d**: 3D data processing and visualization
- **plyfile**: PLY file reading/writing
- **tensorflow**: Deep learning framework

## 🗂️ Data Preparation

1. **Download the Toronto-3D dataset** from the official source
2. **Upload the .zip file** to your Google Colab environment
3. **Extract the dataset** using the following code:

```python
import zipfile
import os

# Extract the dataset
with zipfile.ZipFile('Toronto_3D.zip', 'r') as zip_ref:
    zip_ref.extractall('.')
    
# Verify extraction
print("Dataset files:")
for file in os.listdir('Toronto_3D/'):
    print(f"  - {file}")
```

### Dataset Structure

The Toronto-3D dataset contains:
- **Point cloud files**: L001.ply, L002.ply, etc.
- **Class definitions**: Mavericks_classes_9.txt (9 semantic classes)
- **Color mapping**: Colors.xml for visualization

### Class Labels (9 Classes)

1. Road
2. Road marking
3. Natural
4. Building
5. Utility line
6. Pole
7. Car
8. Fence
9. Unclassified

## 🏗️ Model Architecture

### PointNet Overview

- **Input**: (4096, 3) sampled block of 3D points
- **Layers**: MLPs, Max pooling, and per-point classification layers
- **Output**: Predicted class label per point
- **Based on**: Official PointNet paper by Qi et al.

### Architecture Details

```python
# Example model structure
Input Layer: (batch_size, 4096, 3)
    ↓
Transformation Network (T-Net)
    ↓
Feature Transform + MLPs
    ↓
Max Pooling (Global Features)
    ↓
Per-point Classification Head
    ↓
Output: (batch_size, 4096, 9)
```

## 🔄 Data Generator

The data generator handles efficient batch processing:

1. **PLY file loading** using `plyfile`
2. **Block sampling** of 4096 points from each scene
3. **Batch generation** yielding (points, labels) pairs to the model
4. **Data augmentation** (optional): rotation, scaling, jittering

### Usage Example

```python
from data_generator import DataGenerator

# Initialize generator
generator = DataGenerator(
    data_path='Toronto_3D/',
    batch_size=32,
    block_size=4096,
    shuffle=True
)

# Use with model training
model.fit(generator, epochs=100, validation_data=val_generator)
```

## 🏋️ Training

### Quick Start

```python
# Run the training script
python train.py --epochs 100 --batch_size 32 --learning_rate 0.001
```

### Training Parameters

- **Epochs**: 100 (recommended)
- **Batch Size**: 32 (adjust based on GPU memory)
- **Learning Rate**: 0.001 with decay
- **Optimizer**: Adam
- **Loss Function**: Sparse Categorical Crossentropy

### Monitoring Training

The training script includes:
- Loss and accuracy tracking
- Model checkpointing
- Early stopping
- TensorBoard logging

## 📊 Visualization

### 3D Point Cloud Visualization

The project supports multiple visualization methods:

1. **3D scatter plots** using matplotlib
2. **Interactive visualization** with Open3D
3. **Ground truth vs predictions** comparison
4. **Color-coded class legends**
5. **PLY file export** for external viewers

### Visualization Examples

```python
# Visualize predictions
from visualization import visualize_predictions

visualize_predictions(
    points=test_points,
    ground_truth=test_labels,
    predictions=model_predictions,
    save_path='results/visualization.png'
)
```

### Features

- **Color mapping**: Each class has a distinct color
- **Side-by-side comparison**: Ground truth vs predictions
- **Interactive 3D viewer**: Rotate, zoom, and inspect results
- **Export options**: Save as PNG, PLY, or other formats

## 📈 Results

### Expected Performance

- **Overall Accuracy**: ~85-90% on test set
- **Per-class IoU**: Varies by class complexity
- **Training Time**: ~2-3 hours on GPU (Google Colab)

### Evaluation Metrics

- Overall Accuracy
- Per-class Accuracy
- Mean IoU (Intersection over Union)
- Confusion Matrix

## 🚀 Getting Started

1. **Clone or download** this repository
2. **Install dependencies** using pip
3. **Download Toronto-3D dataset** and extract to project folder
4. **Run training**:
   ```bash
   python train.py
   ```
5. **Visualize results**:
   ```bash
   python visualize.py --model_path saved_models/pointnet_model.h5
   ```

## 🔧 Configuration

### Model Hyperparameters

Adjust hyperparameters in `config.py` or as command-line arguments:

```python
BLOCK_SIZE = 4096          # Points per block
NUM_CLASSES = 9            # Toronto-3D classes
BATCH_SIZE = 32           # Training batch size
LEARNING_RATE = 0.001     # Initial learning rate
EPOCHS = 100              # Training epochs
```

### Data Augmentation

Enable data augmentation for improved generalization:

```python
AUGMENTATION = {
    'rotation': True,      # Random rotation
    'scaling': True,       # Random scaling
    'jittering': True,     # Point jittering
    'dropout': 0.1         # Random point dropout
}
```

## 📝 Usage Notes

### Google Colab Setup

```python
# Mount Google Drive (optional)
from google.colab import drive
drive.mount('/content/drive')

# Install additional packages
!pip install open3d plyfile

# Set GPU runtime for faster training
```

### Memory Considerations

- **Block size**: Reduce from 4096 if memory issues occur
- **Batch size**: Adjust based on available GPU memory
- **Data caching**: Enable for faster training iterations

## 🤝 Contributing

Contributions are welcome! Please feel free to submit:

- Bug reports
- Feature requests
- Code improvements
- Documentation updates

## 📚 References

- **PointNet Paper**: Qi, Charles R., et al. "PointNet: Deep learning on point sets for 3d classification and segmentation." CVPR 2017.
- **Toronto-3D Dataset**: Tan, Weikai, et al. "Toronto-3D: A large-scale mobile LiDAR dataset for semantic segmentation of urban roadways." CVPR Workshops 2020.

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- Toronto-3D dataset creators
- PointNet original authors
- Keras Examples community
- Google Colab for computational resources

---

**Note**: This implementation is designed for educational and research purposes. For production use, consider additional optimizations and validation.


