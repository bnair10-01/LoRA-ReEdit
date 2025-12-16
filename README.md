# LoRA-ReEdit: Exemplar-Based Image Editing with Low-Rank Adaptation

A PyTorch implementation combining Low-Rank Adaptation (LoRA) with ReEdit for efficient, structure-preserving exemplar-based image editing using Stable Diffusion.

## Overview

This project demonstrates how LoRA can enhance structure-preserving image editing capabilities while maintaining computational efficiency. The system uses a two-pass UNet architecture with DDIM inversion and feature injection for high-quality edits guided by exemplar image pairs.

### Key Features

- **LoRA Integration**: Efficient fine-tuning with rank-6 low-rank adaptation on attention layers
- **IP-Adapter**: Image-based conditioning with 4-token projection
- **Two-Pass Denoising**: Structure preservation through DDIM inversion and feature injection
- **Comprehensive Evaluation**: CLIP scores, L1 loss, and LPIPS metrics
- **Optimized Training**: Gradient checkpointing, mixed precision, and smart hyperparameter configuration

### Architecture Highlights

- **LoRA Rank**: 6 (sweet-spot configuration)
- **IP-Adapter Tokens**: 4
- **Edit Scale**: 1.2
- **Training Epochs**: 8
- **Text Dropout**: 30%
- **Image Size**: 256x256

## Installation

### Prerequisites

- Python 3.8+
- CUDA-compatible GPU (recommended: 12GB+ VRAM)
- PyTorch 2.0+

### Setup

1. Extract the submission zip file:
```bash
unzip lora-reedit-submission.zip
cd lora-reedit
```

2. Install dependencies:
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install diffusers transformers accelerate datasets pillow scikit-learn lpips
```



## Dataset

### Using HuggingFace Dataset (Recommended)

The code automatically downloads the ReEdit benchmark dataset:

```python
from datasets import load_dataset
dataset = load_dataset("tarun-menta/re-edit-bench")
```

The dataset contains image pairs with:
- `x_original`: Original input images
- `x_edited`: Target edited images  
- `caption`: Optional edit descriptions

### Custom Dataset Structure

To use your own dataset, organize it as:

```
dataset/
├── train/
│   ├── original/
│   │   ├── img001.jpg
│   │   ├── img002.jpg
│   │   └── ...
│   └── edited/
│       ├── img001.jpg
│       ├── img002.jpg
│       └── ...
└── val/
    ├── original/
    └── edited/
```

Then modify the dataset loading code in the notebook.

## Model Weights

### Download Pre-trained Weights

Download the trained model weights from Google Drive:

🔗 **[Model Weights & Checkpoints](https://drive.google.com/drive/folders/1VJS0W_y9xmt2j7whfs28kBRjhYSVU9wL?usp=sharing)**

The folder contains:
- `lora_reedit_final_model.pt` - Complete trained model (UNet + IP-Adapter + Edit MLP)
- `lora_weights_only.pt` - Isolated LoRA parameters (256 weights)
- `ip_adapter_weights.pt` - IP-Adapter projection weights
- `edit_mlp_weights.pt` - Edit direction MLP weights


### Model Directory Structure

After downloading, organize the weights:

```
weights/
├── lora_reedit_final_model.pt
├── lora_weights_only.pt
├── ip_adapter_weights.pt
├── edit_mlp_weights.pt

```

## Usage

### Training from Scratch

1. Open the Jupyter notebook or better yet use Google Colab:
```bash
jupyter notebook BNair_GenAI_FinalPro.ipynb
```


2. Configure hyperparameters in the CONFIG cell:
```python
CONFIG = {
    'n_samples': 500,           # Dataset size
    'image_size': 256,          # Image resolution
    'batch_size': 1,            # Batch size
    'lora_rank': 6,             # LoRA rank (sweet-spot)
    'ip_adapter_tokens': 4,     # IP-Adapter tokens
    'edit_scale': 1.2,          # Edit strength
    'epochs': 8,                # Training epochs
    'text_dropout': 0.3,        # Text conditioning dropout
    'lr': 5e-5,                 # Learning rate
    'inference_steps': 20,      # Denoising steps
}
```

3. Run all cells sequentially to:
   - Load models and dataset
   - Initialize LoRA, IP-Adapter, and Edit MLP
   - Train the model
   - Save checkpoints and weights

### Inference with Pre-trained Weights

```python
import torch
from PIL import Image

# Load pre-trained model
checkpoint = torch.load('weights/lora_reedit_final_model.pt')
unet.load_state_dict(checkpoint['unet_state_dict'])
ip_adapter.load_state_dict(checkpoint['ip_adapter_state_dict'])
edit_mlp.load_state_dict(checkpoint['edit_mlp_state_dict'])

# Load input images
original_img = Image.open('path/to/original.jpg').convert('RGB')
target_img = Image.open('path/to/target.jpg').convert('RGB')

# Perform editing (see notebook for complete inference pipeline)
edited_result = inference(
    original_img=original_img,
    target_img=target_img,
    unet=unet,
    ip_adapter=ip_adapter,
    edit_mlp=edit_mlp,
    scheduler=scheduler,
    vae=vae,
    clip_encoder=clip_encoder,
    config=CONFIG
)

# Save result
edited_result.save('output/edited_image.png')
```

### Evaluation

The notebook includes comprehensive evaluation code:

```python
# Run evaluation on validation set
results = evaluate_model(
    model=unet,
    val_loader=val_loader,
    metrics=['clip', 'l1', 'lpips']
)

print(f"CLIP Score: {results['clip_score']:.4f}")
print(f"L1 Loss: {results['l1_loss']:.4f}")
print(f"LPIPS: {results['lpips']:.4f}")
```

## Configuration Details

### Optimal Hyperparameters (Sweet-Spot)

These hyperparameters were found through extensive ablation studies:

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| LoRA Rank | 6 | Balances capacity and overfitting prevention |
| IP-Adapter Tokens | 4 | Sufficient for image conditioning without bloat |
| Edit Scale | 1.2 | Optimal edit strength without artifacts |
| Training Epochs | 8 | Convergence without overfitting |
| Text Dropout | 30% | Forces reliance on image features |
| Learning Rate | 5e-5 | Stable training for LoRA parameters |
| Inference Steps | 20 | Quality-speed trade-off |

### Training Details

- **Optimizer**: Adam with gradient scaling
- **Precision**: Mixed precision (FP16)
- **Gradient Checkpointing**: Enabled for memory efficiency
- **Bias Timesteps**: Weighted sampling toward mid-range
- **VAE Encoding**: Uses mean of latent distribution

## Project Structure

```
lora-reedit-submission/
├── BNair_GenAI_FinalPro.ipynb  # Main training/inference notebook
├── README.md                    # This documentation file
├── weights/                     # Model weights (download from Google Drive)
│   ├── lora_reedit_final_model.pt
│   ├── lora_weights_only.pt
│   ├── ip_adapter_weights.pt
│   └── edit_mlp_weights.pt
├── training_plots/              # Generated training curves
├── inference_results/           # Output edited images
└── evaluation_results/          # Quantitative metrics
```

**Note**: Due to file size constraints, model weights are hosted on Google Drive (link provided above). Download and place them in the `weights/` directory before running inference.

## Key Components

### 1. LoRA Layer Implementation

```python
class LoRALayer(nn.Module):
    """Low-Rank Adaptation layer for efficient fine-tuning"""
    def __init__(self, layer, rank=6, alpha=1.0):
        super().__init__()
        self.layer = layer
        self.rank, self.alpha = rank, alpha
        # Initialize low-rank matrices
        self.lora_A = nn.Parameter(torch.randn(...) * 0.01)
        self.lora_B = nn.Parameter(torch.zeros(...))
```

### 2. IP-Adapter

```python
class IPAdapter(nn.Module):
    """Image-based conditioning adapter"""
    def __init__(self, num_tokens=4):
        super().__init__()
        self.proj = nn.Linear(768, 768 * num_tokens)
```

### 3. Edit MLP

```python
class EditMLP(nn.Module):
    """Maps edit directions in CLIP space"""
    def __init__(self):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(768, 768),
            nn.ReLU(),
            nn.Linear(768, 768)
        )
```

### 4. Two-Pass Denoising

1. **DDIM Inversion**: Extract structure from original image
2. **Feature Injection**: Inject structure features during denoising
3. **Conditional Generation**: Apply edit based on target exemplar

## Evaluation Results

Based on Re-Edit's own Dataset as subset (500 samples, 400 train / 100 val):

| Method | CLIP Score ↑ | L1 Loss ↓ | LPIPS ↓ |
|--------|-------------|-----------|---------|
| ReEdit + LoRA | 0.167 | 0.285 | 0.320 |
| ReEdit (baseline) | 0.165 | 0.290 | 0.325 |
| Vanilla SD | 0.142 | 0.380 | 0.410 |

**Key Findings**:
- LoRA effects are subtle due to strong feature injection from ReEdit architecture
- Both ReEdit variants significantly outperform vanilla Stable Diffusion
- LoRA shows marginal improvements in semantic alignment (CLIP scores)
- Feature injection dominates structural preservation

## Troubleshooting

### Common Issues

**1. CUDA Out of Memory**
```python
# Reduce batch size
CONFIG['batch_size'] = 1

# Enable gradient checkpointing (already enabled)
unet.enable_gradient_checkpointing()

# Clear cache
torch.cuda.empty_cache()
```

**2. Model Loading Errors**
```python
# Ensure weights are downloaded from Google Drive
# Place them in the weights/ directory
checkpoint = torch.load('weights/lora_reedit_final_model.pt', map_location='cuda')
assert checkpoint['config']['lora_rank'] == 6
```

**3. Dataset Download Issues**
```python
# Use cache directory
from datasets import load_dataset
dataset = load_dataset("tarun-menta/re-edit-bench", cache_dir="./cache")
```

**4. Missing Weights**
- Download all weight files from the Google Drive link provided above
- Ensure they are placed in a `weights/` directory in the submission folder
- Verify file integrity (final model should be ~3.4GB)

## Technical Notes

### Device Placement

All modules must be explicitly placed on the correct device:
```python
unet = unet.to(device)
ip_adapter = ip_adapter.to(device)
edit_mlp = edit_mlp.to(device) if edit_mlp else None
```

### Circular Reference Prevention

Avoid storing full modules in training history:
```python
# Good: Store only necessary info
history['epoch'] = epoch_num

# Bad: Causes memory leaks
history['model'] = unet  # Don't do this!
```

## Citation

If you use this work, please cite:

```bibtex
@misc{nair-lora-reedit-2025,
  author = {Nair, B.},
  title = {LoRA-ReEdit: Exemplar-Based Image Editing with Low-Rank Adaptation},
  year = {2024},
  note = {GenAI Final Project}
}
```

## References

1. **LoRA**: [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685)
2. **ReEdit**: [Exemplar-Based Image Editing with Diffusion Models](https://arxiv.org/abs/2302.06671)
3. **Stable Diffusion**: [High-Resolution Image Synthesis with Latent Diffusion Models](https://arxiv.org/abs/2112.10752)
4. **IP-Adapter**: [Image Prompt Adapter for Text-to-Image Diffusion Models](https://arxiv.org/abs/2308.06721)

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Tarun Mehta and Re-edit team dataset for training data
- Hugging Face for model hosting and datasets library
- Stable Diffusion community for pretrained models
- ReEdit authors for benchmark dataset

## Contact

For questions about this implementation, please refer to the submitted notebook and documentation.

---

**Submission Package Contents**:
- `BNair_GenAI_FinalPro.ipynb` - Complete implementation with training and inference
- `README.md` - This documentation file
- `weights/` - Pre-trained model weights (see Google Drive link above)
- Presentation materials and evaluation results

**Note**: This is an academic project demonstrating LoRA integration with ReEdit for exemplar-based image editing. The "sweet-spot" configuration (rank-6 LoRA, 4 IP-Adapter tokens, edit scale 1.2, 8 epochs, 30% text dropout) represents optimized hyperparameters found through systematic ablation studies for this specific implementation and dataset.
