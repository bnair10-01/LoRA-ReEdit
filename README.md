# LoRA-ReEdit: LoRA Fine-Tuning for Exemplar-Based Image Editing

A PyTorch/diffusers notebook that learns to transfer an edit shown by an **exemplar pair** (before → after) onto a new image, using Stable Diffusion v1.5 with a small set of trainable parts: LoRA on the UNet attention layers, an IP-Adapter-style image-token projection, and a small MLP on CLIP edit directions.

This is a **simplified, ReEdit-inspired** system, not a re-implementation of the full ReEdit pipeline (see [Differences from ReEdit](#differences-from-reedit)). It was built as a course final project; the evaluation is small and exploratory, and the results section says exactly what it does and does not show.

## How it works

For a training/inference example the dataset provides an exemplar pair `(x_original, x_edited)`, a test image `y_original`, and an optional caption.

1. **Edit direction.** `CLIP-ViT-L/14` image embeddings: `edit = CLIP(x_edited) − CLIP(x_original)` (768-d).
2. **Edit MLP.** `Linear(768,768) → ReLU → Linear(768,768)` refines the direction.
3. **IP-Adapter-style tokens.** `Linear(768, 4×768)` turns it into 4 tokens, scaled by `edit_scale = 1.2`, and **concatenated to the 77 text-token embeddings** along the sequence axis as the UNet's cross-attention context. (This is a simplified variant: there is no separate decoupled cross-attention branch as in the original IP-Adapter.)
4. **LoRA.** A hand-written `LoRALayer` (rank 6, scale 1.0, `A ~ N(0, 0.01)`, `B = 0`) wraps every `to_q`, `to_k`, `to_v`, `to_out.0` linear in the SD-1.5 UNet: 128 layers, about 1.2M parameters.
5. **Trainable parameters:** 4,739,328 in total (LoRA + IP projection + Edit MLP), with the UNet, VAE and both CLIP encoders frozen.

### Training
- Objective: standard noise-prediction MSE on the VAE latents (distribution mean) of `x_edited`, conditioned on caption text (30% of steps the text embedding is zeroed) plus the edit tokens.
- Timesteps sampled from the middle of the schedule (200–800 of 1000).
- AdamW (lr 5e-5, weight decay 1e-2), FP16 autocast with GradScaler, gradient clipping at 1.0, gradient checkpointing, batch size 1, 8 epochs, 256×256 images, seed 42.
- Data: the first 500 samples of `tarun-menta/re-edit-bench`, split 400 train / 100 validation (`random_state=42`).
- Observed cost: about 10 minutes per epoch on an A100; about 10 GB of GPU memory after loading the models.

### Inference
Encode the test image `y_original` with the VAE, add noise to step index 7 of a 20-step DDIM schedule (t ≈ 600), then denoise the remaining 13 steps with the text + edit-token conditioning and decode. This is img2img-style editing from a partially noised latent. There is no DDIM inversion and no feature injection.

## Results

Evaluated in the notebook on **5 validation samples** (not the full 100-sample validation set), with the same noise for all three methods:

| Method | CLIP score ↑ | LPIPS ↓ | L1 ↓ |
|--------|-------------:|--------:|-----:|
| Vanilla SD (LoRA off, text only) | 0.5892 | 0.7463 | 0.7403 |
| Edit tokens, LoRA off | 0.5935 | 0.7426 | 0.7400 |
| **LoRA-ReEdit (all components)** | **0.6946** | **0.6773** | **0.7168** |

Training loss moved only from 0.129 (epoch 1) to 0.123 (epoch 8), with its lowest average at epoch 2 (0.116): the loss is noisy at batch size 1 and essentially flat.

**How to read this**
- The direction of the comparison favors the full model on all three metrics, but n = 5, one seed, no confidence intervals: it is an illustration, not a benchmark.
- The "edit tokens, LoRA off" row is **not an independent ReEdit baseline**. It is the same trained IP projection and Edit MLP with the LoRA scale set to 0 at inference, and those parts were trained jointly with LoRA. The gap therefore partly reflects removing something the other modules were trained alongside.
- The metrics compare each output with the **exemplar's edited image** (`x_edited`), not with a ground-truth edit of the test image. "CLIP score" is the cosine similarity between CLIP-L image embeddings of the two images.
- Known evaluation bug: generated images are in [0, 1] while the target is in [−1, 1], and the metric helpers assume [−1, 1]. This affects all three methods the same way, but it makes the absolute values (L1 ≈ 0.74, LPIPS ≈ 0.7) uninformative; only the ordering should be read.
- Qualitative comparisons (exemplar pair, test image, three outputs) are produced by the notebook for visual inspection.

## Differences from ReEdit

ReEdit ([Srivastava et al., WACV 2025](https://arxiv.org/abs/2411.03982)) is an optimization-free pipeline using DDIM inversion with feature injection and VLM-derived edit captions. This project borrows the exemplar-pair idea and the CLIP-space edit direction, and adds a trained LoRA and IP-style projection on top. It does **not** implement DDIM inversion, feature/attention injection, or VLM captioning (captions come from the dataset). Results here are not comparable to published ReEdit numbers.

## Limitations

- Small evaluation (5 images), a single run, and the metric issues listed above. No test-set protocol, human study, or comparison to the original ReEdit.
- No ablation results are included in this repository. Rank 6, 4 tokens, edit scale 1.2, text dropout 0.3 and 8 epochs are the final configuration, not a documented sweep.
- Training loss is nearly flat, so how much the learned parts help beyond the pretrained model is unclear from this evidence.
- 256×256 resolution only; one exemplar pair per edit; batch size 1.

## Setup and usage

The notebook (`BNair_GenAI_FinalPro.ipynb`) was run in Google Colab on an A100. Cells, in order: install and imports → `CONFIG` → dataset → load SD-1.5, CLIP, LPIPS → LoRA / IP-Adapter / Edit-MLP definitions → data loader → train → save → training plots → generate comparisons and metrics → final inference cell.

```bash
pip install torch torchvision diffusers transformers accelerate datasets pillow scikit-learn lpips matplotlib seaborn pandas tqdm
```

Package versions were not recorded. A CUDA GPU with roughly 12 GB or more of memory is needed; the first run downloads `runwayml/stable-diffusion-v1-5`, `openai/clip-vit-large-patch14` and the dataset from the Hugging Face Hub.

`CONFIG` (second code cell):

```python
CONFIG = {
    'n_samples': 500, 'image_size': 256, 'batch_size': 1,
    'lora_rank': 6, 'ip_adapter_tokens': 4, 'edit_scale': 1.2,
    'epochs': 8, 'text_dropout': 0.3, 'lr': 5e-5,
    'start_idx': 7, 'inference_steps': 20,
    # also: use_edit_mlp, use_vae_mean, use_text, bias_timesteps, noise_scale, ...
}
```

### Dataset

`load_dataset("tarun-menta/re-edit-bench")`. Fields used: `x_original`, `x_edited` (exemplar pair), `y_original` (image to edit) and `caption`.

### Weights

The trained checkpoint files are hosted outside the repo: [Google Drive folder](https://drive.google.com/drive/folders/1VJS0W_y9xmt2j7whfs28kBRjhYSVU9wL?usp=sharing).

- `lora_reedit_final_model.pt`: UNet state dict (including LoRA), IP projection, Edit MLP, config, training history (the full UNet state dict makes it several GB)
- `lora_weights_only.pt`: the 256 LoRA tensors (A and B for each of the 128 layers)
- `ip_adapter_weights.pt`, `edit_mlp_weights.pt`

The last notebook cell rebuilds the models, injects LoRA, loads `lora_reedit_final_model.pt` (the path is hard-coded to a Google Drive location, so edit `checkpoint_path`), and writes five example edits to `inference_final/`. The training cell also saves a full checkpoint every epoch (`checkpoint_epoch_N.pt`), so allow ample disk space.

### Other outputs
`training_plots/training_curves_comprehensive.png` (loss curves), `inference_final/sample_*_complete.png` and `sample_*_output.png`.

## References

- Hu et al., *LoRA: Low-Rank Adaptation of Large Language Models*. https://arxiv.org/abs/2106.09685
- Srivastava et al., *ReEdit: Multimodal Exemplar-Based Image Editing with Diffusion Models*, WACV 2025. https://arxiv.org/abs/2411.03982
- Rombach et al., *High-Resolution Image Synthesis with Latent Diffusion Models*. https://arxiv.org/abs/2112.10752
- Ye et al., *IP-Adapter: Text Compatible Image Prompt Adapter for Text-to-Image Diffusion Models*. https://arxiv.org/abs/2308.06721

## Acknowledgments

The ReEdit authors (including Tarun Ram Menta) for the exemplar-editing benchmark dataset; Hugging Face for model and dataset hosting.

## License

MIT, see [LICENSE](LICENSE).
