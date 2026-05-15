# Zero-Order Fine-Tuning of ResNet18 on CIFAR100

## Reproducibility

Environment:

```bash
pip install -r requirements.txt
```

Main run used for the final submission:

```bash
python validate.py \
  --data_dir ./data \
  --batch_size 256 \
  --n_batches 32 \
  --seed 42 \
  --output results.json
```

This uses the full allowed sample budget: `32 * 256 = 8192`.

The solution does not require changes to `validate.py` or `model.py`. Only the four allowed files are modified:

- `zo_optimizer.py`
- `head_init.py`
- `augmentation.py`
- `train_data.py`

## Final solution description

### Zero-order optimizer

The optimizer tunes only the final classifier head:

```python
self.layer_names = ["fc.weight", "fc.bias"]
```

The ResNet18 backbone is already pretrained on ImageNet, so the main mismatch is the new CIFAR100 classification head. Under the small sample budget, tuning deeper convolutional layers makes the SPSA search space much larger and increases estimator variance.

The finite-difference baseline was replaced with antithetic SPSA. Each SPSA direction perturbs all selected parameters simultaneously and estimates one pseudo-gradient with two scalar loss evaluations:

```text
g ~= (f(theta + eps * u) - f(theta - eps * u)) / (2 * eps) * u
```

The final optimizer averages several Rademacher SPSA directions per step and applies Adam-style moment updates. It also uses perturbation annealing, light weight decay on `fc.weight`, and per-tensor RMS clipping of the update.

### Head initialization

The new classification head is initialized with small-gain Xavier initialization:

```python
nn.init.xavier_uniform_(layer.weight, gain=0.25)
nn.init.zeros_(layer.bias)
```

This keeps initial logits close to uniform while still breaking symmetry. It was more stable for SPSA than larger Kaiming-scale logits.

### Augmentation

The training pipeline uses moderate augmentation:

- resize to 256
- random crop to 224
- horizontal flip
- small color jitter
- normalization with CIFAR100 statistics
- low-probability random erasing

The validation transform is unchanged and deterministic.

### Training data

`train_data.py` creates a deterministic class-balanced CIFAR100 subset with 96 examples per class. This gives 9600 examples total, enough for the recommended `32 x 256` run without cycling through the loader, while avoiding class imbalance in the limited training budget.

## What contributed most

The biggest improvement came from replacing per-parameter finite differences with SPSA and restricting optimization to the classifier head. Averaging several antithetic directions improved stability, while Adam-style updates made the method less sensitive to raw SPSA gradient scale.

## Experiments and failed attempts

- **Per-parameter finite differences** were discarded because even the final head has about 51k parameters. It is too expensive and does not use the budget efficiently.
- **Tuning `layer4` plus the head** was less stable. The additional capacity did not compensate for the much noisier zero-order signal.
- **Very strong augmentation**, including AutoAugment and high-probability RandomErasing, increased the variance of repeated scalar loss evaluations and made updates less reliable.
- **Zero initialization** was avoided because it delays useful symmetry breaking. Large Kaiming-style initialization was also worse because early logits and losses were less stable for finite differences.
