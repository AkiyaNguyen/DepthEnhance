# DepthEnhance

DepthEnhance is a semi-supervised Mean Teacher framework for polyp segmentation that uses high-level features from a pretrained depth estimation model as a privileged signal in the teacher pathway, giving the student geometric guidance without exposing it to noisy raw pseudo-depth.

## 1. Setup

Clone the repository and install dependencies:

```bash
git clone https://github.com/AkiyaNguyen/DepthEnhance.git
cd DepthEnhance
pip install -r requirements.txt
```

## 2. Running training / evaluation

The entry point is [DEMT_DAv2.py](DEMT_DAv2.py). It takes a `--config` flag that accepts **one or more** YAML files separated by `,` — files listed later override duplicate keys in earlier files. The model config is merged first, the data config second.

Basic run (model config + data config):

```bash
python DEMT_DAv2.py \
    --config cfg/DEMT_DAv2.yaml,cfg/data/kvasir_SEG.yaml
```

- [cfg/DEMT_DAv2.yaml](cfg/DEMT_DAv2.yaml) — model / trainer / optimizer / hook settings.
- [cfg/data/kvasir_SEG.yaml](cfg/data/kvasir_SEG.yaml) — dataset paths and dataset-specific overrides (merged on top of the base config).

## 3. Overriding YAML params from the CLI

Any value in the merged YAML can be overridden by appending `key.path=value` arguments after the `--config` flag. The key path uses `.` as the nesting separator and matches the YAML structure exactly.

Examples:

```bash
# Point the trainer at a local dataset directory
python DEMT_DAv2.py \
    --config cfg/DEMT_DAv2.yaml,cfg/data/kvasir_SEG.yaml \
    data.root=/path/to/kvasir_SEG \
    data.data2_dir=Train \
    data.test.dataset_root=/path/to/kvasir_SEG/Test

# Tweak training hyperparameters
python DEMT_DAv2.py \
    --config cfg/DEMT_DAv2.yaml,cfg/data/kvasir_SEG.yaml \
    nEpoch=200 \
    optimizer.lr=0.005 \
    data.batch_size=8 \
    data.labeled_perc=20

# Swap the DAv2 teacher backbone
python DEMT_DAv2.py \
    --config cfg/DEMT_DAv2.yaml,cfg/data/kvasir_SEG.yaml \
    model.tea_model.dav2_model_name=depth-anything/Depth-Anything-V2-Base-hf
```

CLI overrides take precedence over both YAML files.

## 4. Evaluating on multiple test datasets in one run

`data.test.dataset_root` and `data.test.dataset_name` both accept a comma-separated list. The trainer builds one test dataloader per entry, evaluates each at the configured `eval_every_epoch` interval.

See [cfg/data/Kvasir_CVCClinic.yaml](cfg/data/Kvasir_CVCClinic.yaml) for a working example:

```yaml
data:
  test:
    dataset_root: /path/TestDataset/Kvasir,/path/TestDataset/ETIS-LaribPolypDB,/path/TestDataset/CVC-ColonDB,/path/TestDataset/CVC-ClinicDB,/path/TestDataset/CVC-300
    dataset_name: Kvasir,ETIS-LaribPolypDB,CVC-ColonDB,CVC-ClinicDB,CVC-300
```