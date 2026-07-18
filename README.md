# Generative MedFormer Thesis Project

Repository for a bachelor thesis in Computer Engineering.

The project starts from MedFormer and aims to develop a generative version using
a latent diffusion model. The current implementation focuses on a clean research
pipeline for oceanographic Copernicus NetCDF data:

```text
DataManager
-> Preprocessing
-> Temporal split
-> Normalization
-> PyTorch Dataset
-> DataLoader
-> 3D U-Net Autoencoder
-> Latent space
```

Large data files are not tracked in Git.

Expected local data paths:

```text
data/raw/copernicus.nc
data/masks/land_sea_mask.nc
```

Colab example:

```bash
git clone https://github.com/arturomarino/thesis_repo.git
cd thesis_repo
pip install -r requirements.txt
python src/train.py \
  --data-path /content/drive/MyDrive/tesi/copernicus.nc \
  --mask-path /content/drive/MyDrive/tesi/land_sea_mask.nc \
  --stats-path /content/drive/MyDrive/tesi/normalization_stats.nc \
  --first-real-training-step
```

Quick checks:

```bash
python src/train.py --smoke-test-normalization
python src/train.py --smoke-test-dataset
python src/train.py --smoke-test-dataloader
python src/train.py --smoke-test-autoencoder
python src/train.py --smoke-test-training
python src/train.py --first-real-training-step
```
