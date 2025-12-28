import os
import numpy as np
from PIL import Image
import torch
from torch.utils.data import DataLoader
from datasets import load_dataset

# === ENV dédiés HSC (isole du cache Euclid) ===
os.environ["HF_HOME"] = "/pbs/throng/training/astroinfo2025/model/hsc/hf_home"
os.environ["HF_HUB_CACHE"] = "/pbs/throng/training/astroinfo2025/model/hsc/hf_home/hub"
os.environ["HF_DATASETS_CACHE"] = "/pbs/throng/training/astroinfo2025/model/hsc/hf_home/datasets"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"

def to_tensor_image(x, normalize_255=True):
    """Convertit image (PIL/np) -> torch.FloatTensor (H, W) ou (C, H, W) [0..1]."""
    if isinstance(x, Image.Image):
        x = np.array(x)
    if x.ndim == 2:
        t = torch.from_numpy(x).float()
        return (t / 255.0) if normalize_255 else t
    elif x.ndim == 3:
        # Assume HWC
        t = torch.from_numpy(x).permute(2, 0, 1).float()
        return (t / 255.0) if normalize_255 else t
    else:
        raise ValueError(f"Unexpected image ndim={x.ndim}")

class HSCDataset(torch.utils.data.Dataset):
    """
    PyTorch Dataset wrapper pour 'MultimodalUniverse/hsc'.
    Essaie de récupérer les bandes HSC (i, g, r, z, y). Construit un RGB à partir de (g, r, i) si possible.
    """
    def __init__(self, split="train", transform=None,
                 cache_dir="/pbs/throng/training/astroinfo2025/model/hsc/hf_home/datasets"):
        os.makedirs(cache_dir, exist_ok=True)
        print(f"Loading HSC dataset (split={split}) into cache_dir={cache_dir}")
        self.dataset = load_dataset(
            "MultimodalUniverse/hsc",
            split=split,
            cache_dir=cache_dir
        )
        self.transform = transform

        # Noms de clés possibles (on couvre quelques variantes courantes)
        self.band_keys = {
            "i":  ["HSC_i", "i", "band_i"],
            "g":  ["HSC_g", "g", "band_g"],
            "r":  ["HSC_r", "r", "band_r"],
            "z":  ["HSC_z", "z", "band_z"],
            "y":  ["HSC_y", "y", "band_y"],
        }
        # Métadonnées possibles
        self.meta_keys = ["object_id", "targetid", "id", "source_id"]

    def __len__(self):
        return len(self.dataset)

    def _get_first_present(self, sample, candidates):
        for k in candidates:
            if k in sample and sample[k] is not None:
                return sample[k]
        return None

    def __getitem__(self, idx):
        s = self.dataset[idx]

        # Récupère bandes individuelles (si dispo)
        band_tensors = {}
        for b in ["i", "g", "r", "z", "y"]:
            img = self._get_first_present(s, self.band_keys[b])
            band_tensors[b] = to_tensor_image(img) if img is not None else None

        # Construit un RGB (g, r, i) si possible, sinon None
        rgb = None
        if band_tensors["g"] is not None and band_tensors["r"] is not None and band_tensors["i"] is not None:
            # Normaliser chaque canal au [0,1] (déjà fait) et empiler en CHW
            # Si entrée single-channel (H,W), on ajoute une dimension
            def ensure_chw(t):
                return t if t.ndim == 3 else t.unsqueeze(0)
            g = ensure_chw(band_tensors["g"])
            r = ensure_chw(band_tensors["r"])
            i = ensure_chw(band_tensors["i"])
            rgb = torch.cat([r, g, i], dim=0)  # R= r, G= g, B= i (choix classique pour pseudo-RGB)
        # Optionnel: clamp/scale supplémentaire si attendu par ton modèle

        # Métadonnées
        obj_id = None
        for k in self.meta_keys:
            if k in s:
                obj_id = s[k]
                break

        out = {
            "object_id": obj_id,
            "rgb_image": rgb,           # (3,H,W) ou None
            "hsc_i": band_tensors["i"], # (H,W) ou (C,H,W) selon origine, ici (H,W)
            "hsc_g": band_tensors["g"],
            "hsc_r": band_tensors["r"],
            "hsc_z": band_tensors["z"],
            "hsc_y": band_tensors["y"],
        }

        if self.transform is not None:
            out = self.transform(out)

        return out

def test_hsc_dataloader():
    print("Creating HSC PyTorch dataset...")
    try:
        dataset = HSCDataset(split="train")
        print(f"HSC dataset loaded with {len(dataset)} samples")

        dl = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=4, pin_memory=True)
        batch = next(iter(dl))

        print("\nBatch keys:", batch.keys())
        if batch["rgb_image"] is not None:
            print("RGB shape:", None if batch["rgb_image"][0] is None else batch["rgb_image"].shape)
        for b in ["hsc_g", "hsc_r", "hsc_i", "hsc_z", "hsc_y"]:
            t = batch[b]
            if t is not None and isinstance(t, torch.Tensor):
                print(f"{b} shape:", t.shape)
            else:
                print(f"{b}: None or missing")

        print("Sample object_ids:", batch["object_id"])
        print("HSC DataLoader test OK.")

    except Exception as e:
        print(f"[ERROR] HSC dataloader test failed: {e}")

if __name__ == "__main__":
    print("=== HSC DataLoader Test ===")
    test_hsc_dataloader()
