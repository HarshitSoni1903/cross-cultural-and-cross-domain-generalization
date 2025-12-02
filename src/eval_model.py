# eval_model.py
"""
TSNE + UMAP evaluation utilities for XLM-R sentiment models.
"""

import os
import torch
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
import matplotlib.patches as mpatches

try:
    import umap
    HAS_UMAP = True
except Exception:
    HAS_UMAP = False


# ============================================================
# Helper: get the actual backbone encoder
# ============================================================
def _get_backbone(model):
    """
    Detect whether model is:
      • XLM-R classification model (has .roberta)
      • Your DualEncoderXLMROBERTaRating (has .new_encoder)
      • Plain HF encoder (model itself)
    """
    if hasattr(model, "new_encoder"):     # Your LD encoder
        return model.new_encoder
    if hasattr(model, "roberta"):         # HF XLM-R models
        return model.roberta
    if hasattr(model, "encoder"):         # fallback
        return model.encoder
    return model                           # default: treat model as encoder


# ============================================================
# 1. Extract CLS embeddings robustly
# ============================================================
def _extract_embeddings(model, dataloader, device, max_points=8000):
    backbone = _get_backbone(model)
    backbone.eval()

    all_embeds = []
    all_labels = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Extracting embeddings"):
            if len(all_embeds) >= max_points:
                break

            # single-encoder input
            if "input_ids" in batch:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)

            # dual encoder (LD)
            else:
                input_ids = batch["input_ids_original"].to(device)
                attention_mask = batch["attention_mask_original"].to(device)

            # Forward the *backbone encoder*
            outputs = backbone(
                input_ids=input_ids,
                attention_mask=attention_mask
            )

            # CLS vector — always hidden_states[0]
            # cls_emb = outputs.last_hidden_state[:, 0, :]  # [bs, H]
            # cls_emb = torch.nn.functional.normalize(cls_emb, p=2, dim=1)
            
            if "new_pooled" in outputs:
                pooled = outputs["new_pooled"]
            else:
                pooled = outputs.last_hidden_state[:, 0, :]
                
            pooled = torch.nn.functional.normalize(pooled, p=2, dim=1)
            labels = batch["labels"].cpu().numpy()
            all_embeds.append(pooled.cpu().numpy())
            all_labels.append(labels)

    embeddings = np.vstack(all_embeds)
    labels = np.concatenate(all_labels)
    return embeddings[:max_points], labels[:max_points]


# ============================================================
# 2. Plot helper
# ============================================================
def _plot_2d(xy, labels, title, out_path):
    plt.figure(figsize=(7, 7))
    scatter = plt.scatter(
        xy[:, 0],
        xy[:, 1],
        c=labels,
        cmap="viridis",
        s=8,
        alpha=0.75
    )

    plt.title(title)

    # ---- Create a custom legend ----
    unique_labels = sorted(np.unique(labels))

    class_names = {
        0: "Negative",
        1: "Neutral",
        2: "Positive"
    }

    handles = []
    for lab in unique_labels:
        color = scatter.cmap(scatter.norm(lab))
        handles.append(mpatches.Patch(color=color, label=class_names.get(lab, str(lab))))

    plt.legend(handles=handles, title="Sentiment", loc="best")

    # Colorbar still helps see where class values sit
    plt.colorbar(scatter, label="Label Value")

    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()



# ============================================================
# 3. Main callable
# ============================================================
def run_tsne_umap(model, tokenizer, dataloader, device, output_dir, max_points=8000):
    save_dir = os.path.join(output_dir, "tsne_umap")
    os.makedirs(save_dir, exist_ok=True)

    print("\n[TSNE/UMAP] Extracting embeddings...")
    embeddings, labels = _extract_embeddings(model, dataloader, device, max_points)

    print("[TSNE/UMAP] Standardizing embeddings...")
    X = StandardScaler().fit_transform(embeddings)

    # TSNE
    print("[TSNE] Running TSNE...")
    tsne = TSNE(n_components=2, learning_rate="auto", init="pca", perplexity=30)
    tsne_xy = tsne.fit_transform(X)
    _plot_2d(tsne_xy, labels, "TSNE Embeddings", os.path.join(save_dir, "tsne.png"))

    # UMAP
    if HAS_UMAP:
        print("[UMAP] Running UMAP...")
        reducer = umap.UMAP(n_components=2, n_neighbors=15, min_dist=0.1)
        umap_xy = reducer.fit_transform(X)
        _plot_2d(umap_xy, labels, "UMAP Embeddings", os.path.join(save_dir, "umap.png"))
    else:
        print("[UMAP] Skipped (umap-learn not installed).")

    print(f"[TSNE/UMAP] Finished. Saved under {save_dir}")
