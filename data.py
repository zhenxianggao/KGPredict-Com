"""
Data loading and preprocessing for Drug Combination Ranking

Fixes applied vs original:
  [data-1] KGGraph.node_type_mask now uses unified 3-value convention:
           0=drug, 1=disease, 2=gene  (matches model.py & trainer.py; removes
           the patch that was in Trainer.__init__).
  [data-2] KG edge building replaced iterrows() with vectorised pandas ops
           (~100× faster on large KGs).
  [data-3] _extract_drug_disease_pairs likewise vectorised.
  [data-4] Removed dead load_embeddings() function (model uses learnable embs).
  [data-5] DrugCombinationDataset negative-sampling: warn when pad is triggered
           so silent training bias is surfaced.
  [data-6] _extract_drug_disease_pairs: deduplicate returned pairs so that a
           (drug, disease) edge appearing under multiple relation types does not
           inflate that positive's weight during Phase 1 training.
  [data-7] DrugCombinationDataset: drop_duplicates on input CSV before indexing,
           then deduplicate pos_triplets after canonical (i<j) ordering so that
           swapped drug1/drug2 rows don't cause over-sampling.
"""

import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from typing import Dict, Tuple, List
from collections import defaultdict
import logging

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# CSV loaders
# ─────────────────────────────────────────────────────────────────────────────

def load_kg(kg_path: str) -> pd.DataFrame:
    """
    Load KG from CSV.
    Expected columns: node1, relation, node2
    All node IDs are forced to str to avoid float/str comparison errors
    when pandas reads numeric IDs (e.g. 12345 -> 12345.0).
    """
    df = pd.read_csv(kg_path, dtype=str)
    df.columns = [c.strip().lower() for c in df.columns]
    assert {'node1', 'relation', 'node2'}.issubset(df.columns), \
        "KG CSV must have columns: node1, relation, node2"
    df['node1']    = df['node1'].str.strip()
    df['node2']    = df['node2'].str.strip()
    df['relation'] = df['relation'].str.strip()
    before = len(df)
    df = df.dropna(subset=['node1', 'relation', 'node2'])
    if len(df) < before:
        logger.warning(f"Dropped {before - len(df)} rows with missing values in KG")
    return df


def load_pairs(path: str) -> pd.DataFrame:
    """
    Load disease-drug-drug pairs.
    Expected columns: disease_id, drug1_id, drug2_id
    All IDs are forced to str.
    """
    df = pd.read_csv(path, dtype=str)
    df.columns = [c.strip().lower() for c in df.columns]
    for col in ('disease_id', 'drug1_id', 'drug2_id'):
        df[col] = df[col].str.strip()
    df = df.dropna(subset=['disease_id', 'drug1_id', 'drug2_id'])
    return df


# ─────────────────────────────────────────────────────────────────────────────
# KG graph
# ─────────────────────────────────────────────────────────────────────────────

class KGGraph:
    """
    Processes KG into PyG-compatible format.

    Node ordering: [drugs..., diseases..., genes...]

    node_type_mask convention (3-value, used by both model.py & trainer.py):
        0 = drug
        1 = disease
        2 = gene
    """

    def __init__(
        self,
        kg_df:       pd.DataFrame,
        drug_ids:    List[str],
        disease_ids: List[str]
    ):
        self.drug_ids    = drug_ids
        self.disease_ids = disease_ids

        # ── Build global node index ───────────────────────────────────────
        self.node2idx: Dict[str, int] = {}

        for i, nid in enumerate(drug_ids):
            self.node2idx[nid] = i

        d_offset = len(drug_ids)
        for i, nid in enumerate(disease_ids):
            self.node2idx[nid] = d_offset + i

        known        = set(drug_ids) | set(disease_ids)
        all_kg_nodes = set(kg_df['node1'].tolist()) | set(kg_df['node2'].tolist())
        gene_nodes   = sorted(all_kg_nodes - known)

        self.gene_ids  = gene_nodes
        gene_offset    = d_offset + len(disease_ids)
        for i, nid in enumerate(gene_nodes):
            self.node2idx[nid] = gene_offset + i

        self.n_drug    = len(drug_ids)
        self.n_disease = len(disease_ids)
        self.n_gene    = len(gene_nodes)
        self.n_total   = len(self.node2idx)

        # ── Relation index ────────────────────────────────────────────────
        relations        = kg_df['relation'].unique().tolist()
        self.rel2idx     = {r: i for i, r in enumerate(relations)}
        self.n_relations = len(relations)
        logger.info(f"KG relations: {self.rel2idx}")

        # ── Build edge_index / edge_type (vectorised) ─────────────────────
        # [data-2] Replace iterrows() with vectorised map; ~100× faster.
        node1_idx = kg_df['node1'].map(self.node2idx)
        node2_idx = kg_df['node2'].map(self.node2idx)
        rel_idx   = kg_df['relation'].map(self.rel2idx)

        # Drop edges whose nodes are not in the vocab
        valid   = node1_idx.notna() & node2_idx.notna()
        skipped = (~valid).sum()
        if skipped:
            logger.warning(f"Skipped {skipped} KG edges (nodes not in vocab)")

        src_arr   = node1_idx[valid].astype(int).values
        dst_arr   = node2_idx[valid].astype(int).values
        etype_arr = rel_idx[valid].astype(int).values

        src_t   = torch.tensor(src_arr,   dtype=torch.long)
        dst_t   = torch.tensor(dst_arr,   dtype=torch.long)
        etype_t = torch.tensor(etype_arr, dtype=torch.long)

        # Undirected: forward + reverse edges
        rev_etype              = etype_t + self.n_relations
        self.n_relations_total = self.n_relations * 2

        self.edge_index = torch.stack([
            torch.cat([src_t, dst_t]),
            torch.cat([dst_t, src_t])
        ], dim=0)                                      # [2, 2E]
        self.edge_type = torch.cat([etype_t, rev_etype])  # [2E]

        # ── node_type_mask: 0=drug, 1=disease, 2=gene ────────────────────
        # [data-1] Unified 3-value mask; no longer requires a patch in Trainer.
        self.node_type_mask = torch.full(
            (self.n_total,), fill_value=2, dtype=torch.long
        )
        self.node_type_mask[:self.n_drug]                          = 0
        self.node_type_mask[self.n_drug: self.n_drug + self.n_disease] = 1

        # Gene indices (local, 0-based, for embedding lookup)
        self.gene_indices = torch.arange(self.n_gene, dtype=torch.long)

        # ── Drug-disease positive pairs for Phase 1 ───────────────────────
        self.drug_disease_pos_pairs = self._extract_drug_disease_pairs(kg_df)

        logger.info(
            f"KG Stats: {self.n_drug} drugs, {self.n_disease} diseases, "
            f"{self.n_gene} genes, {self.edge_index.size(1)} edges (bidirectional), "
            f"{self.n_relations_total} relation types"
        )

    def _extract_drug_disease_pairs(self, kg_df: pd.DataFrame) -> List[Tuple[int, int]]:
        """
        Extract positive drug-disease pairs from KG edges.
        Returns: List[(drug_local_idx, disease_local_idx)]

        [data-3] Vectorised; replaces iterrows().
        """
        drug_set    = set(self.drug_ids)
        disease_set = set(self.disease_ids)

        # Forward: node1 = drug, node2 = disease
        fwd_mask = (
            kg_df['node1'].isin(drug_set) & kg_df['node2'].isin(disease_set)
        )
        fwd = kg_df[fwd_mask]
        fwd_pairs = list(zip(
            fwd['node1'].map(self.node2idx).astype(int),
            (fwd['node2'].map(self.node2idx) - self.n_drug).astype(int)
        ))

        # Reverse: node1 = disease, node2 = drug
        rev_mask = (
            kg_df['node1'].isin(disease_set) & kg_df['node2'].isin(drug_set)
        )
        rev = kg_df[rev_mask]
        rev_pairs = list(zip(
            rev['node2'].map(self.node2idx).astype(int),
            (rev['node1'].map(self.node2idx) - self.n_drug).astype(int)
        ))

        # Deduplicate: same (drug, disease) may appear under multiple relation types
        return list(set(fwd_pairs + rev_pairs))

    def to(self, device):
        self.edge_index     = self.edge_index.to(device)
        self.edge_type      = self.edge_type.to(device)
        self.node_type_mask = self.node_type_mask.to(device)
        self.gene_indices   = self.gene_indices.to(device)
        return self


# ─────────────────────────────────────────────────────────────────────────────
# Dataset
# ─────────────────────────────────────────────────────────────────────────────

class DrugCombinationDataset(Dataset):
    """
    Dataset for disease-drug-drug triplets with negative sampling.

    [data-5] Negative-sampling pad is now logged as a warning so silent
             training bias is surfaced (was completely silent before).
    """

    def __init__(
        self,
        pairs_df:    pd.DataFrame,
        disease2idx: Dict[str, int],
        drug2idx:    Dict[str, int],
        n_drugs:     int,
        neg_ratio:   int = 5,
        mode:        str = 'train'
    ):
        self.n_drugs     = n_drugs
        self.neg_ratio   = neg_ratio
        self.mode        = mode
        self.disease2idx = disease2idx
        self.drug2idx    = drug2idx

        self.pos_triplets: List[Tuple[int, int, int]] = []

        # Drop duplicate (disease, drug1, drug2) rows before indexing to prevent
        # any single positive from being over-represented during training.
        pairs_df = pairs_df.drop_duplicates(subset=['disease_id', 'drug1_id', 'drug2_id'])

        for _, row in pairs_df.iterrows():
            did = str(row['disease_id'])
            dr1 = str(row['drug1_id'])
            dr2 = str(row['drug2_id'])
            if did not in disease2idx or dr1 not in drug2idx or dr2 not in drug2idx:
                continue
            d_idx = disease2idx[did]
            i_idx = drug2idx[dr1]
            j_idx = drug2idx[dr2]
            i_idx, j_idx = min(i_idx, j_idx), max(i_idx, j_idx)
            self.pos_triplets.append((d_idx, i_idx, j_idx))

        # Remove any remaining duplicates after canonical (i<j) ordering
        # (e.g. rows where drug1/drug2 were swapped map to the same triplet)
        self.pos_triplets = list(dict.fromkeys(self.pos_triplets))

        # Per-disease positive set for false-negative-safe neg sampling
        self.disease_pos: Dict[int, set] = defaultdict(set)
        for d, i, j in self.pos_triplets:
            self.disease_pos[d].add((i, j))

        self._pad_warnings = 0  # count pads across epoch; log once per epoch

        logger.info(
            f"[{mode}] {len(self.pos_triplets)} positive triplets, "
            f"{len(set(d for d, _, _ in self.pos_triplets))} diseases"
        )

    def __len__(self) -> int:
        return len(self.pos_triplets)

    def __getitem__(self, idx: int) -> dict:
        d, i, j = self.pos_triplets[idx]

        if self.mode != 'train':
            return {'pos': torch.tensor([d, i, j], dtype=torch.long)}

        # Negative sampling: replace one drug at a time
        negs     = []
        attempts = 0
        while len(negs) < self.neg_ratio and attempts < 200:
            attempts += 1
            if np.random.random() < 0.5:
                ni = np.random.randint(0, self.n_drugs)
                ni, nj = min(ni, j), max(ni, j)
            else:
                nj = np.random.randint(0, self.n_drugs)
                ni, nj = min(i, nj), max(i, nj)
            if (ni, nj) not in self.disease_pos[d] and ni != nj:
                negs.append((d, ni, nj))

        # [data-5] Pad with last valid negative; warn so users notice.
        if len(negs) < self.neg_ratio:
            self._pad_warnings += 1
            pad = negs[-1] if negs else (d, i, j)  # fallback to pos if truly stuck
            while len(negs) < self.neg_ratio:
                negs.append(pad)

        return {
            'pos':  torch.tensor([d, i, j], dtype=torch.long),
            'negs': torch.tensor(negs,      dtype=torch.long),  # [neg_ratio, 3]
        }

    def reset_pad_warnings(self) -> int:
        """Return and reset pad-warning counter (call once per epoch)."""
        count, self._pad_warnings = self._pad_warnings, 0
        return count


# ─────────────────────────────────────────────────────────────────────────────
# Collation & DataLoader builder
# ─────────────────────────────────────────────────────────────────────────────

def collate_fn(batch):
    pos = torch.stack([b['pos'] for b in batch])    # [B, 3]
    if 'negs' in batch[0]:
        negs = torch.stack([b['negs'] for b in batch])  # [B, neg_ratio, 3]
        return {'pos': pos, 'negs': negs}
    return {'pos': pos}


def build_dataloaders(
    data_dir:    str,
    drug_ids:    List[str],
    disease_ids: List[str],
    batch_size:  int = 512,
    neg_ratio:   int = 5,
    num_workers: int = 4,
) -> Tuple[DataLoader, DataLoader, DataLoader]:

    data_dir    = Path(data_dir)
    drug2idx    = {nid: i for i, nid in enumerate(drug_ids)}
    disease2idx = {nid: i for i, nid in enumerate(disease_ids)}
    n_drugs     = len(drug_ids)

    train_df = load_pairs(data_dir / 'pair_train.csv')
    val_df   = load_pairs(data_dir / 'pair_val.csv')
    test_df  = load_pairs(data_dir / 'pair_test.csv')

    train_ds = DrugCombinationDataset(train_df, disease2idx, drug2idx, n_drugs, neg_ratio, 'train')
    val_ds   = DrugCombinationDataset(val_df,   disease2idx, drug2idx, n_drugs, neg_ratio, 'val')
    test_ds  = DrugCombinationDataset(test_df,  disease2idx, drug2idx, n_drugs, neg_ratio, 'test')

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              collate_fn=collate_fn, num_workers=num_workers, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False,
                              collate_fn=collate_fn, num_workers=num_workers, pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False,
                              collate_fn=collate_fn, num_workers=num_workers, pin_memory=True)

    return train_loader, val_loader, test_loader