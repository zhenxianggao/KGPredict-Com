"""
Training and evaluation for Drug Combination Ranking Model

Three-phase training:
  Phase 1: Drug-Disease relevance  (KG drug-disease edges)
  Phase 2: Drug-Drug synergy       (GT pairs, disease-agnostic)
  Phase 3: Joint fine-tuning       (full disease-drug_i-drug_j GT)

Key design for Phase 1 & 2 — "two-stage gradient" pattern
──────────────────────────────────────────────────────────
Problem: the encoder (3-layer CompGCN+RAAG over 420k edges) takes ~10-15s
per forward pass. With batch_size=512 and 27k pairs, a naive per-batch
encode gives 53 encoder passes/epoch = 10+ min/epoch = 80 hours total.

Solution: split each epoch into two stages.

  Stage A (batch loop): scorer MLP only
    - encode_all() ONCE under no_grad -> detach -> store as drug/disease tables
    - iterate batches: scorer MLP forward/backward using DETACHED embeddings
    - scorer_optimizer.step() per batch  (encoder params excluded -- fix-D)

  Stage B (epoch end): encoder update
    - re-encode a random subsample WITH grad  (enc_samples <= 4096)
    - BPR loss on subsample + optional GCL (drug + disease nodes -- fix-F)
    - full optimizer.step()  (updates encoder + all shared params)

Result: 3 encoder passes per epoch (Stage A no_grad, Stage B grad, GCL no_grad)
regardless of dataset size. Scorer MLP batches are ~1ms each (pure MLP).

[gcl]  Graph Contrastive Learning: one edge-dropped view encoded under
       no_grad at epoch start; NT-Xent applied in Stage B for both drug
       and disease nodes.

[ohnm] Online Hard Negative Mining in Phase 3: re-mine every ohnm_interval
       epochs and mix with DataLoader's random negatives.

Bug fixes applied:
  [fix-A] OHNM fallback no longer inserts the positive triplet as a negative.
          Fallback now random-samples a valid (ni != nj, not a known positive)
          pair until n_hard slots are filled.
  [fix-B] OHNM candidate pool now filters out known GT positives for the
          disease before selecting hard negatives, preventing false negatives.
  [fix-C] S2 auxiliary loss in Phase 3 now covers both drug_i and drug_j
          symmetrically, matching the score_triplets() definition.
  [fix-D] Stage A uses a dedicated scorer_optimizer (excludes encoder params)
          so AdamW weight decay is not spuriously applied to the encoder
          thousands of times with zero gradient.
  [fix-E] Phase 2 negative sampling filters out self-pairs (neg_i == neg_j)
          to avoid degenerate S1 MLP inputs [2e, e^2, 0].
  [fix-F] Phase 1 GCL now contrasts both drug and disease nodes, matching
          the drug-disease relevance objective of Phase 1.
"""

import logging
import os
import time
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

logger = logging.getLogger(__name__)


# -----------------------------------------------------------------------------
# Loss functions
# -----------------------------------------------------------------------------

def bpr_loss(pos_scores: torch.Tensor, neg_scores: torch.Tensor) -> torch.Tensor:
    if neg_scores.dim() == 1:
        neg_scores = neg_scores.unsqueeze(1)
    return -F.logsigmoid(pos_scores.unsqueeze(1) - neg_scores).mean()


def listnet_loss(pos_scores: torch.Tensor, neg_scores: torch.Tensor) -> torch.Tensor:
    all_scores   = torch.cat([pos_scores.unsqueeze(1), neg_scores], dim=1)
    labels       = torch.zeros_like(all_scores)
    labels[:, 0] = 1.0
    return -(labels * F.log_softmax(all_scores, dim=1)).sum(dim=1).mean()


def combined_ranking_loss(pos_scores, neg_scores, alpha=0.5):
    return (alpha * bpr_loss(pos_scores, neg_scores) +
            (1 - alpha) * listnet_loss(pos_scores, neg_scores))


def nt_xent_loss(z1: torch.Tensor, z2: torch.Tensor,
                 temperature: float = 0.5) -> torch.Tensor:
    """NT-Xent contrastive loss. z1, z2: [N, d]."""
    N   = z1.size(0)
    z1  = F.normalize(z1, dim=1)
    z2  = F.normalize(z2, dim=1)
    z   = torch.cat([z1, z2], dim=0)
    sim = torch.mm(z, z.t()) / temperature
    sim = sim.masked_fill(
        torch.eye(2*N, dtype=torch.bool, device=z.device), float('-inf'))
    labels = torch.cat([torch.arange(N, 2*N, device=z.device),
                        torch.arange(0, N,   device=z.device)])
    return F.cross_entropy(sim, labels)


# -----------------------------------------------------------------------------
# KG edge dropout  [gcl]
# -----------------------------------------------------------------------------

def _drop_edges(edge_index, edge_type, drop_rate):
    keep = torch.rand(edge_index.size(1), device=edge_index.device) > drop_rate
    return edge_index[:, keep], edge_type[keep]


# -----------------------------------------------------------------------------
# Metrics
# -----------------------------------------------------------------------------

def compute_ranking_metrics(ranks, ks=None):
    if ks is None:
        ks = [10, 20, 50, 100]
    arr = np.array(ranks, dtype=np.float64)
    m   = {'MRR': float(np.mean(1.0/arr)),
           'MeanRank': float(np.mean(arr)),
           'MedianRank': float(np.median(arr))}
    for k in ks:
        m[f'Hits@{k}'] = float(np.mean(arr <= k))
    return m


def _compute_auc_aupr_single(all_scores, pos_flat):
    n_pos, n_total = len(pos_flat), len(all_scores)
    n_neg = n_total - n_pos
    if n_pos == 0 or n_neg == 0:
        return float('nan'), float('nan')
    pos_scores = all_scores[pos_flat]
    sorted_asc = np.sort(all_scores)
    rank_lo    = np.searchsorted(sorted_asc, pos_scores, side='left')
    rank_hi    = np.searchsorted(sorted_asc, pos_scores, side='right')
    ranks_asc  = (rank_lo + rank_hi) / 2.0 + 1
    auc = float(np.clip(
        (ranks_asc.sum() - n_pos*(n_pos+1)/2) / (n_pos*n_neg), 0.0, 1.0))
    sorted_pos = np.sort(pos_scores)
    rank_desc  = n_total - np.searchsorted(sorted_asc, pos_scores, side='right') + 1
    n_above    = n_pos - np.searchsorted(sorted_pos, pos_scores, side='left')
    aupr = float(np.mean(n_above / rank_desc))
    return auc, aupr


def _flat_pair_index(pi, pj, n_drugs):
    N, pi, pj = n_drugs, pi.long(), pj.long()
    if __debug__:
        assert (pi >= 0).all() and (pj > pi).all() and (pj < N).all()
    return pi * (2*N - pi - 1) // 2 + (pj - pi - 1)


# -----------------------------------------------------------------------------
# Trainer
# -----------------------------------------------------------------------------

class Trainer:
    def __init__(self, model, kg_graph, config: dict, device: torch.device):
        self.model  = model.to(device)
        self.kg     = kg_graph.to(device)
        self.config = config
        self.device = device

        self._drug_indices    = torch.arange(config['n_drugs'],    device=device)
        self._disease_indices = torch.arange(config['n_diseases'], device=device)

        # Full optimizer: encoder + all scorers (used in Stage B and Phase 3)
        self.optimizer = AdamW(
            model.parameters(),
            lr=config.get('lr', 1e-3),
            weight_decay=config.get('weight_decay', 1e-4),
        )

        # [fix-D] Scorer-only optimizer: used in Stage A so that encoder params
        # never receive spurious AdamW weight decay when their gradient is zero.
        scorer_params = (
            list(model.s1_synergy.parameters()) +
            list(model.s2_relevance.parameters()) +
            list(model.s3_threeway.parameters()) +
            [model.score_weights]
        )
        self.scorer_optimizer = AdamW(
            scorer_params,
            lr=config.get('lr', 1e-3),
            weight_decay=config.get('weight_decay', 1e-4),
        )

        n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logger.info(f"Model total parameters: {n_params:,}")

    # -------------------------------------------------------------------------
    # Helpers
    # -------------------------------------------------------------------------

    def _kg_args(self):
        return (self._drug_indices, self._disease_indices,
                self.kg.gene_indices, self.kg.node_type_mask,
                self.kg.edge_index, self.kg.edge_type)

    def _kg_args_aug(self, edge_drop):
        ei, et = _drop_edges(self.kg.edge_index, self.kg.edge_type, edge_drop)
        return (self._drug_indices, self._disease_indices,
                self.kg.gene_indices, self.kg.node_type_mask, ei, et)

    def _reset_optimizer_state(self, new_lr: float):
        for opt in (self.optimizer, self.scorer_optimizer):
            opt.state.clear()
            for pg in opt.param_groups:
                pg['lr'] = new_lr
        logger.info(f"  Optimizer state cleared; LR reset to {new_lr:.2e}")

    @torch.no_grad()
    def encode_all(self):
        self.model.eval()
        return self.model.encode_all(*self._kg_args())

    @torch.no_grad()
    def _build_pair_index(self):
        if hasattr(self, '_pair_i'):
            return self._pair_i, self._pair_j
        n = self.kg.n_drug
        logger.info(f"  Building pair index: {n} drugs, {n*(n-1)//2:,} pairs ...")
        idx = torch.triu_indices(n, n, offset=1, device=self.device)
        self._pair_i = idx[0].to(torch.int32)
        self._pair_j = idx[1].to(torch.int32)
        return self._pair_i, self._pair_j

    # -------------------------------------------------------------------------
    # Phase 1 - Drug-Disease Relevance
    # -------------------------------------------------------------------------

    def train_phase1(self, n_epochs: int = 20):
        """
        Two-stage gradient pattern (see module docstring).

        Stage A: no_grad encode -> detach -> S2 MLP batches -> scorer_optimizer step
                 [fix-D] scorer_optimizer excludes encoder params -> no spurious decay
        Stage B: with_grad encode (subsample) -> BPR + GCL -> full optimizer step
                 [fix-F] GCL contrasts both drug AND disease nodes
        """
        logger.info("=" * 60)
        logger.info("Phase 1: Drug-Disease Relevance Training")
        logger.info("=" * 60)

        pos_pairs = self.kg.drug_disease_pos_pairs
        if not pos_pairs:
            logger.warning("No drug-disease edges in KG - skipping Phase 1.")
            return

        n_drugs     = self.kg.n_drug
        batch_size  = self.config.get('phase1_batch_size', 1024)
        neg_k       = self.config.get('phase1_neg_k', 10)
        gcl_lambda  = self.config.get('gcl_lambda', 0.1)
        edge_drop   = self.config.get('gcl_edge_drop', 0.1)
        gcl_temp    = self.config.get('gcl_temperature', 0.5)
        enc_samples = min(len(pos_pairs),
                          self.config.get('phase1_enc_samples', 4096))

        scheduler        = CosineAnnealingLR(self.optimizer,        T_max=n_epochs)
        scorer_scheduler = CosineAnnealingLR(self.scorer_optimizer, T_max=n_epochs)
        pos_tensor = torch.tensor(pos_pairs, dtype=torch.long, device=self.device)

        logger.info(f"  Pairs: {len(pos_tensor):,} | batch: {batch_size} | "
                    f"neg_k: {neg_k} | gcl_lambda: {gcl_lambda} | "
                    f"enc_samples: {enc_samples}")

        for epoch in range(n_epochs):
            t_epoch = time.time()
            self.model.train()

            # [gcl] One augmented view, no_grad -- encode both drug AND disease
            if gcl_lambda > 0:
                self.model.eval()
                with torch.no_grad():
                    drug_v1, disease_v1 = self.model.encode_all(
                        *self._kg_args_aug(edge_drop))
                drug_v1    = drug_v1.detach()
                disease_v1 = disease_v1.detach()   # [fix-F]
                self.model.train()
            else:
                drug_v1 = disease_v1 = None

            # -- Stage A: detached encode, scorer MLP batches ---------------
            self.model.eval()
            with torch.no_grad():
                drug_det, disease_det = self.model.encode_all(*self._kg_args())
            drug_det    = drug_det.detach()
            disease_det = disease_det.detach()
            self.model.train()

            perm       = torch.randperm(len(pos_tensor), device=self.device)
            pos_perm   = pos_tensor[perm]
            total_loss = 0.0
            n_batches  = 0

            for start in range(0, len(pos_perm), batch_size):
                batch       = pos_perm[start: start + batch_size]
                drug_idx    = batch[:, 0]
                disease_idx = batch[:, 1]
                B           = len(batch)

                pos_scores = self.model.s2_relevance(
                    disease_det[disease_idx], drug_det[drug_idx])
                neg_idx    = torch.randint(0, n_drugs, (B, neg_k), device=self.device)
                neg_scores = torch.stack([
                    self.model.s2_relevance(
                        disease_det[disease_idx], drug_det[neg_idx[:, k]])
                    for k in range(neg_k)
                ], dim=1)
                loss = bpr_loss(pos_scores, neg_scores)

                # [fix-D] scorer_optimizer only -- encoder params excluded,
                # no spurious weight decay on zero-gradient encoder weights.
                self.scorer_optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.scorer_optimizer.step()

                total_loss += loss.item()
                n_batches  += 1

            # -- Stage B: encoder update on subsample -----------------------
            sub_idx = torch.randperm(len(pos_tensor),
                                     device=self.device)[:enc_samples]
            sub   = pos_tensor[sub_idx]
            d_s   = sub[:, 0]   # drug indices
            dis_s = sub[:, 1]   # disease indices
            Bs    = len(sub)

            drug_g, disease_g = self.model.encode_all(*self._kg_args())

            pos_s = self.model.s2_relevance(disease_g[dis_s], drug_g[d_s])
            nid   = torch.randint(0, n_drugs, (Bs, neg_k), device=self.device)
            neg_s = torch.stack([
                self.model.s2_relevance(disease_g[dis_s], drug_g[nid[:, k]])
                for k in range(neg_k)
            ], dim=1)
            enc_loss = bpr_loss(pos_s, neg_s)

            if gcl_lambda > 0 and drug_v1 is not None:
                uniq_drug = d_s.unique()
                uniq_dis  = dis_s.unique()
                # [fix-F] Contrastive loss on BOTH drug and disease nodes
                enc_loss = (
                    enc_loss
                    + gcl_lambda * nt_xent_loss(
                        drug_g[uniq_drug], drug_v1[uniq_drug],
                        temperature=gcl_temp)
                    + gcl_lambda * nt_xent_loss(
                        disease_g[uniq_dis], disease_v1[uniq_dis],
                        temperature=gcl_temp)
                )

            self.optimizer.zero_grad()
            enc_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()

            scheduler.step()
            scorer_scheduler.step()

            if (epoch + 1) % 5 == 0:
                logger.info(
                    f"  [P1] Epoch {epoch+1:3d}/{n_epochs} | "
                    f"Loss(A): {total_loss/n_batches:.4f} | "
                    f"Loss(B): {enc_loss.item():.4f} | "
                    f"LR: {scheduler.get_last_lr()[0]:.2e} | "
                    f"time: {time.time()-t_epoch:.1f}s"
                )

        logger.info("Phase 1 complete.")

    # -------------------------------------------------------------------------
    # Phase 2 - Drug-Drug Synergy
    # -------------------------------------------------------------------------

    def train_phase2(self, train_loader, n_epochs: int = 20):
        """
        Same two-stage gradient pattern applied to S1 (drug-drug synergy).

        [fix-D] Stage A uses scorer_optimizer (encoder excluded).
        [fix-E] Negative self-pairs (neg_i == neg_j) are filtered out.
        """
        logger.info("=" * 60)
        logger.info("Phase 2: Drug-Drug Synergy Training")
        logger.info("=" * 60)

        n_drugs     = self.kg.n_drug
        neg_k       = self.config.get('phase2_neg_k', 10)
        batch_size  = self.config.get('phase2_batch_size', 512)
        gcl_lambda  = self.config.get('gcl_lambda', 0.1)
        edge_drop   = self.config.get('gcl_edge_drop', 0.1)
        gcl_temp    = self.config.get('gcl_temperature', 0.5)

        self._reset_optimizer_state(self.config.get('lr', 1e-3))
        scheduler        = CosineAnnealingLR(self.optimizer,        T_max=n_epochs)
        scorer_scheduler = CosineAnnealingLR(self.scorer_optimizer, T_max=n_epochs)

        unique_pairs = set()
        for batch in train_loader:
            for row in batch['pos']:
                i, j = int(row[1]), int(row[2])
                unique_pairs.add((min(i, j), max(i, j)))
        pos_tensor = torch.tensor(list(unique_pairs), dtype=torch.long,
                                  device=self.device)

        enc_samples = min(len(pos_tensor),
                          self.config.get('phase2_enc_samples', 4096))

        logger.info(f"  Unique drug pairs: {len(pos_tensor):,} | "
                    f"batch: {batch_size} | neg_k: {neg_k} | "
                    f"gcl_lambda: {gcl_lambda} | enc_samples: {enc_samples}")

        for epoch in range(n_epochs):
            t_epoch = time.time()
            self.model.train()

            # [gcl] Augmented view
            if gcl_lambda > 0:
                self.model.eval()
                with torch.no_grad():
                    drug_v1, _ = self.model.encode_all(*self._kg_args_aug(edge_drop))
                drug_v1 = drug_v1.detach()
                self.model.train()
            else:
                drug_v1 = None

            # -- Stage A ---------------------------------------------------
            self.model.eval()
            with torch.no_grad():
                drug_det, _ = self.model.encode_all(*self._kg_args())
            drug_det = drug_det.detach()
            self.model.train()

            perm       = torch.randperm(len(pos_tensor), device=self.device)
            pos_perm   = pos_tensor[perm]
            total_loss = 0.0
            n_batches  = 0

            for start in range(0, len(pos_perm), batch_size):
                batch = pos_perm[start: start + batch_size]
                i_idx = batch[:, 0]
                j_idx = batch[:, 1]
                B     = len(batch)

                pos_scores = self.model.s1_synergy(drug_det[i_idx], drug_det[j_idx])

                # [fix-E] Filter out self-pairs from negative sampling
                neg_i = torch.randint(0, n_drugs, (B, neg_k), device=self.device)
                neg_j = torch.randint(0, n_drugs, (B, neg_k), device=self.device)
                bad = neg_i == neg_j
                while bad.any():
                    neg_j[bad] = torch.randint(
                        0, n_drugs, (int(bad.sum()),), device=self.device)
                    bad = neg_i == neg_j

                neg_scores = torch.stack([
                    self.model.s1_synergy(drug_det[neg_i[:,k]], drug_det[neg_j[:,k]])
                    for k in range(neg_k)
                ], dim=1)
                loss = bpr_loss(pos_scores, neg_scores)

                # [fix-D] scorer_optimizer only
                self.scorer_optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.scorer_optimizer.step()

                total_loss += loss.item()
                n_batches  += 1

            # -- Stage B ---------------------------------------------------
            sub_idx = torch.randperm(len(pos_tensor),
                                     device=self.device)[:enc_samples]
            sub = pos_tensor[sub_idx]
            i_s = sub[:, 0]
            j_s = sub[:, 1]
            Bs  = len(sub)

            drug_g, _ = self.model.encode_all(*self._kg_args())

            pos_s = self.model.s1_synergy(drug_g[i_s], drug_g[j_s])

            # [fix-E] Filter self-pairs in Stage B negatives too
            ni = torch.randint(0, n_drugs, (Bs, neg_k), device=self.device)
            nj = torch.randint(0, n_drugs, (Bs, neg_k), device=self.device)
            bad = ni == nj
            while bad.any():
                nj[bad] = torch.randint(
                    0, n_drugs, (int(bad.sum()),), device=self.device)
                bad = ni == nj

            neg_s = torch.stack([
                self.model.s1_synergy(drug_g[ni[:,k]], drug_g[nj[:,k]])
                for k in range(neg_k)
            ], dim=1)
            enc_loss = bpr_loss(pos_s, neg_s)

            if gcl_lambda > 0 and drug_v1 is not None:
                uniq     = torch.cat([i_s, j_s]).unique()
                enc_loss = enc_loss + gcl_lambda * nt_xent_loss(
                    drug_g[uniq], drug_v1[uniq], temperature=gcl_temp)

            self.optimizer.zero_grad()
            enc_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()

            scheduler.step()
            scorer_scheduler.step()

            pad_n = train_loader.dataset.reset_pad_warnings()
            if pad_n:
                logger.warning(f"  [P2] Epoch {epoch+1}: {pad_n} neg-sampling pads")

            if (epoch + 1) % 5 == 0:
                logger.info(
                    f"  [P2] Epoch {epoch+1:3d}/{n_epochs} | "
                    f"Loss(A): {total_loss/n_batches:.4f} | "
                    f"Loss(B): {enc_loss.item():.4f} | "
                    f"LR: {scheduler.get_last_lr()[0]:.2e} | "
                    f"time: {time.time()-t_epoch:.1f}s"
                )

        logger.info("Phase 2 complete.")

    # -------------------------------------------------------------------------
    # OHNM helper  [ohnm]
    # -------------------------------------------------------------------------

    @torch.no_grad()
    def _mine_hard_negatives(self, all_pos, neg_ratio, pool_factor=50):
        """
        Mine hard negatives for each positive triplet (d, i, j).

        [fix-B] Candidate pool is filtered to exclude known GT positives for
                disease d, preventing false negatives from poisoning training.
        """
        self.model.eval()
        drug_embs, disease_embs = self.encode_all()
        w  = torch.softmax(self.model.score_weights, dim=0)
        w1, w2  = w[0].item(), w[1].item()
        n_drugs = self.kg.n_drug

        # Build per-disease positive set for false-negative filtering [fix-B]
        disease_pos: Dict[int, set] = defaultdict(set)
        for d, i, j in all_pos:
            ci, cj = min(i, j), max(i, j)
            disease_pos[d].add((ci, cj))

        result: Dict[tuple, list] = {}

        for d, i, j in all_pos:
            pos_set = disease_pos[d]
            pool    = neg_ratio * pool_factor
            ci      = torch.randint(0, n_drugs, (pool,), device=self.device)
            cj      = torch.randint(0, n_drugs, (pool,), device=self.device)

            # [fix-B] Filter: remove self-pairs AND known GT positives for d
            valid_mask = ci != cj
            ci_list = ci[valid_mask].tolist()
            cj_list = cj[valid_mask].tolist()
            filtered_i, filtered_j = [], []
            for a, b in zip(ci_list, cj_list):
                ca, cb = min(a, b), max(a, b)
                if (ca, cb) not in pos_set:
                    filtered_i.append(a)
                    filtered_j.append(b)

            if not filtered_i:
                continue

            ci = torch.tensor(filtered_i, dtype=torch.long, device=self.device)
            cj = torch.tensor(filtered_j, dtype=torch.long, device=self.device)

            e_d    = disease_embs[d].unsqueeze(0).expand(ci.size(0), -1)
            score  = (w1 * self.model.s1_synergy(drug_embs[ci], drug_embs[cj]) +
                      w2 * (self.model.s2_relevance(e_d, drug_embs[ci]) +
                            self.model.s2_relevance(e_d, drug_embs[cj])))
            k      = min(neg_ratio, ci.size(0))
            _, top = torch.topk(score, k=k)
            result[(d, i, j)] = [(d, int(ci[t]), int(cj[t])) for t in top.tolist()]

        self.model.train()
        return result

    # -------------------------------------------------------------------------
    # Phase 3 - Joint Fine-tuning  (+OHNM)
    # -------------------------------------------------------------------------

    def train_phase3(self, train_loader, val_loader,
                     n_epochs=100, checkpoint_path='best_model.pt',
                     stage2_k=2000):
        """
        End-to-end fine-tuning with per-batch encode_all() (required for S3
        to receive disease-specific gradients). Checkpoint on Val MRR.

        [ohnm]  Hard negatives re-mined every ohnm_interval epochs.
        [fix-A] OHNM fallback uses random sampling, not the positive itself.
        [fix-C] S2 auxiliary loss covers both drug_i and drug_j symmetrically.
        """
        logger.info("=" * 60)
        logger.info("Phase 3: Joint Fine-tuning")
        logger.info("=" * 60)

        for pg in self.optimizer.param_groups:
            pg['lr'] = self.config.get('phase3_lr', 1e-4)
        for pg in self.scorer_optimizer.param_groups:
            pg['lr'] = self.config.get('phase3_lr', 1e-4)
        self.optimizer.state.clear()
        self.scorer_optimizer.state.clear()
        logger.info(f"  LR: {self.config.get('phase3_lr', 1e-4):.2e}")

        scheduler     = CosineAnnealingLR(self.optimizer, T_max=n_epochs)
        best_mrr      = 0.0
        patience      = self.config.get('patience', 10)
        no_improve    = 0
        lam           = self.config.get('aux_lambda', 0.1)
        ohnm_interval = self.config.get('ohnm_interval', 10)
        ohnm_mix      = self.config.get('ohnm_mix_ratio', 0.5)
        neg_ratio     = self.config.get('neg_ratio', 5)
        n_drugs       = self.kg.n_drug

        logger.info(f"  stage2_k={stage2_k} | patience={patience} | "
                    f"ohnm_interval={ohnm_interval} | ohnm_mix={ohnm_mix}")

        all_pos = list({
            (int(r[0]), int(r[1]), int(r[2]))
            for batch in train_loader for r in batch['pos']
        })
        # Build GT positive set for OHNM filtering [fix-A, fix-B]
        train_pos_set: Dict[int, set] = defaultdict(set)
        for d, i, j in all_pos:
            ci, cj = min(i, j), max(i, j)
            train_pos_set[d].add((ci, cj))

        hard_neg_dict: dict = {}

        for epoch in range(n_epochs):
            self.model.train()

            if epoch % ohnm_interval == 0:
                logger.info(f"  [OHNM] Mining hard negatives (epoch {epoch+1})...")
                t0 = time.time()
                hard_neg_dict = self._mine_hard_negatives(
                    all_pos, neg_ratio=neg_ratio, pool_factor=50)
                logger.info(f"  [OHNM] Done in {time.time()-t0:.1f}s")
                self.model.train()

            total_loss = 0.0
            n_batches  = 0

            for batch in train_loader:
                pos  = batch['pos'].to(self.device)
                negs = batch['negs'].to(self.device)
                B, K, _ = negs.shape

                if hard_neg_dict and ohnm_mix > 0:
                    n_hard    = max(1, int(K * ohnm_mix))
                    hard_rows = []
                    for b in range(B):
                        key = (int(pos[b,0]), int(pos[b,1]), int(pos[b,2]))
                        hn  = hard_neg_dict.get(key, [])
                        row = list(hn[:n_hard])

                        # [fix-A] Fallback: random sampling, NEVER use the positive
                        d_b     = int(pos[b, 0])
                        pos_set = train_pos_set.get(d_b, set())
                        attempts = 0
                        while len(row) < n_hard and attempts < 200:
                            attempts += 1
                            ni = int(torch.randint(0, n_drugs, (1,)).item())
                            nj = int(torch.randint(0, n_drugs, (1,)).item())
                            if ni == nj:
                                continue
                            ca, cb = min(ni, nj), max(ni, nj)
                            if (ca, cb) not in pos_set:
                                row.append((d_b, ni, nj))

                        # Last-resort: borrow from DataLoader random negatives
                        # (extremely unlikely after 200 attempts)
                        while len(row) < n_hard:
                            row.append(tuple(negs[b, len(row) % K].tolist()))

                        hard_rows.append(torch.tensor(row[:n_hard],
                                                      dtype=torch.long,
                                                      device=self.device))
                    hard_t = torch.stack(hard_rows, 0)      # [B, n_hard, 3]
                    negs   = torch.cat([hard_t, negs[:, n_hard:]], dim=1)

                drug_embs, disease_embs = self.model.encode_all(*self._kg_args())

                out_pos    = self.model.score_triplets(
                    disease_embs[pos[:,0]], drug_embs[pos[:,1]], drug_embs[pos[:,2]])
                pos_scores = out_pos['final']

                neg_flat   = negs.reshape(B*K, 3)
                out_neg    = self.model.score_triplets(
                    disease_embs[neg_flat[:,0]],
                    drug_embs[neg_flat[:,1]],
                    drug_embs[neg_flat[:,2]])
                neg_scores = out_neg['final'].reshape(B, K)

                # [fix-C] S2 aux loss: symmetric over drug_i AND drug_j
                # pos S2 = S2(d, drug_i) + S2(d, drug_j)
                # neg S2 = S2(d, neg_1)  + S2(d, neg_2)   for each k
                pos_s2 = (
                    self.model.s2_relevance(
                        disease_embs[pos[:,0]], drug_embs[pos[:,1]])
                    + self.model.s2_relevance(
                        disease_embs[pos[:,0]], drug_embs[pos[:,2]])
                )
                neg_s2 = torch.stack([
                    self.model.s2_relevance(
                        disease_embs[neg_flat[k::K, 0]],
                        drug_embs[neg_flat[k::K, 1]])
                    + self.model.s2_relevance(
                        disease_embs[neg_flat[k::K, 0]],
                        drug_embs[neg_flat[k::K, 2]])
                    for k in range(K)
                ], dim=1)

                loss = (
                    combined_ranking_loss(pos_scores, neg_scores)
                    + lam * bpr_loss(out_pos['s1'], out_neg['s1'].reshape(B, K))
                    + lam * bpr_loss(pos_s2, neg_s2)
                )

                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()
                total_loss += loss.item()
                n_batches  += 1

            scheduler.step()

            pad_n = train_loader.dataset.reset_pad_warnings()
            if pad_n:
                logger.warning(f"  [P3] Epoch {epoch+1}: {pad_n} neg-sampling pads")

            if (epoch + 1) % 2 == 0:
                metrics = self.evaluate_full_space(val_loader, stage2_k=stage2_k)
                mrr     = metrics.get('MRR', 0.0)
                w       = torch.softmax(self.model.score_weights, dim=0).detach().cpu()
                logger.info(
                    f"  [P3] Epoch {epoch+1:3d}/{n_epochs} | "
                    f"TrainLoss: {total_loss/n_batches:.4f} | "
                    f"MRR: {mrr:.4f} | "
                    f"Hits@10: {metrics.get('Hits@10',0):.4f} | "
                    f"Hits@20: {metrics.get('Hits@20',0):.4f} | "
                    f"Hits@50: {metrics.get('Hits@50',0):.4f} | "
                    f"Hits@100: {metrics.get('Hits@100',0):.4f} | "
                    f"AUC: {metrics.get('AUC',0):.4f} | "
                    f"AUPR: {metrics.get('AUPR',0):.4f} | "
                    f"MeanRank: {metrics.get('MeanRank',0):.1f} | "
                    f"MedianRank: {metrics.get('MedianRank',0):.1f} | "
                    f"w=[S1:{w[0]:.2f} S2:{w[1]:.2f} S3:{w[2]:.2f}] | "
                    f"LR: {scheduler.get_last_lr()[0]:.2e}"
                )
                if mrr > best_mrr:
                    best_mrr = mrr
                    torch.save(self.model.state_dict(), checkpoint_path)
                    no_improve = 0
                    logger.info(f"  Best MRR={best_mrr:.4f} -> {checkpoint_path}")
                else:
                    no_improve += 1
                    logger.info(f"  No improve {no_improve}/{patience}")
                    if no_improve >= patience:
                        logger.info(f"  Early stopping at epoch {epoch+1}.")
                        break

        if os.path.exists(checkpoint_path):
            self.model.load_state_dict(
                torch.load(checkpoint_path, map_location=self.device,
                           weights_only=True))
            logger.info(f"Restored best model (Val MRR={best_mrr:.4f})")
        logger.info("Phase 3 complete.")

    # -------------------------------------------------------------------------
    # Evaluation
    # -------------------------------------------------------------------------

    @torch.no_grad()
    def evaluate_full_space(self, loader, ks=None, stage2_k=2000):
        if ks is None:
            ks = [10, 20, 50, 100]
        self.model.eval()
        device      = self.device
        n_drugs     = self.kg.n_drug
        total_pairs = n_drugs * (n_drugs - 1) // 2

        drug_embs, disease_embs = self.encode_all()
        pair_i, pair_j = self._build_pair_index()

        logger.info("  [Eval] Precomputing S1 ...")
        t0     = time.time()
        s1_all = self.model.precompute_s1_scores_chunked(
            drug_embs, pair_i, pair_j, batch_size=50000)
        logger.info(f"  [Eval] S1 done in {time.time()-t0:.1f}s")

        w = torch.softmax(self.model.score_weights, dim=0)
        w1, w2, w3 = w[0].item(), w[1].item(), w[2].item()

        disease_to_pos  = defaultdict(set)
        all_pos_records = []
        seen            = set()
        for batch in loader:
            for row in batch['pos']:
                d, i, j = int(row[0]), int(row[1]), int(row[2])
                ci, cj  = min(i,j), max(i,j)
                disease_to_pos[d].add((ci, cj))
                key = (d, ci, cj)
                if key not in seen:
                    seen.add(key)
                    all_pos_records.append(key)

        n_records = len(all_pos_records)
        n_unique  = len(disease_to_pos)
        logger.info(f"  [Eval] {n_records} records | {n_unique} diseases | "
                    f"{total_pairs:,} pairs | stage2_k={stage2_k}")

        ranks_per_record = {}
        auc_list, aupr_list = [], []

        for dis_num, (dis_idx, pos_set) in enumerate(disease_to_pos.items(), 1):
            t0  = time.time()
            e_d = disease_embs[dis_idx]

            s2_per  = self.model.precompute_s2_scores(e_d, drug_embs, 4096)
            s2_all  = s2_per[pair_i.long()] + s2_per[pair_j.long()]
            coarse  = w1*s1_all + w2*s2_all

            k2           = min(stage2_k, total_pairs)
            _, top_flat  = torch.topk(coarse, k=k2)
            top_i, top_j = pair_i.long()[top_flat], pair_j.long()[top_flat]
            s3_top       = self.model.score_s3_pairs(e_d, drug_embs, top_i, top_j, 4096)
            final_top    = w1*s1_all[top_flat] + w2*s2_all[top_flat] + w3*s3_top

            final_scores           = coarse.clone()
            final_scores[top_flat] = final_top

            pos_list = list(pos_set)
            pos_t    = torch.tensor(pos_list, dtype=torch.long, device=device)
            flat_idx = _flat_pair_index(pos_t[:,0], pos_t[:,1], n_drugs)

            sorted_desc = torch.argsort(final_scores, descending=True)
            rank_lookup = torch.empty_like(sorted_desc)
            rank_lookup[sorted_desc] = torch.arange(total_pairs, device=device) + 1
            ranks_gpu = rank_lookup[flat_idx].cpu().tolist()

            for k, pair in enumerate(pos_list):
                ranks_per_record[(dis_idx, pair[0], pair[1])] = int(ranks_gpu[k])

            sc_np  = final_scores.cpu().numpy()
            fi_np  = flat_idx.cpu().numpy()
            auc, aupr = _compute_auc_aupr_single(sc_np, fi_np)
            del sc_np, final_scores, s2_all, coarse

            if not np.isnan(auc):  auc_list.append(auc)
            if not np.isnan(aupr): aupr_list.append(aupr)

            #logger.info(
            #    f"  [Eval] dis {dis_num}/{n_unique} (idx={dis_idx}) | "
            #    f"pos={len(pos_list)} | AUC={auc:.4f} AUPR={aupr:.4f} | "
            #    f"{time.time()-t0:.1f}s"
            #)

        all_ranks = [ranks_per_record[(d,i,j)] for d,i,j in all_pos_records]
        metrics   = {**compute_ranking_metrics(all_ranks, ks),
                     'AUC':  float(np.mean(auc_list))  if auc_list  else float('nan'),
                     'AUPR': float(np.mean(aupr_list)) if aupr_list else float('nan')}
        logger.info("  [Eval] " +
                    " | ".join(f"{k}: {v:.4f}" for k,v in sorted(metrics.items())))
        return metrics

    # -------------------------------------------------------------------------
    # Inference
    # -------------------------------------------------------------------------

    @torch.no_grad()
    def rank_all_pairs_for_disease(self, disease_id, disease2idx, drug_ids,
                                   top_k=100, stage2_k=2000):
        if len(drug_ids) != self.kg.n_drug:
            raise ValueError(
                f"len(drug_ids)={len(drug_ids)} != kg.n_drug={self.kg.n_drug}")
        self.model.eval()
        n_drugs     = self.kg.n_drug
        total_pairs = n_drugs * (n_drugs - 1) // 2
        drug_embs, disease_embs = self.encode_all()
        pair_i, pair_j = self._build_pair_index()
        e_d = disease_embs[disease2idx[disease_id]]

        w  = torch.softmax(self.model.score_weights, dim=0)
        w1, w2, w3 = w[0].item(), w[1].item(), w[2].item()

        s1_all  = self.model.precompute_s1_scores_chunked(
            drug_embs, pair_i, pair_j, 50000)
        s2_per  = self.model.precompute_s2_scores(e_d, drug_embs, 4096)
        s2_all  = s2_per[pair_i.long()] + s2_per[pair_j.long()]
        coarse  = w1*s1_all + w2*s2_all

        k2           = min(stage2_k, total_pairs)
        _, top_flat  = torch.topk(coarse, k=k2)
        top_i, top_j = pair_i.long()[top_flat], pair_j.long()[top_flat]
        s3_top       = self.model.score_s3_pairs(e_d, drug_embs, top_i, top_j, 4096)
        final_top    = w1*s1_all[top_flat] + w2*s2_all[top_flat] + w3*s3_top

        _, best = torch.topk(final_top, k=min(top_k, k2))
        return [
            (drug_ids[top_i[b].item()], drug_ids[top_j[b].item()],
             float(final_top[b].item()))
            for b in best.tolist()
        ]