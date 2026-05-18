"""
Drug Combination Ranking Model  —  v2.3 (CompGCN + Relation-Aware Attention)

Key design:
  - Drug, Disease, Gene node features: purely learnable nn.Embedding
  - node_type_mask: 0 = drug, 1 = disease, 2 = gene
  - Three scorers: S1 (drug-drug synergy), S2 (drug-disease relevance), S3 (three-way)
  - Final score = softmax_weighted sum of S1, S2, S3
  - trainer.py / data.py / main.py: zero interface changes

Changes vs v1 (R-GCN):
  [enc-1] CompGCNConv replaces RGCNConv (Vashishth et al., ICLR 2020).
  [enc-2] Relation-Aware Attention Gate (RAAG) after each CompGCN layer.
  [enc-3] Basis decomposition for relation embeddings.

Bug fixes:
  [fix-A] HeterogeneousKGEncoder.forward: node feature init via torch.cat
          instead of masked inplace assignment (x[mask]=emb) which broke
          autograd version counters under retain_graph.
  [fix-B] scatter_sum: uses scatter_add (functional) instead of index_add_.
  [fix-C] CompGCNConv: fwd/inv split via float-mask multiply instead of
          boolean slice to avoid storage-aliasing under retain_graph.
  [fix-OOM] RAAG.forward: edge-chunked processing so peak memory per RAAG
            layer is O(chunk_size * d) instead of O(E * d).
            E=840,638 edges (bidirectional), d=512 → full materialization
            was 840638 * 512 * 4 bytes ≈ 1.7 GB per tensor, several tensors
            at once → OOM on 80 GB GPU when 3 encode_all() calls coexist.
            With raag_chunk=50000 peak is ~100 MB per chunk.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ─────────────────────────────────────────────────────────────────────────────
# Autograd-safe scatter
# ─────────────────────────────────────────────────────────────────────────────

def scatter_sum(src: torch.Tensor,
                index: torch.Tensor,
                dim_size: int) -> torch.Tensor:
    """Functional sum-scatter — returns new tensor, never mutates existing one."""
    C      = src.size(1)
    idx_2d = index.unsqueeze(1).expand(-1, C)
    base   = src.new_zeros(dim_size, C)
    return base.scatter_add(0, idx_2d, src)


# ─────────────────────────────────────────────────────────────────────────────
# CompGCN single-layer conv
# ─────────────────────────────────────────────────────────────────────────────

class CompGCNConv(nn.Module):
    """
    One CompGCN message-passing layer.

    Operators: 'sub' | 'mult' | 'corr' (default: circular correlation via FFT).
    Three channels: forward (W_O), inverse (W_I), self-loop (W_S).
    Optional basis decomposition (num_bases > 0).
    """

    def __init__(
        self,
        d_in:      int,
        d_out:     int,
        num_rels:  int,
        num_bases: int   = 4,
        opn:       str   = 'corr',
        dropout:   float = 0.1,
    ):
        super().__init__()
        self.d_in      = d_in
        self.d_out     = d_out
        self.num_rels  = num_rels
        self.num_bases = num_bases
        self.opn       = opn
        self.drop      = nn.Dropout(dropout)

        self.W_O = nn.Linear(d_in, d_out, bias=False)
        self.W_I = nn.Linear(d_in, d_out, bias=False)
        self.W_S = nn.Linear(d_in, d_out, bias=False)

        if num_bases > 0:
            self.rel_bases  = nn.Parameter(torch.empty(num_bases, d_in))
            self.rel_coeffs = nn.Parameter(torch.empty(num_rels,  num_bases))
            nn.init.xavier_uniform_(self.rel_bases)
            nn.init.xavier_uniform_(self.rel_coeffs)
        else:
            self.rel_emb = nn.Parameter(torch.empty(num_rels, d_in))
            nn.init.xavier_uniform_(self.rel_emb)

        self.W_rel = nn.Linear(d_in, d_out, bias=False)
        self.bias  = nn.Parameter(torch.zeros(d_out))

    def _rel_embeddings(self) -> torch.Tensor:
        if self.num_bases > 0:
            return self.rel_coeffs @ self.rel_bases
        return self.rel_emb

    @staticmethod
    def _ccorr(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        fa = torch.fft.rfft(a, dim=-1)
        fb = torch.fft.rfft(b, dim=-1)
        return torch.fft.irfft(fa.conj() * fb, n=a.size(-1), dim=-1)

    def _compose(self, h: torch.Tensor, hr: torch.Tensor) -> torch.Tensor:
        if self.opn == 'sub':  return h - hr
        if self.opn == 'mult': return h * hr
        if self.opn == 'corr': return self._ccorr(h, hr)
        raise ValueError(f'Unknown opn: {self.opn}')

    def forward(self, x, edge_index, edge_type):
        N        = x.size(0)
        src, dst = edge_index[0], edge_index[1]

        rel_embs = self._rel_embeddings()
        composed = self._compose(x[src], rel_embs[edge_type])
        composed = self.drop(composed)

        # [fix-C] float-mask to avoid boolean-slice storage aliasing
        half_R  = self.num_rels // 2
        is_fwd  = (edge_type < half_R).float().unsqueeze(1)
        is_inv  = 1.0 - is_fwd

        agg_fwd = self.W_O(scatter_sum(composed * is_fwd, dst, N))
        agg_inv = self.W_I(scatter_sum(composed * is_inv, dst, N))
        x_out   = agg_fwd + agg_inv + self.W_S(x) + self.bias
        rel_out = self.W_rel(rel_embs)
        return x_out, rel_out


# ─────────────────────────────────────────────────────────────────────────────
# Relation-Aware Attention Gate  (edge-chunked to avoid OOM)
# ─────────────────────────────────────────────────────────────────────────────

class RelationAwareAttentionGate(nn.Module):
    """
    Per-edge soft attention gate conditioned on (node_dst, relation, node_src).

    [fix-OOM] forward() processes edges in chunks of `chunk_size` so peak
    memory is O(chunk_size * d) instead of O(E * d).

    With E=840,638 and d=512:
      Full materialisation: 840638 × 512 × 4B × (several tensors) ≈ 5-8 GB
      Chunked (50k edges):  50000  × 512 × 4B × (several tensors) ≈ 0.4 GB

    chunk_size is set at construction time via the `raag_chunk` config key
    (default 50,000 edges per chunk).
    """

    def __init__(self, d: int, num_rels: int,
                 dropout: float = 0.1, chunk_size: int = 50_000):
        super().__init__()
        self.chunk_size = chunk_size
        self.W_a     = nn.Linear(d * 2, 1, bias=True)
        self.W_v     = nn.Linear(d, d,     bias=False)
        self.W_r_att = nn.Linear(d, d,     bias=False)
        self.drop    = nn.Dropout(dropout)
        self.norm    = nn.LayerNorm(d)

    def forward(self, x, rel_embs, edge_index, edge_type):
        N        = x.size(0)
        d        = x.size(1)
        src, dst = edge_index[0], edge_index[1]
        E        = src.size(0)

        # Accumulate weighted messages in chunks to cap peak VRAM
        # Each chunk: O(chunk_size * d) — safe even for large KGs
        acc = x.new_zeros(N, d)   # will hold Σ α·W_v(x[src])

        for start in range(0, E, self.chunk_size):
            end = min(start + self.chunk_size, E)

            s_c   = src[start:end]             # [C]
            d_c   = dst[start:end]             # [C]
            et_c  = edge_type[start:end]        # [C]

            h_r       = self.W_r_att(rel_embs[et_c])    # [C, d]
            h_src_r   = x[s_c] * h_r                    # [C, d]
            gate_in   = torch.cat([x[d_c], h_src_r], dim=-1)  # [C, 2d]
            alpha     = torch.sigmoid(
                self.W_a(F.leaky_relu(gate_in, 0.2))
            )                                             # [C, 1]
            val       = self.W_v(x[s_c]) * alpha          # [C, d]
            val       = self.drop(val)

            # scatter_add into acc
            idx_2d = d_c.unsqueeze(1).expand(-1, d)
            acc    = acc.scatter_add(0, idx_2d, val)

        return self.norm(x + acc)


# ─────────────────────────────────────────────────────────────────────────────
# KG encoder
# ─────────────────────────────────────────────────────────────────────────────

class HeterogeneousKGEncoder(nn.Module):
    """
    Multi-layer CompGCN + RAAG encoder.
    Per-layer: CompGCNConv → LayerNorm(GELU) → RAAG → Dropout
    Output: x + residual_proj(x_init)

    [fix-A] Initial x built via torch.cat (no inplace masked assignment).
    """

    def __init__(
        self,
        d_hidden:      int,
        d_out:         int,
        num_relations: int,
        num_drugs:     int,
        num_diseases:  int,
        num_genes:     int,
        num_layers:    int   = 3,
        num_bases:     int   = 4,
        opn:           str   = 'corr',
        dropout:       float = 0.1,
        raag_chunk:    int   = 50_000,
    ):
        super().__init__()

        self.drug_embedding    = nn.Embedding(num_drugs,    d_hidden)
        self.disease_embedding = nn.Embedding(num_diseases, d_hidden)
        self.gene_embedding    = nn.Embedding(num_genes,    d_hidden)
        for emb in (self.drug_embedding, self.disease_embedding, self.gene_embedding):
            nn.init.xavier_uniform_(emb.weight)

        dims = [d_hidden] * num_layers + [d_out]

        self.convs = nn.ModuleList([
            CompGCNConv(dims[i], dims[i+1], num_relations, num_bases, opn, dropout)
            for i in range(num_layers)
        ])
        self.norms   = nn.ModuleList([nn.LayerNorm(dims[i+1]) for i in range(num_layers)])
        self.raags   = nn.ModuleList([
            RelationAwareAttentionGate(dims[i+1], num_relations, dropout, raag_chunk)
            for i in range(num_layers)
        ])
        self.dropout = nn.Dropout(dropout)
        self.residual_proj = (
            nn.Linear(d_hidden, d_out) if d_hidden != d_out else nn.Identity()
        )

    def forward(self, drug_indices, disease_indices, gene_indices,
                node_type_mask, edge_index, edge_type):
        # [fix-A] Pure functional init — no inplace masked assignment
        drug_e    = self.drug_embedding(drug_indices)
        disease_e = self.disease_embedding(disease_indices)
        gene_e    = self.gene_embedding(gene_indices)
        x         = torch.cat([drug_e, disease_e, gene_e], dim=0)

        residual = self.residual_proj(x)

        for conv, norm, raag in zip(self.convs, self.norms, self.raags):
            x, rel_out = conv(x, edge_index, edge_type)
            x = norm(x)
            x = F.gelu(x)
            x = raag(x, rel_out, edge_index, edge_type)
            x = self.dropout(x)

        return x + residual


# Backward-compat alias
HeterogeneousKGAttention = HeterogeneousKGEncoder


# ─────────────────────────────────────────────────────────────────────────────
# Scorers  (unchanged from v1)
# ─────────────────────────────────────────────────────────────────────────────

class DrugSynergyScorer(nn.Module):
    def __init__(self, d_emb: int, dropout: float = 0.1):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(d_emb * 3, d_emb), nn.LayerNorm(d_emb), nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_emb, d_emb // 2), nn.GELU(),
            nn.Linear(d_emb // 2, 1),
        )

    def forward(self, e_i, e_j):
        return self.mlp(torch.cat([e_i+e_j, e_i*e_j, (e_i-e_j).abs()], -1)).squeeze(-1)


class DrugDiseaseRelevanceScorer(nn.Module):
    def __init__(self, d_emb: int, dropout: float = 0.1):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(d_emb * 3, d_emb), nn.LayerNorm(d_emb), nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_emb, d_emb // 2), nn.GELU(),
            nn.Linear(d_emb // 2, 1),
        )

    def forward(self, e_disease, e_drug):
        return self.mlp(torch.cat([e_disease, e_drug, e_disease*e_drug], -1)).squeeze(-1)


class ThreeWayInteractionScorer(nn.Module):
    """S3: disease-specific combination scorer. drug_i/j share type_emb[1] → symmetric."""

    def __init__(self, d_emb: int, n_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.type_emb = nn.Embedding(2, d_emb)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_emb, nhead=n_heads, dim_feedforward=d_emb*2,
            dropout=dropout, batch_first=True, norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(enc_layer, num_layers=2)
        self.head = nn.Sequential(
            nn.Linear(d_emb*3, d_emb), nn.GELU(), nn.Linear(d_emb, 1),
        )

    def forward(self, e_disease, e_drug_i, e_drug_j):
        type_ids  = torch.tensor([0, 1, 1], device=e_disease.device)
        type_embs = self.type_emb(type_ids)
        tokens    = torch.stack([e_disease, e_drug_i, e_drug_j], 1) + type_embs
        out       = self.transformer(tokens)
        flat      = torch.cat([out[:,0], out[:,1:].mean(1), out[:,0]*out[:,1:].mean(1)], -1)
        return self.head(flat).squeeze(-1)


# ─────────────────────────────────────────────────────────────────────────────
# Full model
# ─────────────────────────────────────────────────────────────────────────────

class DrugCombinationModel(nn.Module):
    """
    End-to-end drug combination ranking model.

    New config keys vs v1 (all optional):
        num_bases  (int, default 4)       : basis vectors for relation decomp
        opn        (str, default 'corr')  : CompGCN composition operator
        raag_chunk (int, default 50000)   : edges per RAAG chunk (memory control)
    """

    def __init__(self, config: dict):
        super().__init__()
        self.config = config

        self.kg_attention = HeterogeneousKGEncoder(
            d_hidden      = config['d_hidden'],
            d_out         = config['d_emb'],
            num_relations = config['n_relations'],
            num_drugs     = config['n_drugs'],
            num_diseases  = config['n_diseases'],
            num_genes     = config['n_genes'],
            num_layers    = config.get('n_layers',   3),
            num_bases     = config.get('num_bases',  4),
            opn           = config.get('opn',       'corr'),
            dropout       = config.get('dropout',    0.1),
            raag_chunk    = config.get('raag_chunk', 50_000),
        )

        d_emb   = config['d_emb']
        dropout = config.get('dropout', 0.1)
        n_heads = config.get('n_heads', 8)

        self.s1_synergy    = DrugSynergyScorer(d_emb, dropout)
        self.s2_relevance  = DrugDiseaseRelevanceScorer(d_emb, dropout)
        self.s3_threeway   = ThreeWayInteractionScorer(d_emb, n_heads, dropout)
        self.score_weights = nn.Parameter(torch.ones(3))

    def encode_all(self, drug_indices, disease_indices, gene_indices,
                   node_type_mask, edge_index, edge_type):
        n_drug    = drug_indices.size(0)
        n_disease = disease_indices.size(0)
        x_all     = self.kg_attention(
            drug_indices, disease_indices, gene_indices,
            node_type_mask, edge_index, edge_type,
        )
        return x_all[:n_drug], x_all[n_drug: n_drug + n_disease]

    def score_triplets(self, e_disease, e_drug_i, e_drug_j):
        s1    = self.s1_synergy(e_drug_i, e_drug_j)
        s2    = self.s2_relevance(e_disease, e_drug_i) + \
                self.s2_relevance(e_disease, e_drug_j)
        s3    = self.s3_threeway(e_disease, e_drug_i, e_drug_j)
        w     = F.softmax(self.score_weights, dim=0)
        final = w[0]*s1 + w[1]*s2 + w[2]*s3
        return {'s1': s1, 's2': s2, 's3': s3, 'final': final}

    def forward(self, drug_indices, disease_indices, gene_indices,
                node_type_mask, edge_index, edge_type,
                disease_idx, drug_i_idx, drug_j_idx):
        de, dre = self.encode_all(drug_indices, disease_indices, gene_indices,
                                  node_type_mask, edge_index, edge_type)
        return self.score_triplets(de[disease_idx], dre[drug_i_idx], dre[drug_j_idx])

    # ── Fast batch scoring APIs (unchanged from v1) ────────────────────────

    @torch.no_grad()
    def precompute_s2_scores(self, e_disease, drug_embs, batch_size=4096):
        scores, e_d = [], e_disease.unsqueeze(0)
        for s in range(0, drug_embs.size(0), batch_size):
            e = min(s+batch_size, drug_embs.size(0))
            scores.append(self.s2_relevance(e_d.expand(e-s,-1), drug_embs[s:e]))
        return torch.cat(scores)

    @torch.no_grad()
    def precompute_s1_scores_chunked(self, drug_embs, pair_i, pair_j, batch_size=50000):
        scores, pi, pj = [], pair_i.long(), pair_j.long()
        for s in range(0, pair_i.size(0), batch_size):
            e = min(s+batch_size, pair_i.size(0))
            scores.append(self.s1_synergy(drug_embs[pi[s:e]], drug_embs[pj[s:e]]))
        return torch.cat(scores)

    @torch.no_grad()
    def score_s3_pairs(self, e_disease, drug_embs, pair_i_sub, pair_j_sub, batch_size=4096):
        scores, e_d = [], e_disease.unsqueeze(0)
        pi, pj = pair_i_sub.long(), pair_j_sub.long()
        for s in range(0, pair_i_sub.size(0), batch_size):
            e = min(s+batch_size, pair_i_sub.size(0))
            scores.append(self.s3_threeway(
                e_d.expand(e-s,-1), drug_embs[pi[s:e]], drug_embs[pj[s:e]]))
        return torch.cat(scores)