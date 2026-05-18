"""
Main entry point for Drug Combination Ranking
Usage:
  python main.py --data_dir ./data --drug_ids ./data/drug_ids.csv \
                 --disease_ids ./data/disease_ids.csv
"""

import argparse
import json
import logging
import os
from pathlib import Path

import torch

from data    import load_kg, KGGraph, build_dataloaders
from model   import DrugCombinationModel
from trainer import Trainer


def setup_logging(output_dir: str):
    os.makedirs(output_dir, exist_ok=True)
    log_path = os.path.join(output_dir, 'train.log')
    fmt  = logging.Formatter(
        fmt='[%(asctime)s] %(levelname)-8s %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
    )
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    # Guard against duplicate handlers (e.g. if main() is called more than once
    # in tests, or if the logging module was already initialised by a dependency).
    if not root.handlers:
        for h in [logging.StreamHandler(), logging.FileHandler(log_path, mode='a')]:
            h.setFormatter(fmt)
            root.addHandler(h)
    logging.info(f"Logging to {log_path}")


def parse_args():
    parser = argparse.ArgumentParser(description='Drug Combination Ranking')

    # Data
    parser.add_argument('--data_dir',    type=str, default='./data')
    parser.add_argument('--drug_ids',    type=str, required=True,
                        help='File with one drug ID per line (no header)')
    parser.add_argument('--disease_ids', type=str, required=True,
                        help='File with one disease ID per line (no header)')
    parser.add_argument('--output_dir',  type=str, default='./outputs')

    # Model
    parser.add_argument('--d_hidden',  type=int,   default=512)
    parser.add_argument('--d_emb',     type=int,   default=512)
    parser.add_argument('--n_heads',   type=int,   default=8)
    parser.add_argument('--n_layers',  type=int,   default=3)
    parser.add_argument('--dropout',   type=float, default=0.1)

    # Training
    parser.add_argument('--phase1_epochs', type=int,   default=20)
    parser.add_argument('--phase2_epochs', type=int,   default=20)
    parser.add_argument('--phase3_epochs', type=int,   default=100)
    parser.add_argument('--lr',            type=float, default=1e-3)
    parser.add_argument('--phase3_lr',     type=float, default=1e-4)
    parser.add_argument('--weight_decay',  type=float, default=1e-4)
    parser.add_argument('--batch_size',    type=int,   default=512)
    parser.add_argument('--neg_ratio',     type=int,   default=5)
    parser.add_argument('--aux_lambda',    type=float, default=0.1)
    parser.add_argument('--patience',      type=int,   default=10)

    # Inference
    parser.add_argument('--stage1_k', type=int, default=500,
                        help='Stage-1 candidate pool size for inference')
    parser.add_argument('--top_k',    type=int, default=100,
                        help='Final top-K drug pairs to return')

    # Misc
    parser.add_argument('--seed',        type=int, default=42)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--no_phase1',   action='store_true')
    parser.add_argument('--no_phase2',   action='store_true')
    parser.add_argument('--eval_only',   action='store_true')
    parser.add_argument('--checkpoint',  type=str, default=None)

    return parser.parse_args()


def set_seed(seed: int):
    import random
    import numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main():
    args = parse_args()
    setup_logging(args.output_dir)
    set_seed(args.seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f"Using device: {device}")

    os.makedirs(args.output_dir, exist_ok=True)
    data_dir = Path(args.data_dir)

    # ── Entity ID lists ───────────────────────────────────────────────────────
    logging.info("Loading entity lists …")
    with open(args.drug_ids)    as f: drug_ids    = [l.strip() for l in f if l.strip()]
    with open(args.disease_ids) as f: disease_ids = [l.strip() for l in f if l.strip()]

    # [main-2] Deduplicate while preserving order.  Duplicate entries would
    # cause the same ID to receive two different local indices in KGGraph and
    # build_dataloaders, leading to silent embedding lookup errors.
    drug_ids    = list(dict.fromkeys(drug_ids))
    disease_ids = list(dict.fromkeys(disease_ids))

    # ── KG ────────────────────────────────────────────────────────────────────
    logging.info("Loading KG …")
    kg_df = load_kg(data_dir / 'KG.csv')
    kg    = KGGraph(kg_df, drug_ids, disease_ids)
    # [main-1] KGGraph now emits the correct 3-value node_type_mask directly;
    #          no patch needed in Trainer.__init__().

    logging.info(f"  Drugs:     {len(drug_ids)}")
    logging.info(f"  Diseases:  {len(disease_ids)}")
    logging.info(f"  Genes:     {kg.n_gene}")
    logging.info(f"  Relations: {kg.n_relations_total}")

    # ── DataLoaders ───────────────────────────────────────────────────────────
    logging.info("Building dataloaders …")
    train_loader, val_loader, test_loader = build_dataloaders(
        data_dir    = data_dir,
        drug_ids    = drug_ids,
        disease_ids = disease_ids,
        batch_size  = args.batch_size,
        neg_ratio   = args.neg_ratio,
        num_workers = args.num_workers,
    )

    # ── Model ─────────────────────────────────────────────────────────────────
    # Consolidate all hyper-parameters in one dict so Trainer and Model share
    # a single source of truth (no duplicate / divergent keys).
    config = {
        'n_drugs':     len(drug_ids),
        'n_diseases':  len(disease_ids),
        'n_genes':     kg.n_gene,
        'n_relations': kg.n_relations_total,

        'd_hidden':    args.d_hidden,
        'd_emb':       args.d_emb,
        'n_heads':     args.n_heads,
        'n_layers':    args.n_layers,
        'dropout':     args.dropout,

        'lr':           args.lr,
        'phase3_lr':    args.phase3_lr,
        'weight_decay': args.weight_decay,
        'aux_lambda':   args.aux_lambda,
        'patience':     args.patience,

        # Phase 1 / 2 training knobs (shared batch_size / neg_k)
        'phase1_batch_size': args.batch_size,
        'phase2_batch_size': args.batch_size,
        'phase1_neg_k':      10,
        'phase2_neg_k':      10,
    }

    logging.info("Building model …")
    model    = DrugCombinationModel(config)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logging.info(f"  Parameters: {n_params:,}")

    if args.checkpoint:
        logging.info(f"  Loading checkpoint: {args.checkpoint}")
        model.load_state_dict(torch.load(args.checkpoint, map_location='cpu', weights_only=True))

    # ── Trainer ───────────────────────────────────────────────────────────────
    trainer = Trainer(model, kg, config, device)

    # [main-3] Fail fast: evaluating a randomly-initialised model produces
    # meaningless metrics with no error message.
    if args.eval_only and not args.checkpoint:
        raise ValueError(
            "--eval_only requires --checkpoint. "
            "Without a checkpoint the model is randomly initialised and "
            "evaluation results are meaningless."
        )

    # ── Training ──────────────────────────────────────────────────────────────
    if not args.eval_only:
        ckpt_path = os.path.join(args.output_dir, 'best_model.pt')

        if not args.no_phase1:
            trainer.train_phase1(n_epochs=args.phase1_epochs)

        if not args.no_phase2:
            trainer.train_phase2(train_loader, n_epochs=args.phase2_epochs)

        trainer.train_phase3(
            train_loader, val_loader,
            n_epochs=args.phase3_epochs,
            checkpoint_path=ckpt_path,
            stage2_k=args.stage1_k,        # [NEW-2] val eval must use the same
                                            # reranking window as inference
        )

    # ── Final evaluation ──────────────────────────────────────────────────────
    logging.info("\n=== Final Test Evaluation ===")
    test_metrics = trainer.evaluate_full_space(test_loader, stage2_k=args.stage1_k)
    for k, v in sorted(test_metrics.items()):
        logging.info(f"  {k}: {v:.4f}")

    metrics_path = os.path.join(args.output_dir, 'test_metrics.json')
    with open(metrics_path, 'w') as f:
        json.dump(test_metrics, f, indent=2)
    logging.info(f"Metrics saved to {metrics_path}")

    # ── Example inference ─────────────────────────────────────────────────────
    disease2idx    = {nid: i for i, nid in enumerate(disease_ids)}
    sample_disease = disease_ids[0]
    logging.info(f"\n=== Example Inference: disease={sample_disease} ===")
    results = trainer.rank_all_pairs_for_disease(
        disease_id  = sample_disease,
        disease2idx = disease2idx,
        drug_ids    = drug_ids,
        top_k       = args.top_k,
        stage2_k    = args.stage1_k,
    )
    logging.info("Top-5 drug combinations:")
    for drug1, drug2, score in results[:5]:
        logging.info(f"  ({drug1}, {drug2})  score={score:.4f}")

    results_path = os.path.join(args.output_dir, f'rankings_{sample_disease}.json')
    with open(results_path, 'w') as f:
        json.dump(
            [{'drug1': d1, 'drug2': d2, 'score': s} for d1, d2, s in results],
            f, indent=2,
        )
    logging.info(f"Rankings saved to {results_path}")


if __name__ == '__main__':
    main()