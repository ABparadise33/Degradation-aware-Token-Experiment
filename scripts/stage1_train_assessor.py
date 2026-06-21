import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from stage1.engine import train_assessor


def main():
    parser = argparse.ArgumentParser(description="Train Stage 1 degradation-aware SFIQA-style assessor.")
    parser.add_argument("--labels-csv", required=True)
    parser.add_argument(
        "--dataset-root",
        default=None,
        help="UIEB root containing raw-890/reference-890 or raw/GT.",
    )
    parser.add_argument("--architecture", choices=["token_mlp", "task_attention"], default="token_mlp")
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--decoder-layers", type=int, default=1)
    parser.add_argument("--lambda-contrast", type=float, default=0.0)
    parser.add_argument("--lambda-order", type=float, default=0.0)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--backbone", default="resnet50", help="Any timm model name, e.g. resnet50 or convnext_tiny.")
    parser.add_argument("--no-pretrained", action="store_true")
    parser.add_argument("--freeze-backbone", action="store_true")
    parser.add_argument(
        "--legacy-direct-score",
        action="store_true",
        help="Use the V1 baseline where scores bypass z_deg and are predicted directly from backbone features.",
    )
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--latent-dim", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--rank-margin", type=float, default=0.1)
    parser.add_argument("--lambda-rank", type=float, default=1.0)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    if args.legacy_direct_score and args.architecture != "token_mlp":
        parser.error("--legacy-direct-score is only valid with --architecture token_mlp.")
    if args.lambda_contrast < 0 or args.lambda_order < 0:
        parser.error("Synthetic loss weights must be non-negative.")

    best_path = train_assessor(
        labels_csv=args.labels_csv,
        output_dir=args.output_dir,
        backbone=args.backbone,
        pretrained=not args.no_pretrained,
        freeze_backbone=args.freeze_backbone,
        image_size=args.image_size,
        latent_dim=args.latent_dim,
        dropout=args.dropout,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        rank_margin=args.rank_margin,
        lambda_rank=args.lambda_rank,
        num_workers=args.num_workers,
        device_name=args.device,
        dataset_root=args.dataset_root,
        score_from_token=not args.legacy_direct_score,
        architecture=args.architecture,
        num_heads=args.num_heads,
        decoder_layers=args.decoder_layers,
        lambda_contrast=args.lambda_contrast,
        lambda_order=args.lambda_order,
    )
    print(f"Best checkpoint: {best_path}")


if __name__ == "__main__":
    main()
