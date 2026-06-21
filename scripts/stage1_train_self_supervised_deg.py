import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from stage1.self_supervised_engine import train_self_supervised_encoder


def main():
    parser = argparse.ArgumentParser(
        description="Train anonymous-slot self-supervised degradation representation."
    )
    parser.add_argument("--labels-csv", required=True)
    parser.add_argument("--dataset-root", default=None)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--backbone", default="convnext_tiny")
    parser.add_argument("--no-pretrained", action="store_true")
    parser.add_argument("--freeze-backbone", action="store_true")
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--latent-dim", type=int, default=128)
    parser.add_argument("--num-slots", type=int, default=4)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--decoder-layers", type=int, default=1)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--separation-margin", type=float, default=0.5)
    parser.add_argument("--order-margin", type=float, default=0.1)
    parser.add_argument("--ranking-margin", type=float, default=0.1)
    parser.add_argument("--lambda-same", type=float, default=1.0)
    parser.add_argument("--lambda-different", type=float, default=0.5)
    parser.add_argument("--lambda-order", type=float, default=0.5)
    parser.add_argument("--lambda-pair", type=float, default=0.5)
    parser.add_argument("--lambda-variance", type=float, default=0.01)
    parser.add_argument("--lambda-covariance", type=float, default=0.01)
    parser.add_argument("--lambda-slot-diversity", type=float, default=0.01)
    args = parser.parse_args()

    weights = [
        args.lambda_same,
        args.lambda_different,
        args.lambda_order,
        args.lambda_pair,
        args.lambda_variance,
        args.lambda_covariance,
        args.lambda_slot_diversity,
    ]
    if min(weights) < 0:
        parser.error("Loss weights must be non-negative.")

    best_path = train_self_supervised_encoder(
        labels_csv=args.labels_csv,
        dataset_root=args.dataset_root,
        output_dir=args.output_dir,
        backbone=args.backbone,
        pretrained=not args.no_pretrained,
        freeze_backbone=args.freeze_backbone,
        image_size=args.image_size,
        latent_dim=args.latent_dim,
        num_slots=args.num_slots,
        num_heads=args.num_heads,
        decoder_layers=args.decoder_layers,
        dropout=args.dropout,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        num_workers=args.num_workers,
        device_name=args.device,
        temperature=args.temperature,
        separation_margin=args.separation_margin,
        order_margin=args.order_margin,
        ranking_margin=args.ranking_margin,
        lambda_same=args.lambda_same,
        lambda_different=args.lambda_different,
        lambda_order=args.lambda_order,
        lambda_pair=args.lambda_pair,
        lambda_variance=args.lambda_variance,
        lambda_covariance=args.lambda_covariance,
        lambda_slot_diversity=args.lambda_slot_diversity,
    )
    print(f"Best checkpoint: {best_path}")


if __name__ == "__main__":
    main()
