import argparse
import json
from pathlib import Path
from typing import Optional, Dict, Any

import torch

from RL.reinforcement_learning import RelationshipReinforcementLearning


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fine-tune RelTR with global context vectors extracted from a directory of images."
    )
    parser.add_argument(
        "--image_dir",
        required=True,
        help="Directory containing input images for fine-tuning.",
    )
    parser.add_argument(
        "--reltr_checkpoint",
        required=False,
        default=None,
        help="Path to the initial RelTR checkpoint (default: none).",
    )
    parser.add_argument(
        "--yolo_weights",
        required=False,
        default=None,
        help="Optional path to YOLO weights used for object detection.",
    )
    parser.add_argument(
        "--experiment_dir",
        required=False,
        default=None,
        help="Directory where experiment artifacts (snapshots, logs) will be written.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=1,
        help="Number of fine-tuning epochs (calls to train_relationship_model).",
    )
    parser.add_argument(
        "--output",
        required=False,
        default=None,
        help="Path to save the fine-tuned RelTR checkpoint. If omitted, the checkpoint is not saved.",
    )
    parser.add_argument(
        "--keep_dataset",
        action="store_true",
        help="Keep previously prepared dataset samples instead of rebuilding from scratch.",
    )
    parser.add_argument(
        "--report",
        required=False,
        default=None,
        help="Optional path to save a JSON report with training losses.",
    )
    parser.add_argument(
        "--relationships_dir",
        required=False,
        default=None,
        help="If provided, export predicted relationships for each image as JSON before training.",
    )
    return parser.parse_args()


def build_agent(args: argparse.Namespace) -> RelationshipReinforcementLearning:
    data_paths: Dict[str, Optional[str]] = {}
    if args.reltr_checkpoint:
        data_paths["reltr_checkpoint"] = args.reltr_checkpoint
    if args.yolo_weights:
        data_paths["yolo_weights"] = args.yolo_weights

    agent = RelationshipReinforcementLearning(
        detection_model=None,
        relationship_model=None,
        generator=None,
        experiment_dir=args.experiment_dir,
        data_paths=data_paths or None,
    )
    return agent


def ensure_relationship_annotations(
    agent: RelationshipReinforcementLearning,
    export_dir: Optional[Path] = None,
) -> int:
    if export_dir:
        export_dir.mkdir(parents=True, exist_ok=True)

    valid_samples = 0
    for sample in agent.dataset_samples:
        relationships = sample.get("relationships") or []
        if not relationships:
            image_tensor = agent._load_image_tensor(sample["image_path"])
            relationships = agent._run_reltr_inference(
                image_tensor,
                sample.get("objects", []),
                sample.get("global_context"),
            )
            sample["relationships"] = relationships

        if relationships:
            valid_samples += 1

        if export_dir is not None:
            image_path = Path(sample["image_path"])
            export_path = export_dir / f"{image_path.stem}_relationships.json"
            with export_path.open("w", encoding="utf-8") as f:
                json.dump(relationships, f, ensure_ascii=False, indent=2)

    return valid_samples


def main() -> None:
    args = parse_args()

    image_dir = Path(args.image_dir)
    if not image_dir.exists() or not image_dir.is_dir():
        raise FileNotFoundError(f"Image directory not found: {image_dir}")

    agent = build_agent(args)

    print(f"[FineTune] Building dataset from {image_dir}")
    dataset_size = agent.build_dataset_from_directory(str(image_dir), clear_previous=not args.keep_dataset)
    if dataset_size == 0:
        print("[FineTune] No samples prepared; aborting.")
        return
    print(f"[FineTune] Prepared {dataset_size} samples.")

    relationships_dir = Path(args.relationships_dir) if args.relationships_dir else None
    valid_relationship_samples = ensure_relationship_annotations(agent, relationships_dir)
    print(f"[FineTune] Relationship annotations available for {valid_relationship_samples}/{dataset_size} samples.")
    if valid_relationship_samples == 0:
        print("[FineTune] Warning: no relationship annotations available; RelTR will not update meaningfully.")

    losses = []
    for epoch in range(1, args.epochs + 1):
        loss = agent.train_relationship_model(synthetic_data=None)
        losses.append(float(loss))
        print(f"[FineTune] Epoch {epoch}/{args.epochs} - relationship loss: {loss:.6f}")

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {"model": agent.relationship_model.state_dict()},
            output_path,
        )
        print(f"[FineTune] Saved fine-tuned checkpoint to {output_path}")

    if args.report:
        report_path = Path(args.report)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report: Dict[str, Any] = {
            "image_dir": str(image_dir.resolve()),
            "epochs": args.epochs,
            "losses": losses,
            "output_checkpoint": str(Path(args.output).resolve()) if args.output else None,
            "valid_relationship_samples": valid_relationship_samples,
        }
        with report_path.open("w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
        print(f"[FineTune] Wrote training report to {report_path}")


if __name__ == "__main__":
    main()
