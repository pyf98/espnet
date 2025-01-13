import argparse
import copy
import importlib
import logging
from pathlib import Path

from espnet2.train.lightning_espnet_model import task_choices
from espnet2.train.lightning_trainer import LightningTrainer


def get_base_parser():
    """Create the base parser with task selection."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--task",
        type=str,
        required=True,
        choices=list(task_choices.keys()),
        help="Task to execute.",
    )
    return parser


def get_parser(task_class):
    parser = task_class.get_parser()
    parser.add_argument(
        "--lightning_conf",
        action=NestedDictAction,
        default=dict(),
        help="Arguments related to Lightning Trainer.",
    )
    return parser


def main():
    # First parse the task and then parse task-specific arguments
    base_parser = get_base_parser()
    task_args, remaining_args = base_parser.parse_known_args()
    task_class = task_choices[task_args.task]
    parser = get_parser(task_class)
    args = parser.parse_args(remaining_args)
    args.task = task_args.task

    # Instantiate the Lightning Model
    lit_model = LitESPnetModel(args=args)

    # Create callbacks
    espnet_callbacks = get_default_callbacks(args)

    # Define the trainer
    trainer = LightningTrainer(args, espnet_callbacks, lit_model)

    # Start training with automatic resuming from the last checkpoint
    trainer.fit(model=lit_model, ckpt_path="last")


if __name__ == "__main__":
    main()
