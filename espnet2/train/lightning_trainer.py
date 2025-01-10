import argparse
from argparse import Namespace
import copy
import importlib
from pathlib import Path

import lightning as L
import torch
from lightning.pytorch.callbacks import (
    LearningRateMonitor,
    ModelCheckpoint,
    TQDMProgressBar,
)
from lightning.pytorch.loggers import TensorBoardLogger, WandbLogger
from lightning.pytorch.strategies import DDPStrategy, FSDPStrategy
from torch.distributed.algorithms.ddp_comm_hooks import default_hooks

import espnetez as ez
from src.lightning_trainer.callbacks import get_default_callbacks
from src.lightning_trainer.model import LitESPnetModel


class LightningTrainer:
    def __init__(
        self,
        args,
        callbacks,
    ):
        self.args = args
        self.callbacks = callbacks

        # Set random seed
        L.seed_everything(args.seed)

        # Set additional configurations that might be helpful
        torch.set_float32_matmul_precision("high")

        # Instantiate the strategy
        trainer_conf = copy.deepcopy(args.lightning_conf)
        strategy = trainer_conf.pop("strategy", "ddp")
        strategy_conf = trainer_conf.pop("strategy_conf", dict())

        if strategy == "ddp":
            ddp_comm_hook = strategy_conf.pop("ddp_comm_hook", None)
            if ddp_comm_hook is not None:
                ddp_comm_hook = getattr(default_hooks, ddp_comm_hook)

            strategy = DDPStrategy(
                ddp_comm_hook=ddp_comm_hook,
                **strategy_conf,
            )

        elif strategy == "fsdp":
            auto_wrap_policy = strategy_conf.pop("auto_wrap_policy", None)
            if auto_wrap_policy is not None and len(auto_wrap_policy) > 0:
                auto_wrap_policy = set(
                    getattr(
                        importlib.import_module(".".join(policy.split(".")[:-1])),
                        policy.split(".")[-1],
                    )
                    for policy in auto_wrap_policy
                )
            else:
                auto_wrap_policy = None

            activation_checkpointing_policy = strategy_conf.pop(
                "activation_checkpointing_policy", None
            )
            if (
                activation_checkpointing_policy is not None
                and len(activation_checkpointing_policy) > 0
            ):
                activation_checkpointing_policy = set(
                    getattr(
                        importlib.import_module(".".join(policy.split(".")[:-1])),
                        policy.split(".")[-1],
                    )
                    for policy in activation_checkpointing_policy
                )
            else:
                activation_checkpointing_policy = None

            strategy = FSDPStrategy(
                auto_wrap_policy=auto_wrap_policy,
                activation_checkpointing_policy=activation_checkpointing_policy,
                **strategy_conf,
            )

        else:
            raise ValueError(f"Unknown strategy: {strategy}")

        # Create loggers
        loggers = []

        if args.use_tensorboard:
            tb_logger = TensorBoardLogger(
                save_dir=args.output_dir,
                name="lightning_logs",
            )
            loggers.append(tb_logger)

        if args.use_wandb:
            wandb_latest_id = None
            # Resume the latest run if exists by setting the version
            if (Path(args.output_dir) / "wandb" / "latest-run").exists():
                wandb_latest_id = (
                    (Path(args.output_dir) / "wandb" / "latest-run")
                    .resolve()
                    .name.split("-")[-1]
                )
            wandb_logger = WandbLogger(
                project=args.wandb_project or "ESPnet_" + task_class.__name__,
                name=args.wandb_name or str(Path(".").resolve()).replace("/", "_"),
                save_dir=args.output_dir,
                version=wandb_latest_id,
            )
            loggers.append(wandb_logger)

        # Instantiate the Lightning Trainer
        self.trainer = L.Trainer(
            # Reload dataloaders every epoch to reuse ESPnet's dataloader
            reload_dataloaders_every_n_epochs=1,
            # ESPnet's dataloader already shards the dataset based on distributed setups
            use_distributed_sampler=False,
            **trainer_conf,
            callbacks=self.callbacks,
            strategy=strategy,
            logger=loggers,
        )

    def fit(self, *args, **kwargs):
        self.trainer.fit(*args, **kwargs)
