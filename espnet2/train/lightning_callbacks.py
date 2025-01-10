from pathlib import Path

import torch
from lightning.pytorch.callbacks import (
    Callback,
    LearningRateMonitor,
    ModelCheckpoint,
    TQDMProgressBar,
)


class AverageCheckpointsCallback(Callback):
    def __init__(self, output_dir, best_ckpt_callbacks):
        self.output_dir = output_dir
        self.best_ckpt_callbacks = best_ckpt_callbacks

    def on_fit_end(self, trainer, pl_module):
        if trainer.is_global_zero:
            for ckpt_callback in self.best_ckpt_callbacks:
                checkpoints = list(ckpt_callback.best_k_models.keys())

                avg_state_dict = None
                for ckpt_path in checkpoints:
                    state_dict = torch.load(ckpt_path, map_location="cpu")["state_dict"]

                    if avg_state_dict is None:
                        avg_state_dict = state_dict
                    else:
                        for k in avg_state_dict:
                            avg_state_dict[k] = avg_state_dict[k] + state_dict[k]

                for k in avg_state_dict:
                    if str(avg_state_dict[k].dtype).startswith("torch.int"):
                        # For int type, not averaged, but only accumulated.
                        # e.g. BatchNorm.num_batches_tracked
                        # (If there are any cases that requires averaging
                        #  or the other reducing method, e.g. max/min, for integer type,
                        #  please report.)
                        pass
                    else:
                        avg_state_dict[k] = avg_state_dict[k] / len(checkpoints)

                # remove extra prefix in model keys
                new_avg_state_dict = {
                    k.removeprefix("model."): v
                    for k, v in avg_state_dict.items()
                    if k.startswith("model.")
                }

                avg_ckpt_path = (
                    Path(self.output_dir)
                    / f"{ckpt_callback.monitor.replace('/', '.')}.ave_{len(checkpoints)}best.pth"
                )
                torch.save(new_avg_state_dict, avg_ckpt_path)


def get_default_callbacks(args):
    last_ckpt_callback = ModelCheckpoint(
        dirpath=args.output_dir,
        save_last="link",
        filename="step{step}",
        auto_insert_metric_name=False,
        save_on_train_epoch_end=True,
        save_weights_only=False,
    )

    best_ckpt_callbacks = []
    for monitor, mode, nbest in trainer_conf.pop("best_model_criterion", []):
        best_ckpt_callbacks.append(
            ModelCheckpoint(
                save_top_k=nbest,
                monitor=monitor,
                mode=mode,  # "min" or "max"
                dirpath=args.output_dir,
                save_last=False,
                # Add monitor to filename to avoid overwriting when multiple metrics are used
                filename="epoch{epoch}_step{step}_" + monitor.replace("/", "."),
                auto_insert_metric_name=False,
                save_on_train_epoch_end=False,
                save_weights_only=True,
                enable_version_counter=False,  # just overwrite
            )
        )

    # Average best models after training
    ave_ckpt_callback = AverageCheckpointsCallback(
        output_dir=args.output_dir, best_ckpt_callbacks=best_ckpt_callbacks
    )

    # Monitor learning rate
    lr_callback = LearningRateMonitor()

    # Progress bar
    progress_bar_callback = TQDMProgressBar(refresh_rate=args.lightning_conf["log_every_n_steps"])

    return [
        last_ckpt_callback,
        *best_ckpt_callbacks,  # unpack list to add them to the list of callbacks.
        ave_ckpt_callback,
        lr_callback,
        progress_bar_callback
    ]
