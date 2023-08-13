from ray import air, tune
from ray.air import session
from ray.tune import CLIReporter
from ray.tune.schedulers import ASHAScheduler, PopulationBasedTraining
from ray.tune.integration.pytorch_lightning import TuneReportCallback, TuneReportCheckpointCallback
import lightning.pytorch as pl
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from lightning.pytorch.loggers import WandbLogger
from Classifier import EEGClassifier


def train_test_eeg_no_tune(config, num_epochs = 1, num_gpus = 4, data = None, core_model = None, PREFIX = None, idx = None):
    # Set up Wandb logger
    model_name = core_model({"Name": True})
    wandb_logger = WandbLogger(project="Emotion", name=f"KFoldTest--{model_name}-{data.emotion}-{data.data_name}", group="KFoldTest")

    model = EEGClassifier(core_model(config), config)

    trainer = pl.Trainer(logger = wandb_logger, callbacks=[EarlyStopping(monitor="train/loss", mode="min")],
                         accelerator="gpu", devices=num_gpus, default_root_dir=PREFIX + "/path", max_epochs=num_epochs)

    trainer.test(model, data.gender_test_loader(idx, config["batch_size"]))
    
    # trainer.predict(model, data.)

    trainer.fit(model, data.train_loader(idx, config["batch_size"]))

    model.mode = 'post'

    trainer.test(model, data.gender_test_loader(idx, config["batch_size"]))

def train_eeg_tune(config, num_epochs=1, num_gpus=4, data = None, core_model = None, PREFIX = None, idx = None):
    tune_callback = TuneReportCallback(
        {
            "loss": "val/loss",
            "mean_accuracy": "val/accuracy"
        },
        on="validation_end"
    )

    trainer = pl.Trainer(callbacks=[tune_callback, EarlyStopping(monitor="val/loss", mode="min")],
                         accelerator="gpu", devices=num_gpus, default_root_dir=PREFIX + "/path", max_epochs=num_epochs)

    for (train_loader, val_loader) in data.train_val_loader(idx, config["batch_size"]):
        model = EEGClassifier(core_model(config), config)
        trainer.fit(model, train_loader, val_loader)

def tune_asha(dsm, model, PREFIX, idx, num_samples=10, num_epochs=1, gpus_per_trial=1):
    config = {
        "lr": tune.loguniform(1e-4, 1e-1),
        "batch_size": tune.choice([32, 64, 128, 256])
    }

    scheduler = ASHAScheduler(
        max_t=num_epochs,
        grace_period=1,
        reduction_factor=2
    )

    reporter = CLIReporter(
        parameter_columns=["lr", "batch_size"],
        metric_columns=["loss", "mean_accuracy", "training_iteration"]
    )

    train_fn_with_parameters = tune.with_parameters(train_eeg_tune, num_epochs=num_epochs, num_gpus=gpus_per_trial, data = dsm, core_model = model, PREFIX = PREFIX, idx = idx)

    resources_per_trial = {"cpu": 4, "gpu": gpus_per_trial}

    tuner = tune.Tuner(
        tune.with_resources(
            train_fn_with_parameters,
            resources=resources_per_trial
        ),
        tune_config=tune.TuneConfig(
            metric="loss",
            mode="min",
            scheduler=scheduler,
            num_samples=num_samples,
            reuse_actors = False
        ),
        run_config=air.RunConfig(
            name="tune_eeg_asha",
            progress_reporter=reporter
        ),
        param_space=config,
    )
    results = tuner.fit()

    return results.get_best_result().config
