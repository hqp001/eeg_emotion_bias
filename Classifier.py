import lightning.pytorch as pl
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
import torchmetrics
import torch

class EEGClassifier(pl.LightningModule):
    def __init__(self, model, config):
        super().__init__()
        self.save_hyperparameters(ignore="model")
        self.model = model
        self.lr = config["lr"]
        # self.num_classes = num_classes
        self.mode = 'pre'
        self.val_acc = torchmetrics.Accuracy(task="multiclass", num_classes=2, top_k=1)
        self.micro_f1_score = torchmetrics.F1Score(task="multiclass", num_classes=2, average = 'micro')
        self.weighted_f1 = torchmetrics.F1Score(task="multiclass", num_classes=2, average = 'weighted')
        self.macro_f1_score = torchmetrics.F1Score(task="multiclass", num_classes=2, average = 'macro')
        self.trial_prediction = [0] * 23

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        X = batch[0]
        y = batch[1]

        logits = self.forward(X)
        loss = torch.nn.functional.cross_entropy(logits, y.long())
        self.log("train/loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        X = batch[0]
        y = batch[1]

        logits = self.forward(X)
        loss = torch.nn.functional.cross_entropy(logits, y.long())

        acc = self.val_acc(logits, y)
        mif1 = self.micro_f1_score(logits, y)
        wf1 = self.weighted_f1(logits, y)
        maf1 = self.macro_f1_score(logits, y)
        # eprint("val_acc", acc)
        # eprint("val_loss", loss)
        self.log("val/acc", acc, prog_bar = True)
        self.log("val/loss", loss, prog_bar = True)
        self.log("val/micro_f1", mif1, prog_bar=True)
        self.log("val/weighted_f1", wf1, prog_bar=True)
        self.log("val/macro_f1", maf1, prog_bar=True)


    def test_step(self, batch, batch_idx, dataloader_idx = 0):
        X = batch[0]
        y = batch[1]

        logits = self.forward(X)
        loss = torch.nn.functional.cross_entropy(logits, y.long())

        acc = self.val_acc(logits, y)
        mif1 = self.micro_f1_score(logits, y)
        wf1 = self.weighted_f1(logits, y)
        maf1 = self.macro_f1_score(logits, y)
        pre = ''
        if self.mode == 'pre':
            pre = 'pre'
        # eprint("test_acc", acc)
        # eprint("test_loss", loss)
        if dataloader_idx == 0:
            self.log(pre + "test/male/acc", acc, prog_bar = True, on_epoch = True)
            self.log(pre + "test/male/loss", loss, prog_bar = True, on_epoch = True)
            self.log(pre + "test/male/micro_f1", mif1, prog_bar = True, on_epoch = True)
            self.log(pre + "test/male/weighted_f1", wf1, prog_bar = True, on_epoch = True)
            self.log(pre + "test/male/macro_f1", maf1, prog_bar = True, on_epoch = True)


        else:
            self.log(pre + "test/female/acc", acc, prog_bar = True, on_epoch = True)
            self.log(pre + "test/female/loss", loss, prog_bar = True, on_epoch = True)
            self.log(pre + "test/female/micro_f1", mif1, prog_bar = True, on_epoch = True)
            self.log(pre + "test/female/weighted_f1", wf1, prog_bar = True, on_epoch = True)
            self.log(pre + "test/female/macro_f1", maf1, prog_bar = True, on_epoch = True)



    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr = self.lr)
        return optimizer
