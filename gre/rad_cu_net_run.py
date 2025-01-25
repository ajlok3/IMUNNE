import sys
import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor
import torchvision
from rad_cinenet_pl import CineNet
sys.path.append('..')
from dataloaders.dataloader_supervised import *

b_eff = 1
model = CUNet()
model.learning_rate = 5e-04
train_ds = TrainingDataset()
train_dl = DataLoader(train_ds, batch_size=b_eff, num_workers=4, pin_memory=True, shuffle=True, persistent_workers=True)
val_ds = ValidationDataset()
val_dl = DataLoader(val_ds, batch_size=b_eff, num_workers=4, pin_memory=True, shuffle=False, persistent_workers=True)

lr_monitor = LearningRateMonitor(logging_interval='step')
trainer = pl.Trainer(
    accelerator="gpu",
    devices=[1],
    precision=32, 
    max_epochs=1000,
    log_every_n_steps=1,
    accumulate_grad_batches=7, 
    gradient_clip_val=0.5,
    check_val_every_n_epoch=10,
    callbacks=[lr_monitor],
)

print(model.learning_rate)
trainer.fit(model, train_dl, val_dl)
