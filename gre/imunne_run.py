import pytorch_lightning as pl
from pytorch_lightning.strategies import DDPStrategy
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning import loggers as pl_loggers

mode = 'rad'

from imunne_rad import IMUNNE
from dataloaders.dataloader_supervised import *
    
b_eff = 1
#model.learning_rate = 5e-04
model = ImplicitMRIAutoencoder(lr=5e-04)

train_ds = TrainingDataset()
train_dl = DataLoader(train_ds, batch_size=b_eff, num_workers=1, pin_memory=True, shuffle=True, persistent_workers=True)
val_ds = ValidationDataset()
val_dl = DataLoader(val_ds, batch_size=b_eff, num_workers=1, pin_memory=True, shuffle=True, persistent_workers=True)

# pytorch lightning
lr_monitor = LearningRateMonitor(logging_interval='step')
trainer = pl.Trainer(
    accelerator="gpu",
    devices=[0],
    callbacks = [lr_monitor],
#    logger=tb_logger,
#    strategy=DDPStrategy(find_unused_parameters=False),
#    strategy='dp',
    precision=32, 
    max_epochs=500,
    log_every_n_steps=1,
    accumulate_grad_batches=7,
    check_val_every_n_epoch=10,
    #track_grad_norm=2, 
    #gradient_clip_val=0.5
)
if __name__ == '__main__':
    trainer.fit(model, train_dl, val_dl)
