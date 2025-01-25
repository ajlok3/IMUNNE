import pytorch_lightning as pl
from pytorch_lightning.strategies import DDPStrategy
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning import loggers as pl_loggers
import sys
sys.path.append('..')
from dataloaders.dataloader_supervised import *
from imunne import IMUNNE
    
b_eff = 1

model = ImplicitMRIAutoencoder(lr=2e-04)
model.load_state_dict(model_dict)

train_ds = TrainingDataset()
train_dl = DataLoader(train_ds, batch_size=b_eff, num_workers=1, pin_memory=True, shuffle=True)
val_ds = ValidationDataset()
val_dl = DataLoader(val_ds, batch_size=b_eff, num_workers=2, pin_memory=True, shuffle=True, persistent_workers=True)

# pytorch lightning
lr_monitor = LearningRateMonitor(logging_interval='step')
#tb_logger = pl_loggers.TensorBoardLogger(save_dir="radial_logs/")
trainer = pl.Trainer(
    accelerator="gpu",
    devices=[0],
    callbacks = [lr_monitor],
#    logger=tb_logger,
#    strategy=DDPStrategy(find_unused_parameters=False),
#    strategy='dp',
    precision=32, 
    max_epochs=100,
    log_every_n_steps=1,
    accumulate_grad_batches=7,
    check_val_every_n_epoch=10,
    #track_grad_norm=2, 
    #gradient_clip_val=0.5
)
if __name__ == '__main__':
    trainer.fit(model, train_dl, val_dl)
