import pytorch_lightning as pl
from pytorch_lightning.strategies import DDPStrategy
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning import loggers as pl_loggers

mode = 'rad'

if mode == 'rad':
    from implicit_mri_autoencoder_radial import ImplicitMRIAutoencoder
    from dataloaders.dataloader_supervised import *
elif mode == 'cart':
    from implicit_mri_autoencoder import ImplicitMRIAutoencoder
    from dataloaders.dataloader_ocmr import *
    
b_eff = 1
#model_dict = torch.load('./lightning_logs/version_8/checkpoints/epoch=499-step=10000.ckpt')['state_dict']

#model_dict['cs.lmbda_t'] = torch.Tensor([0.09]).to('cuda')
model = ImplicitMRIAutoencoder(lr=5e-04)
#model.load_state_dict(model_dict)


# w = torch.Tensor(range(-144, 144)).abs().to(dtype=torch.complex64)
# w += 2
# w /= w.abs().max()
# w = torch.concat([w for i in range(5)])
# w = w.unsqueeze(0).unsqueeze(0)
# fact = 0.3
# w = torch.min(torch.Tensor([fact]), w.real).to(dtype=torch.complex64)
# w /= fact
# model.dcomp = w #torch.ones_like(w)

#model = ImplicitMRIAutoencoder.load_from_checkpoint('./lightning_logs/version_108/checkpoints/epoch=499-step=10000.ckpt')
#model.t *= 0.6/model.t

#model.learning_rate = 5e-04

train_ds = TrainingDataset()
train_dl = DataLoader(train_ds, batch_size=b_eff, num_workers=1, pin_memory=True, shuffle=True)
val_ds = ValidationDataset()
val_dl = DataLoader(val_ds, batch_size=b_eff, num_workers=2, pin_memory=True, shuffle=True, persistent_workers=True)

# pytorch lightning
lr_monitor = LearningRateMonitor(logging_interval='step')
#tb_logger = pl_loggers.TensorBoardLogger(save_dir="radial_logs/")
trainer = pl.Trainer(
    accelerator="gpu",
    devices=[1],
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
