import torch

def comb_display(cs_scaled, refs):
    comb = torch.concat([cs_scaled.squeeze()] + [ref.squeeze() for ref in refs], dim=1)
    err = torch.concat([torch.zeros_like(cs_scaled.squeeze())] + [ref.squeeze() - cs_scaled.squeeze() for ref in refs], dim=1)/0.3
    comb = torch.concat([comb, err], dim=2)
    comb = torch.clamp(comb, max=torch.quantile(comb.abs().flatten(), 1.0))
    return comb