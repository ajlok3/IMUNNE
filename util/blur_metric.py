import torch
import torch.nn as nn

def blur_metric(I,a=320,b=320):
    weights = torch.Tensor([1.,1.,1.,1.,1.,1.,1.,1.,1.,]).unsqueeze(0).unsqueeze(0).unsqueeze(0)/9
    B_Ver = nn.functional.conv2d(I.unsqueeze(0).unsqueeze(0), weights, padding='same').squeeze()
    B_Hor = nn.functional.conv2d(I.unsqueeze(0).unsqueeze(0), weights.permute(0,1,3,2), padding='same').squeeze()

    D_F_Ver = (I[:,:-1] - I[:,1:]).abs();
    D_F_Hor = (I[:-1,:] - I[1:,:]).abs();

    D_B_Ver = (B_Ver[:,:-1]-B_Ver[:,1:]).abs();
    D_B_Hor = (B_Hor[:-1,:]-B_Hor[1:,:]).abs();

    T_Ver = D_F_Ver - D_B_Ver;
    T_Hor = D_F_Hor - D_B_Hor;

    V_Ver = torch.max(torch.zeros(1),T_Ver);
    V_Hor = torch.max(torch.zeros(1),T_Hor);

    S_D_Ver = D_F_Ver[1:b-1,1:a-1].sum();
    S_D_Hor = D_F_Hor[1:b-1,1:a-1].sum();

    S_V_Ver = V_Ver[1:b-1,1:a-1].sum();
    S_V_Hor = V_Hor[1:b-1,1:a-1].sum();

    blur_F_Ver = (S_D_Ver-S_V_Ver)/S_D_Ver;
    blur_F_Hor = (S_D_Hor-S_V_Hor)/S_D_Hor;

    blur = torch.max(blur_F_Ver,blur_F_Hor);
    return blur