import torch

def compute_accumulated_transmittance(betas):
    accumulated_transmittance=torch.cumprod(betas,1)

    return torch.cat((torch.ones(accumulated_transmittance.shape[0],1, device=accumulated_transmittance.device),
                      accumulated_transmittance[:,:-1]),dim=1)

def Rendering(model, rays_o, rays_d, tn, tf, nb_bins=100, device='cpu',white_bg=True):
    t=torch.linspace(tn,tf,nb_bins).to(device)
    delta=torch.cat((t[1:]-t[:-1], torch.tensor([1e10],device=device)))

    #[nb_bins]
    #[nb_bins,3]

    x=rays_o.unsqueeze(1)+t.unsqueeze(0).unsqueeze(-1)*rays_d.unsqueeze(1)

    colors, density=model.intersect(x.reshape(-1,3),rays_d.expand(x.shape[1],x.shape[0],3).transpose(0,1).reshape(-1,3))

    colors=colors.reshape((x.shape[0],nb_bins,3)) #[nb_rays, nb_bins, 3]
    density= density.reshape((x.shape[0],nb_bins))

    alpha=1-torch.exp(- density * delta.unsqueeze(0)) #[nb_rays, nb_bins]
    weights=compute_accumulated_transmittance(1-alpha)*alpha

    if white_bg:
        c=(weights.unsqueeze(-1)*colors).sum(1) #[nb_rays,3]
        weight_sum=weights.sum(-1)
        return c+1-weight_sum.unsqueeze(-1)
    else: 
        c=(weights.unsqueeze(-1)*colors).sum(1) #[nb_rays,3]

    return c