import torch

def construct_vs(kspace, traj, nspokes_p_frame=6, adj=9, gap=12, side=320, kwic='const'):
    #view sharing
    traj = traj.reshape(*traj.shape[:2], kspace.shape[3], kspace.shape[0])
    
    kspace = kspace.reshape(*kspace.shape[:2], -1)
    nspokes = kspace.shape[-1]//nspokes_p_frame
    kspace_vs = torch.zeros(*kspace.shape[:2], nspokes, nspokes_p_frame+2*adj, dtype=torch.complex64).to(kspace.device)
    
    
    traj = traj.permute(1,3,0,2)
    traj = traj.reshape(*traj.shape[:2], -1)
    traj_vs = torch.zeros(*traj.shape[:2], nspokes, nspokes_p_frame+2*adj).to(kspace.device)
    
    for fr in range(kspace_vs.shape[2]):
        kspace_vs[
            :,:,fr, max(adj - nspokes_p_frame*fr, 0) : min(kspace_vs.shape[3],kspace.shape[2]-(nspokes_p_frame*fr-adj))
        ] = kspace[
            :,:,max(0,nspokes_p_frame*fr-adj) : min(nspokes_p_frame*(fr+1)+adj, kspace.shape[2])
        ]
        
        traj_vs[
            :,:,fr, max(adj - nspokes_p_frame*fr, 0) : min(traj_vs.shape[3],traj.shape[2]-(nspokes_p_frame*fr-adj))
        ] = traj[
            :,:,max(0,nspokes_p_frame*fr-adj) : min(nspokes_p_frame*(fr+1)+adj, traj.shape[2])
        ]
    
        
    #KWIC filtering
    if kwic == 'const':
        sp_len = kspace_vs.shape[0]
        kspace_vs[sp_len//2-gap//2:sp_len//2+gap//2,:,:,[list(range(adj)) + list(range(adj+nspokes_p_frame, 2*adj+nspokes_p_frame))]] = 0
        traj_vs[:, sp_len//2-gap//2:sp_len//2+gap//2,:,[list(range(adj)) + list(range(adj+nspokes_p_frame, 2*adj+nspokes_p_frame))]] = 0
        
    traj_vs = traj_vs.permute(2,0,3,1)
    traj_vs = traj_vs.reshape(*traj_vs.shape[:2],-1)
    return kspace_vs, traj_vs