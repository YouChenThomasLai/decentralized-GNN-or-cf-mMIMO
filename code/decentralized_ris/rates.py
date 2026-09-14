import numpy as np
import torch


def random_locations(K, radius):
    center = np.array([0,0])
    locations = np.zeros((K,2))
    for k in range(K):
        theta = np.random.uniform(0,2*np.pi)
        length = np.random.uniform(0, radius)
        x = length*np.cos(theta)
        y = length*np.sin(theta)
        locations[k,:] = center + np.array([x,y])

    return locations

def ring_locations(K, radius):
   center = np.array([0,0])
   locations = np.zeros((K,2))
   theta = 2*np.pi/K
   for k in range(K):
       x = radius*np.cos(k*theta)
       y = radius*np.sin(k*theta)
       locations[k,:] = center + np.array([x,y])
   return locations

def generate_los(num_rev, num_trans, rician_factor, count):
    LOS_array = []
    for _ in range(count):
        AoA = np.ones((num_rev,1),dtype=np.complex128)
        AoD = np.ones((num_trans,1),dtype=np.complex128)
        angle_AoA = 2*np.pi*np.random.uniform(0,1)
        for n in range(1,num_rev):
            AoA[n,:] = np.exp(1j*n*np.pi*np.sin(angle_AoA))
        angle_AoD = 2*np.pi*np.random.uniform(0,1)

        for n in range(1,num_trans):
            AoD[n,:] = np.exp(1j*n*np.pi*np.sin(angle_AoD))
        mat_LOS = np.dot(AoA,np.conj(AoD.T))
        LOS = np.sqrt(rician_factor/(rician_factor+1))*mat_LOS
        if count == 1:
            return LOS
        LOS_array.append(LOS)

    return LOS_array

class Channel():
    def __init__(self,num_trans,num_rev,factor):
        self.num_trans = num_trans
        self.num_rev = num_rev
        self.factor = factor
        self.AoA = np.ones((num_rev,1),dtype=np.complex128)
        self.AoD = np.ones((num_trans,1),dtype=np.complex128)
        self.mat = np.zeros((num_rev,num_trans),dtype=np.complex128)
        self.NLOS = np.zeros((num_rev,num_trans),dtype=np.complex128)
        self.LOS = np.zeros((num_rev,num_trans),dtype=np.complex128)

    def generate_value(self,mean_NLOS,cov_NLOS,LOS):
        self.LOS = LOS
        R_NLOS = np.linalg.cholesky(cov_NLOS)
        mat_real_NLOS = np.ones((self.num_rev,self.num_trans))*mean_NLOS + np.matmul(np.random.randn(self.num_rev,self.num_trans),R_NLOS)
        mat_imag_NLOS = np.ones((self.num_rev,self.num_trans))*mean_NLOS + np.matmul(np.random.randn(self.num_rev,self.num_trans),R_NLOS)
        self.NLOS = np.sqrt(1/(self.factor+1))*(mat_real_NLOS+1j*mat_imag_NLOS)/np.sqrt(2)

        self.mat = self.NLOS + self.LOS

    def large_scale_loss(self,fading_NLOS,exp_NLOS,fading_LOS,exp_LOS,dist):

        self.large_scale_fading_NLOS = fading_NLOS*dist**(-exp_NLOS)
        self.large_scale_fading_LOS = fading_LOS*dist**(-exp_LOS)
        self.ori_mat = self.large_scale_fading_NLOS*self.NLOS+self.large_scale_fading_LOS*self.LOS
        self.mat = self.large_scale_fading_NLOS*self.NLOS+self.large_scale_fading_LOS*self.LOS

        return self.mat

def generate_channels(M, N, L, K, batch_size, LOS_bs_ris, sigma, loc_RIS, loc_BS, loc_user):
    Rician_factor = 10
    fading_BS_users = 10**(-4.5)
    path_loss_exp_BS_user = 3.5
    fading_BS_RIS = 10**(-0.5)
    fading_RIS_users = 10**(-0.5)
    path_loss_exp_LOS = 2
    path_loss_exp_NLOS = 2.5

    channel_bs_ris = []
    channel_ris_user = []
    channel_bs_user = []
    for sample in range(batch_size):

        tmp_bs_ris = []
        for l in range(L):
            H_BS_RIS = Channel(M,N,Rician_factor)
            H_BS_RIS.generate_value(0,np.eye(M),LOS_bs_ris[l])
            H_BS_RIS = H_BS_RIS.large_scale_loss(fading_BS_RIS,path_loss_exp_NLOS,fading_BS_RIS,path_loss_exp_LOS,np.linalg.norm(loc_BS-loc_RIS[l]))
            tmp_bs_ris.append(H_BS_RIS)
        channel_bs_ris.append(tmp_bs_ris)

        tmp_ris_user = [[] for l in range(L)]
        tmp_bs_user = []
        for k in range(K):

            for l in range(L):
                h_LOS = generate_los(1, N, Rician_factor, 1)
                h_RIS = Channel(N,1,Rician_factor)
                h_RIS.generate_value(0,np.eye(N),h_LOS)
                h_RIS = h_RIS.large_scale_loss(fading_RIS_users,path_loss_exp_NLOS,fading_RIS_users,path_loss_exp_LOS,np.linalg.norm(loc_user[k,:]-loc_RIS[l,:]))
                tmp_ris_user[l].append(np.diag(h_RIS[0]))

            h_LOS = generate_los(1, M, 0, 1)
            h_bs = Channel(M,1,0)
            h_bs.generate_value(0, np.eye(M), h_LOS)
            h_bs = h_bs.large_scale_loss(fading_BS_users,path_loss_exp_BS_user,0,0,np.linalg.norm(loc_BS-loc_user[k,:]))
            tmp_bs_user.append(h_bs[0])

        channel_ris_user.append(tmp_ris_user)
        channel_bs_user.append(tmp_bs_user)

    scale = -7
    channel_bs_ris = np.array(channel_bs_ris)/np.sqrt(10**scale)
    channel_ris_user = np.array(channel_ris_user)/np.sqrt(10**scale)
    channel_bs_user = np.array(channel_bs_user)/10**scale

    H = np.zeros((batch_size,M,N,L,K),dtype=np.complex128)


    for k in range(K):
        for l in range(L):
            channel_ris_user_tmp = channel_ris_user[:,l,k,:,:]
            combined_ris = np.matmul(channel_bs_ris[:,l,:,:].transpose(0,2,1),channel_ris_user_tmp)
            H[:,:,:,l,k] = combined_ris

    H_imperfect = np.zeros((batch_size,M,N,L,K),dtype=np.complex128)
    if sigma > 0:
        imperfect_bs_user = np.random.normal(0,sigma,size=channel_bs_user.shape) + 1j*np.random.normal(0,sigma,size=channel_bs_user.shape)
        imperfect_ris_user = np.random.normal(0,sigma,size=channel_ris_user.shape) + 1j*np.random.normal(0,sigma,size=channel_ris_user.shape)
        imperfect_bs_user = imperfect_bs_user + channel_bs_user
        imperfect_ris_user = imperfect_ris_user + channel_ris_user
        for k in range(K):
            for l in range(L):
                channel_ris_user_tmp = imperfect_ris_user[:,l,k,:,:]
                combined_ris = np.matmul(channel_bs_ris[:,l,:,:].transpose(0,2,1),channel_ris_user_tmp)
                H_imperfect[:,:,:,l,k] = combined_ris
        return H_imperfect, imperfect_bs_user, H, channel_bs_user
    return H, channel_bs_user, H, channel_bs_user

def quantize_phase(theta, num_bits):
    level = 2 ** num_bits
    angles = torch.arange(level, device=theta.device, dtype=theta.dtype)
    angles = angles * (2 * torch.pi / level)
    phase = torch.stack((angles.cos(), angles.sin()), dim=-1)
    distances = torch.linalg.vector_norm(theta.unsqueeze(-2) - phase, dim=-1)
    return phase[distances.argmin(dim=-1)]


def random_phase_like(theta, num_bits=None):
    if num_bits is None:
        phase = 2 * torch.pi * torch.rand_like(theta[..., 0])
    else:
        level = 2 ** num_bits
        phase = torch.randint(level, theta.shape[:-1], device=theta.device)
        phase = phase.to(theta.dtype) * (2 * torch.pi / level)
    return torch.stack((phase.cos(), phase.sin()), dim=-1)


def phase_levels(num_bits, device):
    angles = torch.arange(2 ** num_bits, device=device) * (2 * torch.pi / 2 ** num_bits)
    return torch.stack((angles.cos(), angles.sin()), dim=1)


class RatePrecompute:
    """Cache channel-dependent terms for repeated phase evaluations."""

    SIGMA = (2e-2) ** 2

    def __init__(self, simulator, device):
        channels, direct = simulator.all_channels()
        self.device = device
        self.num_bs = simulator.n_ap
        self.batch, self.M, self.N, self.L, self.K_tot = (
            channels.shape[0],
            channels.shape[1],
            channels.shape[2],
            channels.shape[3],
            channels.shape[4],
        )
        self.K_user = self.K_tot // self.num_bs

        channels = np.transpose(channels, (0, 3, 4, 1, 2))
        top = np.concatenate([channels.real, channels.imag], axis=4)
        bottom = np.concatenate([-channels.imag, channels.real], axis=4)
        embedded = np.concatenate([top, bottom], axis=3)
        self.embedded = torch.as_tensor(embedded, dtype=torch.float32, device=device)
        columns = self.embedded.reshape(
            self.batch, self.L, self.K_tot, 2 * self.M, 2, self.N
        )
        self.elements = columns.permute(0, 1, 5, 2, 3, 4).contiguous()
        self.direct = torch.as_tensor(
            np.concatenate([direct.real, direct.imag], axis=2),
            dtype=torch.float32,
            device=device,
        )

    def effective_channel(self, theta):
        phase = torch.cat((theta[..., 0], theta[..., 1]), dim=2)
        contribution = torch.einsum("blkam,blm->bka", self.embedded, phase)
        return contribution + self.direct

    def element_contribution(self, ris_index, element_index, value):
        return torch.einsum(
            "bkam,bm->bka", self.elements[:, ris_index, element_index], value
        )

    def sum_rate_from_effective(self, beamformer, effective):
        weights = beamformer.transpose(2, 1)
        real = weights[:, :, :self.M].reshape(
            self.batch, self.num_bs, self.K_user, self.M
        )
        imag = weights[:, :, self.M:].reshape(
            self.batch, self.num_bs, self.K_user, self.M
        )
        channel_real = effective[:, :, :self.M].reshape(
            self.batch, self.num_bs, self.K_user, self.M
        )
        channel_imag = effective[:, :, self.M:].reshape(
            self.batch, self.num_bs, self.K_user, self.M
        )
        received_real = (
            torch.einsum("bsjm,bsim->bsij", real, channel_real)
            - torch.einsum("bsjm,bsim->bsij", imag, channel_imag)
        ).sum(dim=1)
        received_imag = (
            torch.einsum("bsjm,bsim->bsij", imag, channel_real)
            + torch.einsum("bsjm,bsim->bsij", real, channel_imag)
        ).sum(dim=1)
        index = torch.arange(self.K_user, device=effective.device)
        signal = received_real[:, index, index] ** 2 + received_imag[:, index, index] ** 2
        power = received_real ** 2 + received_imag ** 2
        power[:, index, index] = 0.0
        interference = power.sum(dim=2)
        sinr = signal / (interference + self.SIGMA)
        return torch.log2(1.0 + sinr).sum(dim=1)

    def sum_rate(self, beamformer, theta):
        return self.sum_rate_from_effective(
            beamformer, self.effective_channel(theta)
        )

def sum_rate_loss(W, Theta, H, channel_bs_user, num_BS, device):
    batch_size = W.shape[0]
    W = torch.transpose(W,2,1)
    M = W.shape[2]//2
    W_re = W[:,:,:M]
    W_im = W[:,:,M:]

    K = W.shape[1]
    sigma = (2e-2)**2                                               #! If too low, varying Pmax seems to be useless

    concat_H = torch.zeros((batch_size,K,2*M,1)).to(device)
    for k in range(K):
        h_k = torch.cat((torch.Tensor(channel_bs_user[:,k,:].real),torch.Tensor(channel_bs_user[:,k,:].imag)),dim=1).unsqueeze(2).to(device)
        concat_H_l = torch.zeros((batch_size,2*M,1)).to(device)

        L = H.shape[3]
        for l in range(L):
            A = H[:,:,:,l,k]
            A_re1 = np.concatenate([A.real,A.imag],axis=2)
            A_re2 = np.concatenate([-A.imag,A.real],axis=2)
            A_re = np.concatenate([A_re1,A_re2],axis=1)
            A_re = torch.Tensor(A_re).to(device)

            phase_re = (Theta[:,l,:,0])
            phase_im = (Theta[:,l,:,1])
            phase_A = torch.cat((phase_re,phase_im),dim=1).unsqueeze(2)

            concat_H_l += torch.matmul(A_re,phase_A)

        concat_H[:,k,:,:] = concat_H_l + h_k

    signal = torch.zeros((batch_size,K,2)).to(device)
    interference = torch.zeros((batch_size,K,K,2)).to(device)
    for BS in range(num_BS):

        for k1 in range(K//num_BS):
            A = concat_H[:,k1+BS*K//num_BS,:,:]

            signal_power = []
            sum_power = torch.zeros(batch_size)
            for k2 in range(K//num_BS):
                W_re_tmp = W_re[:,k2+BS*K//num_BS,:].unsqueeze(2)
                W_im_tmp = W_im[:,k2+BS*K//num_BS,:].unsqueeze(2)
                W_mat1 = torch.cat((W_re_tmp,W_im_tmp),dim=2)
                W_mat2 = torch.cat((-W_im_tmp,W_re_tmp),dim=2)
                W_mat = torch.cat((W_mat1,W_mat2),dim=1)

                z_in = torch.matmul(W_mat.transpose(2,1),A).squeeze()
                signal_power.append(z_in)

                if k1==k2:
                    continue
                else:
                    interference[:,k1+BS*K//num_BS,k2+BS*K//num_BS,:] = z_in
            signal[:,k1+BS*K//num_BS,:] = signal_power[k1]

    signal = torch.sum(signal.reshape((batch_size,num_BS,K//num_BS,2)),dim=1)
    signal = torch.square(signal[:,:,0]) + torch.square(signal[:,:,1])

    for i in range(1,num_BS):
        interference[:,:K//num_BS,:K//num_BS,:] += interference[:,K//num_BS*i:K//num_BS*(i+1),K//num_BS*i:K//num_BS*(i+1),:]

    interference = torch.square(interference[:,:K//num_BS,:K//num_BS,0]) + torch.square(interference[:,:K//num_BS,:K//num_BS,1])

    SINR = torch.zeros((batch_size,K//num_BS)).to(device)                 #! SINR: shape = torch.Size([4, 8])

    for k1 in range(K//num_BS):
        gamma = signal[:,k1] / (torch.sum(interference[:,k1,:],dim=1)+sigma)
        SINR[:,k1] = gamma
    rate = (torch.log2(1+SINR))                                           #! rate.shape = torch.Size([4, 8])
    sum_rate = torch.sum(rate,dim=1)                                      #! sum_rate.shape = torch.Size([4])

    loss = -torch.mean(sum_rate)                                          #! No min rate penalty

    return loss, torch.mean(sum_rate), torch.mean(rate, dim=0)
