import torch
import torch.nn as nn
import torch.nn.functional as F
from data import *
from utils_return_indivial_rates import *


def quick_stats(x, name):
    x = x.detach()
    print(f"{name}: shape={tuple(x.shape)}  "
          f"min={x.min().item():.3e}  max={x.max().item():.3e}  "
          f"mean={x.mean().item():.3e}  std={x.std().item():.3e}")

class initial_layer(nn.Module):
    def __init__(self,M,N,L,ch, device):
        super(initial_layer,self).__init__()
        # self.device = 'cuda'                                         
        self.device = device
        self.ch = ch                                                   #! self.ch = 64
        self.M = M                                                     #! self.M = 4
        self.N = N                                                     #! self.N = 30
        self.L = L                                                     #! self.L = 4
        self.in_dim = 2*M*(N+1)                                        #! self.in_dim = 248

        self.fu = nn.Sequential(
            nn.Linear(self.in_dim,self.ch*2),
            nn.LeakyReLU(),                                        #! Previous: nn.ReLU()
            nn.Linear(self.ch*2,self.ch)
        )

        self.f = nn.Sequential(
            nn.Linear(self.ch*2,self.ch*2),
            nn.LeakyReLU(),                                        #! Previous: nn.ReLU()
            nn.Linear(self.ch*2,self.ch)
        )

    def forward(self,H,e,e_dir):
                                                                # H.shape = torch.Size([1, 4, 40, 248])
        batch_size = H.shape[0]                  
        uk = self.fu(H)                                         # uk.shape = torch.Size([1, 4, 40, 64])
        uk = torch.mean(uk,dim=1)                               # uk.shape = torch.Size([1, 40, 64])
        e_h = F.normalize(e,p=1,dim=2)                          # e_h.shape = torch.Size([1, 4, 40])
        e_dir = F.normalize(e_dir,p=1,dim=2)                    # e_dir.shape = torch.Size([1, 1, 40])
        rl_tmp1 = torch.matmul(e_dir,uk)                        # rl_tmp1.shape = torch.Size([1, 1, 64]) 
        rl_tmp2 = torch.matmul(e_h,uk)                          # rl_tmp2.shape = torch.Size([1, 4, 64])
        rl = torch.zeros((1, self.L, rl_tmp2.size()[2]*2)).to(self.device)     # rl.shape = torch.Size([1, 4, 128])
        for l in range(self.L):
            rl[:,l,:] = torch.cat((rl_tmp2[:,l,:], rl_tmp1[:,0,:]), dim = 1)
        rl = self.f(rl)                                         # rl.shape = torch.Size([1, 4, 64])

        return uk, rl

class node_update_layer(nn.Module):
    def __init__(self,in_dim,M,N,L,ch, device):
        super(node_update_layer,self).__init__()
        # self.device = 'cuda'                                         
        self.device = device
        self.M = M
        self.N = N
        self.L = L
        self.ch = ch
        layer = in_dim/ch
        self.in_dim_p = in_dim*3
        self.in_dim_c = in_dim*2
        self.in_dim_l = int(in_dim*2 + self.ch*layer)

        self.fu = nn.Sequential(
            nn.Linear(self.in_dim_p,self.ch*2),
            nn.LeakyReLU(),                                          #! Previous: nn.ReLU()
            nn.Linear(self.ch*2,self.ch)
        )

        self.f = nn.Sequential(
            nn.Linear(self.in_dim_l,self.ch*2),
            nn.LeakyReLU(),                                          #! Previous: nn.ReLU()
            nn.Linear(self.ch*2,self.ch)
        )

        self.fc = nn.Sequential(
            nn.Linear(self.in_dim_c,self.ch*2),
            nn.LeakyReLU(),                                          #! Previous: nn.ReLU()
            nn.Linear(self.ch*2,self.ch)
        )

        self.edge_update = nn.Sequential(
            nn.Linear(in_dim+self.ch,self.ch),
            nn.LeakyReLU(),                                          #! Previous: nn.ReLU()
            nn.Linear(self.ch,1)
        )

    def forward(self,uk, rl, e, A, e_dir):
        batch_size = uk.shape[0]
        self.K = uk.shape[1]
        uk_update = torch.zeros((batch_size,self.K,self.ch+uk.shape[-1])).to(self.device)       
        e_l = e.transpose(2,1)
        e_l = F.normalize(e_l,p=1,dim=2)
        mean_uk = torch.matmul(e_l,rl)
        for k in range(self.K):
            if self.K == 1:
                max_user = uk[:,k,:]
            else:
                max_user = torch.amax(uk[:,A[k],:],dim=1)
            tmp_uk = self.fu(torch.cat((uk[:,k,:],max_user,mean_uk[:,k,:]),dim=1))
            uk_update[:,k,:] = torch.cat((tmp_uk,uk[:,k,:]),dim=1)
        
        e_h = F.normalize(e,p=1,dim=2)
        e_dir = F.normalize(e_dir,p=1,dim=2)
        rl_tmp2 = torch.matmul(e_h,uk)
        rl_tmp1 = torch.matmul(e_dir,uk)
        mean_rl = torch.zeros((1, self.L, rl_tmp2.size()[2]*2)).to(self.device)
        for l in range(self.L):
            mean_rl[:,l,:] = torch.cat((rl_tmp2[:,l,:], rl_tmp1[:,0,:]), dim = 1)
        tmp_rl = self.f(torch.cat((rl,mean_rl),dim=2))
        rl_update = torch.cat((tmp_rl,rl),dim=2)

        return uk_update, rl_update



class BS_readout(nn.Module):
    def __init__(self,M,N,L,Pt,in_dim):
        super(BS_readout,self).__init__()
        self.M = M                                       # self.M = 4
        self.N = N                                       # self.N = 30
        self.L = L                                       # self.L = 4
        self.Pt = Pt                                     # self.Pt = 10.0
        self.in_dim = in_dim                             # self.in_dim = 256
        self.fu = nn.Linear(self.in_dim,self.M*2)

    def forward(self,uk):

        batch_size = uk.shape[0]
        self.K = uk.shape[1]
                                                        # uk.shape = torch.Size([1, 40, 256])
        W = self.fu(uk)                                 # W.shape = torch.Size([1, 40, 8])
        W = torch.transpose(W,2,1)                      # W.shape = torch.Size([1, 8, 40])
        
        #! NOTE: the following was previous normalization. Now normalization is done after coeff_DNN2 is called
        # K = W.shape[2]
        # W = F.normalize(W,dim=2)*np.sqrt(self.Pt)/self.K          

        return W

    
class RIS_readout_AP(nn.Module):
    def __init__(self,M,N,L,Pt,in_dim, in_dim2):
        super(RIS_readout_AP,self).__init__()
        self.M = M
        self.N = N
        self.L = L
        self.Pt = Pt
        self.in_dim = in_dim
        self.in_dim2 = in_dim2
        self.f = nn.Linear(self.in_dim,self.N*2)
        self.fe_AP = nn.Linear(32,self.N*2)                        #! 32 is hard coded, TODO: make this dynamic
        self.f_merge = nn.Linear(self.N*4,self.N*2)

    def forward(self, rl, e_AP):
        if e_AP.dim() == 1:                                        # handle (in_dim2,)
            e_AP = e_AP.unsqueeze(0)                               # (1, in_dim2)

        Rl = self.f(rl)                                            # (B, L, 2N)
        Rl_AP = self.fe_AP(e_AP)                                   # (B, 2N)
        Rl_AP = Rl_AP.unsqueeze(1).expand(-1, rl.size(1), -1)      # (B, L, 2N)
        Rl_custom = torch.cat((Rl, Rl_AP), dim=2)                  # (B, L, 4N)

        return Rl_custom

class RIS_merge(nn.Module):
    def __init__(self, N):
        super(RIS_merge,self).__init__()
        self.N = N
        self.f_merge = nn.Linear(self.N*4,self.N*2)

    def forward(self,Rl_list):
        """
        Rl_list: list of (B, L, 4N) tensors (from APs)
        return: (B, L, N, 2) final phase
        """
        # Merge across APs
        Rl = torch.stack(Rl_list, dim=0).sum(dim=0)                # (B, L, 4N)
        Rl = self.f_merge(Rl)                                      # (B, L, 2N)

        phase_re = Rl[:,:,:self.N].unsqueeze(3)
        phase_im = Rl[:,:,self.N:].unsqueeze(3)
        phase = torch.cat((phase_re,phase_im),dim=3)
        phase = F.normalize(phase,dim=3)

        return phase


class coeff_DNN2(nn.Module):                                       
    def __init__(self,M,N,L,Pt,in_dim,ch=64):
        super(coeff_DNN2,self).__init__()
        self.M = M                                  # self.M = 4
        self.N = N                                  # self.N = 30
        self.L = L                                  # self.L = 4
        self.Pt = Pt                                # self.Pt = 10.0
        self.in_dim = in_dim                        # self.in_dim = 256 (TODO: double check)
        self.ch = ch

        self.fu = nn.Sequential(                                 
            nn.Linear(self.in_dim,self.ch*2),
            nn.LeakyReLU(),                                 #! Previous: nn.ReLU()
            nn.Linear(self.ch*2,self.ch)
        )
        
        # final scalar output
        self.final = nn.Linear(self.ch, 1)
        self.out_act = nn.Sigmoid()                         # squash to (0,1)

    def forward(self,uk):
        batch_size = uk.shape[0]
        self.K = uk.shape[1]                                # TODO: check if this is redundant 
        W = self.fu(uk)                                     # [B, ?, ch]
        W = W.mean(dim=1)                                   # aggregate → [B, ch]
        out = self.final(W)                                 # [B, 1]
        out = torch.clamp(out, -20, 20)                     # safe range for sigmoid                  #! or else NaNs could occur
        out = self.out_act(out)                             # squash scalar → [0,1]
        
        return out


class node_update(nn.Module):
    def __init__(self,M,N,L,D,Pmax,b,ch,AP, device):
        super(node_update,self).__init__()
        self.device = device                                                    
        self.N = N
        self.D = D
        self.M = M
        self.L = L
        self.ch = ch
        self.AP = AP
        self.Pmax = Pmax
        self.init_user = initial_layer(M,N,L,ch, device)

        update_list = []
        for d in range(D):
            update_list.append(node_update_layer(ch*(d+1),M,N,L,ch, device))
        self.update_list = nn.ModuleList(update_list)

        AP_coeff_NN_list = []
        RIS_readout_AP_list = []
        for BS_idx in range(self.AP):
            AP_coeff_NN_list.append(coeff_DNN2(M,N,L,Pmax,ch*(D+1)))
            RIS_readout_AP_list.append(RIS_readout_AP(M,N,L,Pmax,ch*(D+1), in_dim2=None))        #! in_dim2 is NOT used for now
        self.AP_coeff_NN_list = nn.ModuleList(AP_coeff_NN_list)
        self.RIS_readout_AP_list = nn.ModuleList(RIS_readout_AP_list)
        self.RIS_merge = RIS_merge(N)

        self.BS_readout = BS_readout(M,N,L,Pmax,ch*(D+1))
    
    def forward(self,user_feature,e,user_index,e_dir,training=True,duplicate=False,mean_ue=[]):
        if training:
            #! user_feature.shape = torch.Size([32, 4, 40, 248])
            #! e.shape = torch.Size([32, 4, 40])
            #! user_index.shape = (32, 40)
            #! e_dir.shape = torch.Size([32, 1, 40])
            batch_size = user_feature.shape[0]
            L = user_feature.shape[1]
            duplicate_L = e.shape[1]

            W = torch.zeros((batch_size,2*self.M,user_feature.shape[2])).to(self.device)
            theta = torch.zeros((batch_size,L,self.N,2)).to(self.device)

            for sample in range(batch_size):

                ############################ preping data for initial layer ############################
                actual_served_node = np.copy(user_index[sample])                                            #! actual_served_node.shape = (40,)
                num_actual_served_node = len(actual_served_node[actual_served_node==True])    
                
                tmp_user_index = user_index[sample]==False                                                  #! tmp_user_index.shape = (40,)
                
                user_feature[sample,:,tmp_user_index,:] = 0                     # set black nodes to zero
                e[sample,:,tmp_user_index] = 0                                  # set black ris nodes to zero
                e_dir[sample,:,tmp_user_index] = 0
                
                user_index[sample][user_index[sample]==False]=True              # enable all nodes          #! user_index.shape = (32, 40)
                
                #! important input argument shape. How initial layer deals with multiple users
                user_feature_ext = user_feature[sample,:,user_index[sample],:].unsqueeze(0)                 #! user_feature_ext.shape = torch.Size([1, 4, 40, 248])
                

                A = user_pruning(user_feature_ext.shape[2],0.2,duplicate)                   #! Hank: pruning effect not used
                A = A == 1                                                                  #! Hank: pruning effect not used


                e_ext = e[sample,:,user_index[sample]].unsqueeze(0)
                e_ext = RIS_pruning(e_ext,0.2,duplicate)
                e_dir_ext = e_dir[sample,:].unsqueeze(0)

                ############################ initial layer & GNN ############################ 
                uk, rl = self.init_user(user_feature_ext,e_ext,e_dir_ext)
                for i, _ in enumerate(self.update_list):
                    update_layer = self.update_list[i]
                    uk, rl = update_layer(uk, rl, e_ext, A, e_dir_ext)
                    uk = uk.to(self.device)
                    rl = rl.to(self.device)

                ########################## power control ##########################
                AP_coeffs = []
                for BS_idx in range(self.AP):
                    AP_coeffs.append(self.AP_coeff_NN_list[BS_idx](uk))

                W_out = self.BS_readout(uk)                         # unnormalized                  #! W_out.shape = torch.Size([1, 8, 40])
                user = int(user_feature.shape[2]/self.AP)                                           #! user = 8
                for BS in range(self.AP):        
                    BS_served_node = np.array([False]).repeat(user_feature.shape[2])                #! BS_served_node.shape = (40,)
                    BS_served_node[BS*user:(BS+1)*user] = actual_served_node[BS*user:(BS+1)*user]
                    W[sample,:,BS_served_node] = W_out[:,:,BS_served_node].squeeze(0)
                
                #! W.shape = torch.Size([32, 8, 40])
                for BS in range(self.AP):
                    BS_W = W[sample,:,BS*user:(BS+1)*user].reshape((1,-1))
                    BS_W = F.normalize(BS_W,dim=1, eps=1e-8)*torch.sqrt(self.Pmax * AP_coeffs[BS])       
                    W[sample,:,BS*user:(BS+1)*user] = BS_W.reshape((2*self.M,-1))  

                ####################################  RIS readout AP (start) ####################################
                RIS_prior_merge = []
                for BS in range(self.AP):
                    e_AP = e[sample, :, BS*user:(BS+1)*user].reshape(1, -1)   # (1, 32) if user=8, L=4  
                    Rl_custom = self.RIS_readout_AP_list[BS](rl, e_AP)
                    RIS_prior_merge.append(Rl_custom)

                theta_out = self.RIS_merge(RIS_prior_merge)                                   #! theta_out.shape = torch.Size([1, 4, 30, 2])
                theta[sample] = theta_out.squeeze()                                           #! theta.shape = torch.Size([32, 4, 30, 2])

            return W, theta
        
        else:    # decentralized
            batch_size = user_feature[0].shape[0]

            W = torch.zeros((batch_size,2*self.M,user_feature[0].shape[2]*len(user_feature))).to(self.device)
            theta = torch.zeros((batch_size,user_feature[0].shape[1],self.N,2)).to(self.device)
            for sample in range(batch_size):
                RIS_prior_merge = []           
                served_node = np.zeros((len(user_feature)))
                rl_feature = torch.zeros((1,self.L,self.ch*(self.D+1))).to(self.device)

                used_BS = 0
                for num_BS in range(len(user_feature)): 
                    user_feature_cat = user_feature[0]           
                    e_cat = e[0]                                 
                    e_dir_cat = e_dir[0]                         
                    for i in range(len(user_feature)-1):
                        user_feature_cat = torch.cat((user_feature_cat, user_feature[i+1]),dim=2)
                        e_cat = torch.cat((e_cat,e[i+1]),dim=2)
                        e_dir_cat = torch.cat((e_dir_cat,e_dir[i+1]),dim=2)
                    k = len(user_index[num_BS][sample])
                    new_user_index = np.array([False]).repeat(len(user_feature)*k)
                    new_user_index[num_BS*k:(num_BS+1)*k] = user_index[num_BS][sample]
                    actual_served_node = np.copy(new_user_index)
                    num_actual_served_node = np.sum(actual_served_node)
                    served_node[num_BS] = num_actual_served_node
                    
                    # Additional local CSI acquisition
                    served_user_id = np.where(user_index[num_BS][sample]==True)[0] 
                    unknown_ue = user_index[num_BS][sample]==False
                    if len(served_user_id) != 0:
                        for tmp_BS in range(len(user_feature)):
                            if tmp_BS == num_BS:
                                continue
                            else:
                                for user_id in served_user_id:
                                    if user_index[tmp_BS][sample][user_id] == True:
                                        new_user_index[tmp_BS*k:(tmp_BS+1)*k][user_id] = True   
                            
                    new_user_index = new_user_index == False
                    user_feature_cat[sample,:,new_user_index,:] = 0
                    e_cat[sample,:,new_user_index] = 0
                    e_dir_cat[sample,:,new_user_index] = 0
                    new_user_index[new_user_index==False]=True
                    user_feature_ext = user_feature_cat[sample,:,new_user_index,:].unsqueeze(0)

                    A = user_pruning(user_feature_ext.shape[2],0,duplicate=False)
                    A = A == 1

                    e_ext = e_cat[sample,:,new_user_index].unsqueeze(0)  
                    e_dir_ext = e_dir_cat[sample,:,new_user_index].unsqueeze(0)
                    if np.sum(user_index[num_BS][sample]) == 0:
                        continue
                    else:
                        used_BS += 1
                        uk, rl = self.init_user(user_feature_ext,e_ext,e_dir_ext)
                        for i, _ in enumerate(self.update_list):
                            update_layer = self.update_list[i]
                            uk, rl = update_layer(uk, rl, e_ext, A, e_dir_ext)
                            uk = uk.to(self.device)
                            rl = rl.to(self.device)

                        AP_coeff = self.AP_coeff_NN_list[num_BS](uk)                             #! double check this line                        
                        W_out = self.BS_readout(uk)                                              # Without normalizing

                        user = k
                        tmp_W = W_out[0,:,:]
                        W[sample,:,actual_served_node] = tmp_W[:,actual_served_node]
                        
                        ############################### Per AP power normalization ###############################
                        BS_W = W[sample,:,num_BS*user:(num_BS+1)*user].reshape((1,-1))
                        BS_W = F.normalize(BS_W,dim=1, eps=1e-8)*torch.sqrt(self.Pmax * AP_coeff)          
                        W[sample,:,num_BS*user:(num_BS+1)*user] = BS_W.reshape((2*self.M,-1)) 
                       
                        ################################ RIS readout AP customize (start) ####################################
                        e_AP = e[num_BS][sample, :, :].reshape(1, -1)   # shape: (1, L*user)
                        Rl_custom = self.RIS_readout_AP_list[num_BS](rl, e_AP)
                        RIS_prior_merge.append(Rl_custom)

                theta_out = self.RIS_merge(RIS_prior_merge)                              
                theta[sample] = theta_out.squeeze()
            return W, theta


if __name__ == '__main__':
    M = 4                                                                                # num of antennas per AP
    N = 10                                                                               # num of elements per RIS
    L = 4                                                                                # num of RISs
    K = 8                                                                                # num of users
    Pmax = 10
    batch_size = 2
    dataloader = MyDataLoader(M,N,L,batch_size)
    dataloader.BS_RIS_association()
    user_feature, e, user_index = dataloader.gen_training_data(K,0.95)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")              
    user_feature = user_feature.to(device)                                               
    e = e.to(device)                                                                    
    
    model = node_update(M, N, L, 3, Pmax, 2, 4, device=device).to(device)                
    W, theta = model(user_feature,e,user_index)
    loss = dataloader.compute_loss(W,theta,Pmax)
    print(loss)