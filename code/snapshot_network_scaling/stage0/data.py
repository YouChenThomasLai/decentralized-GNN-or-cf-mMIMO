from utils_return_indivial_rates import *
import time
import random
import numpy as np
import torch
from torch.utils.data import Dataset
import torch.nn as nn
import torch.nn.functional as F

class Base_station(Dataset):
    def __init__(self,M,N,L,loc):
        super(Base_station,self).__init__()
        self.M = M
        self.N = N
        self.L = L
        self.loc = loc
        self.RIS_array = []
        self.user_array = []
        self.user_array_for_testing = []
        self.LOS_bs_ris = []
        self.channel = 0
        self.H = 0
        self.channel_bs_user = 0
        self.LOS_bs_ris = gen_LOS(self.N,self.M,10,self.L)


    def get_loc(self):
        return self.loc

    def add_RIS(self,RIS):
        self.RIS_array.append(RIS)

    def get_RIS_loc(self):
        RIS_loc = np.zeros((len(self.RIS_array),2))
        for i in range(len(self.RIS_array)):
            RIS_loc[i] = np.array(self.RIS_array[i].get_loc())
        return RIS_loc

    def get_RIS_id(self):
        RIS_id = []
        for i in range(len(self.RIS_array)):
            RIS_id.append(self.RIS_array[i].get_id())
        return RIS_id

    def set_user(self,users):
        self.user_array = users

    def set_user_for_testing(self,users_for_testing):
        self.user_array_for_testing = users_for_testing

    def get_user(self):
        return self.user_array

    def get_user_for_testing(self):
        return self.user_array_for_testing

    def get_user_loc(self):
        return np.array(self.user_array)

    def get_LOS(self):
        return self.LOS_bs_ris

    def set_channel(self,H,channel_bs_user):
        self.H = H
        self.channel_bs_user = channel_bs_user

    def get_channel(self):
        return self.H, self.channel_bs_user





class RIS(Dataset):
    def __init__(self,id,loc):
        super(RIS,self).__init__()
        self.id = id
        self.loc = loc

    def get_id(self):
        return self.id

    def get_loc(self):
        return self.loc



class MyDataLoader(Dataset):
    def __init__(self,M,N,L,batch_size,num_ap=5,spatial_scale=100):
        super(MyDataLoader,self).__init__()
        if num_ap <= 0 or spatial_scale <= 0:
            raise ValueError("num_ap and spatial_scale must be positive")
        self.M = M
        self.N = N
        self.L = L
        self.batch_size = batch_size
        self.expanding_iter = 1500
        self.length = spatial_scale
        self.BS_Loc_array = gen_fixed_location(num_ap, self.length*2)
        self.BS_array = [
            Base_station(M,N,L,[location])
            for location in self.BS_Loc_array
        ]

        self.RIS_Loc_array = gen_fixed_location(L, self.length)
        self.RIS_array = [
            RIS(ris_id, location)
            for ris_id, location in enumerate(self.RIS_Loc_array)
        ]

        self.RIS_loc = []
        self.user_loc = None
        self.received_power = None
        self.association_mask = None
        self.snapshot_id = 0

    def BS_RIS_association(self):
        for BS in self.BS_array:
            for ris in self.RIS_array:
                BS.add_RIS(ris)

    def BS_user_association(self,K,ratio,testing_ratio):
        self.K = K
        loc_user = gen_location(K,self.length)                                              #! loc_user shape = (8, 2)
        self.user_loc = loc_user

        #Channel Generation and Assignment
        RSSI = np.zeros((self.batch_size,K,len(self.BS_array)))
        for i in range(len(self.BS_array)):                                                 # for each AP
            loc_BS = self.BS_array[i].get_loc()
            loc_RIS = self.BS_array[i].get_RIS_loc()
            L = loc_RIS.shape[0]
            LOS = self.BS_array[i].get_LOS()                                                #! len(LOS) = 4, #! LOS[0].shape = (30, 4)
            H, channel_bs_user, perfect_H, perfect_bs_user = generate_channel(self.M,self.N,L,K,self.batch_size,LOS,0,loc_RIS,loc_BS,loc_user)
            #! H.shape = (32, 4, 30, 4, 8)
            #! channel_bs_user.shape = (32, 8, 4)

            self.BS_array[i].set_channel(H,channel_bs_user)
            user_array = np.zeros((self.batch_size,K))
            for k in range(K):
                temp_bs_user = channel_bs_user[:,k,:]
                temp_bs_user = temp_bs_user.reshape((self.batch_size,1,-1))
                w_h = np.matmul(temp_bs_user,np.conj(temp_bs_user.transpose(0,2,1)))
                user_array[:,k] = (np.real(w_h[:,0,0]))
            user_array = np.array(user_array)
            RSSI[:,:,i] = user_array

        user_index = RSSI >= np.max(RSSI,axis=2).reshape((self.batch_size,K,-1)).repeat(len(self.BS_array),axis=2)*ratio
        user_index_testing = RSSI >= np.max(RSSI,axis=2).reshape((self.batch_size,K,-1)).repeat(len(self.BS_array),axis=2)*testing_ratio
        for i in range(len(self.BS_array)):
            self.BS_array[i].set_user(user_index[:,:,i])
            self.BS_array[i].set_user_for_testing(user_index_testing[:,:,i])
        self.received_power = RSSI
        self.association_mask = user_index
        self.snapshot_id += 1


    def load_data(self,BS):
        H, channel_bs_user = BS.get_channel()
        batch_size = H.shape[0]
        L = len(BS.get_RIS_id())
        user_index = BS.get_user()
        user_index_for_testing = BS.get_user_for_testing()
        RIS_index = BS.get_RIS_id()

        user_feature = np.zeros((H.shape[0],self.K,self.L,2*self.M,self.N+1))   # Direct Channel and Non-direct Channel Information
        e = np.zeros((H.shape[0],self.K,self.L))                                # Direct Channel Quality and Non-direct Channel Quality
        e_dir = np.zeros((H.shape[0],self.K))                                   # Direct Channel Quality

        for l in range(L):
            temp_H = H[:,:,:,l,:].transpose((0,3,1,2))
            temp_H = temp_H[user_index,:,:]
            user_feature[user_index,RIS_index[l],:self.M,:self.N] = temp_H.real
            user_feature[user_index,RIS_index[l],self.M:,:self.N] = temp_H.imag

            w_H = np.matmul(temp_H,np.conj(temp_H.transpose(0,2,1)))
            w_H = np.trace(w_H,axis1=1,axis2=2)
            e[user_index,RIS_index[l]] = np.real(w_H)


            temp_bs_user = channel_bs_user[user_index,:]
            user_feature[user_index,RIS_index[l],:self.M,self.N] = temp_bs_user.real
            user_feature[user_index,RIS_index[l],self.M:,self.N] = temp_bs_user.imag

            if temp_bs_user.shape[0] != 0:
                temp_bs_user = temp_bs_user.reshape((temp_bs_user.shape[0],1,-1))
                w_h = np.matmul(temp_bs_user,np.conj(temp_bs_user.transpose(0,2,1)))
                e[user_index,RIS_index[l]] = e[user_index,RIS_index[l]]
                e_dir[user_index] = np.real(w_h[:,0,0])

        e_dir = torch.Tensor(e_dir).unsqueeze(1)
        user_feature = user_feature.reshape((batch_size,self.K,self.L,-1)).transpose((0,2,1,3))
        user_feature = torch.Tensor(user_feature)
        user_feature = F.normalize(user_feature,dim=2)
        e = torch.Tensor(e.transpose((0,2,1)))

        return user_feature, e, user_index, e_dir, user_index_for_testing


    def gen_training_data(self,K,ratio,testing_ratio,duplicate=False):
        self.BS_user_association(K,ratio,testing_ratio)
        if duplicate==False:
            ap_data = [self.load_data(BS) for BS in self.BS_array]
            user_feature = torch.cat([data[0] for data in ap_data],dim=2)
            e = torch.cat([data[1] for data in ap_data],dim=2)
            user_index = np.concatenate([data[2] for data in ap_data],axis=1)
            e_dir = torch.cat([data[3] for data in ap_data],axis=2)
            user_index_for_testing = [data[4] for data in ap_data]

            return user_feature, e, user_index, e_dir, user_index_for_testing


    def gen_testing_data(self,K,ratio,testing_ratio,duplicate=False,regenerate_channels=True):
        if regenerate_channels:
            self.BS_user_association(K,ratio,testing_ratio)
        if duplicate==False:
            ap_data = [self.load_data(BS) for BS in self.BS_array]
            user_feature = [data[0] for data in ap_data]
            e = [data[1] for data in ap_data]
            user_index = [data[2] for data in ap_data]
            e_dir = [data[3] for data in ap_data]

            return user_feature, e, user_index, e_dir

    def compute_loss(self,W,theta,Pmax, device):
        channels = [BS.get_channel() for BS in self.BS_array]
        H = np.concatenate([channel[0] for channel in channels],axis=4)
        channel_bs_user = np.concatenate([channel[1] for channel in channels],axis=1)

        loss,sum_rate,rate = cal_loss(W, theta, H, channel_bs_user, Pmax, len(self.BS_array), device)

        return loss,sum_rate,rate

    def compute_rates(self,W,theta,Pmax,device):
        channels = [BS.get_channel() for BS in self.BS_array]
        H = np.concatenate([channel[0] for channel in channels],axis=4)
        channel_bs_user = np.concatenate([channel[1] for channel in channels],axis=1)
        return cal_loss(
            W, theta, H, channel_bs_user, Pmax, len(self.BS_array), device,
            return_raw=True,
        )[3]

    def get_association_mask(self):
        if self.association_mask is None:
            raise RuntimeError("No stored association mask")
        return self.association_mask

    def get_topology_batch(self):
        if self.user_loc is None or self.received_power is None:
            raise RuntimeError("No stored topology batch")
        distances = np.linalg.norm(
            self.user_loc[:,None,:] - self.BS_Loc_array[None,:,:], axis=2
        )
        return {
            "user_locations": self.user_loc.copy(),
            "nearest_ap_distance": distances.min(axis=1),
            "strongest_received_power": self.received_power.max(axis=2),
        }


if __name__ == '__main__':
    M = 4
    N = 10
    L = 4
    K = 8
    batch_size = 2
    dataloader = MyDataLoader(M,N,L,batch_size)
    dataloader.BS_RIS_association()
    user_feature, e = dataloader.gen_training_data(K,0.5)

