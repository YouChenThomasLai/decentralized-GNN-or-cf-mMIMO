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
    def __init__(self,M,N,L,batch_size):
        super(MyDataLoader,self).__init__()
        self.M = M
        self.N = N
        self.L = L
        self.batch_size = batch_size
        self.expanding_iter = 1500
        self.length = 100                 
        self.BS_Loc_array = gen_fixed_location(5, self.length*2)
        self.BS_array = [Base_station(M,N,L,[self.BS_Loc_array[0]]),Base_station(M,N,L,[self.BS_Loc_array[1]]), \
                         Base_station(M,N,L,[self.BS_Loc_array[2]]), Base_station(M,N,L,[self.BS_Loc_array[3]]), Base_station(M,N,L,[self.BS_Loc_array[4]])]

        self.RIS_Loc_array = gen_fixed_location(4, self.length)                                                                    
        self.RIS_array = [RIS(0,self.RIS_Loc_array[0]),RIS(1,self.RIS_Loc_array[1]),\
                          RIS(2,self.RIS_Loc_array[2]),RIS(3,self.RIS_Loc_array[3])]
        
        self.RIS_loc = []
        self.user_loc = []
    
    def BS_RIS_association(self):
        for BS in self.BS_array:
            BS.add_RIS(self.RIS_array[0])
            BS.add_RIS(self.RIS_array[1])
            BS.add_RIS(self.RIS_array[2])
            BS.add_RIS(self.RIS_array[3])

    def BS_user_association(self,K,ratio,testing_ratio):                               
        self.K = K
        loc_user = gen_location(K,self.length)                                              #! loc_user shape = (8, 2)
        self.user_loc.append(loc_user)
        
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
            user_feature0, e0, user_index0, e_dir0, user_index_for_testing0 = self.load_data(self.BS_array[0])
            user_feature1, e1, user_index1, e_dir1, user_index_for_testing1 = self.load_data(self.BS_array[1])
            user_feature2, e2, user_index2, e_dir2, user_index_for_testing2 = self.load_data(self.BS_array[2])
            user_feature3, e3, user_index3, e_dir3, user_index_for_testing3 = self.load_data(self.BS_array[3])
            user_feature4, e4, user_index4, e_dir4, user_index_for_testing4 = self.load_data(self.BS_array[4])
            user_feature = torch.cat((user_feature0,user_feature1,user_feature2,user_feature3,user_feature4),dim=2)
            e = torch.cat((e0,e1,e2,e3,e4),dim=2)
            user_index = np.concatenate((user_index0,user_index1,user_index2,user_index3,user_index4),axis=1)
            e_dir = torch.cat((e_dir0,e_dir1,e_dir2,e_dir3,e_dir4),axis=2)
            user_index_for_testing = [user_index_for_testing0,user_index_for_testing1,user_index_for_testing2,user_index_for_testing3,user_index_for_testing4]
           
            return user_feature, e, user_index, e_dir, user_index_for_testing
        

    def gen_testing_data(self,K,ratio,testing_ratio,duplicate=False):
        self.BS_user_association(K,ratio,testing_ratio)
        if duplicate==False:
            user_feature0, e0, user_index0, e_dir0, user_index_for_testing0 = self.load_data(self.BS_array[0])
            user_feature1, e1, user_index1, e_dir1, user_index_for_testing1 = self.load_data(self.BS_array[1])
            user_feature2, e2, user_index2, e_dir2, user_index_for_testing2 = self.load_data(self.BS_array[2])
            user_feature3, e3, user_index3, e_dir3, user_index_for_testing3 = self.load_data(self.BS_array[3])
            user_feature4, e4, user_index4, e_dir4, user_index_for_testing4 = self.load_data(self.BS_array[4])
            user_feature = [user_feature0,user_feature1,user_feature2,user_feature3,user_feature4]
            e = [e0,e1,e2,e3,e4]
            user_index = [user_index0,user_index1,user_index2,user_index3,user_index4]
            e_dir = [e_dir0,e_dir1,e_dir2,e_dir3,e_dir4]

            return user_feature, e, user_index, e_dir

    def compute_loss(self,W,theta,Pmax, device):                                                     
        H0, channel_bs_user0 = self.BS_array[0].get_channel()
        H1, channel_bs_user1 = self.BS_array[1].get_channel()
        H2, channel_bs_user2 = self.BS_array[2].get_channel()
        H3, channel_bs_user3 = self.BS_array[3].get_channel()
        H4, channel_bs_user4 = self.BS_array[4].get_channel()

        H = np.concatenate((H0,H1,H2,H3,H4),axis=4)
        channel_bs_user = np.concatenate((channel_bs_user0,channel_bs_user1,channel_bs_user2,channel_bs_user3,channel_bs_user4),axis=1)

        loss,sum_rate,rate = cal_loss(W, theta, H, channel_bs_user, Pmax, len(self.BS_array), device)

        return loss,sum_rate,rate


if __name__ == '__main__':
    M = 4
    N = 10
    L = 4
    K = 8
    batch_size = 2
    dataloader = MyDataLoader(M,N,L,batch_size)
    dataloader.BS_RIS_association()
    user_feature, e = dataloader.gen_training_data(K,0.5)
        
