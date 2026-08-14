import argparse
import os
import random

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

from data import MyDataLoader
from model_2 import node_update
from utils_return_indivial_rates import discrete_mapping

# -------------------------------
# FULL DETERMINISM SETUP
# -------------------------------

SEED = 0

random.seed(SEED)
os.environ["PYTHONHASHSEED"] = str(SEED)

np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)  # for multi-GPU
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
torch.use_deterministic_algorithms(True, warn_only=True)

# Optional but helps with reproducibility in data loading
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

print(f"[INFO] All random seeds fixed to {SEED} for full determinism.")

# Debugging helpers
torch.autograd.set_detect_anomaly(True)   # shows backward-pass NaN source

def _assert_finite(where, *tensors):
    for t in tensors:
        if torch.is_tensor(t) and not torch.isfinite(t).all():
            t_ = torch.nan_to_num(t)
            print(f"[Non-finite @ {where}] "
                  f"min={t_.min().item():.3e}, max={t_.max().item():.3e}, "
                  f"norm={t_.norm().item():.3e}, shape={tuple(t.shape)}")
            raise RuntimeError(f"Non-finite detected in {where}")

class Trainer():
    def __init__(self,M,N,L,K,batch_size,at,Pt=40, device="cuda:0"):                         
        self.M = M                            # num of antennas per AP
        self.N = N                            # num of elements per RIS
        self.K = K                            # num of users
        self.L = L                            # num of RISs                                            (Note: In paper, L refers to num of APs)
        self.Pmax = 10**((Pt-30)/10)
        self.batch_size = batch_size
        self.n_iter = 2000
        self.dataloader = MyDataLoader(M,N,L,batch_size)
        self.dataloader.BS_RIS_association()
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")          
        self.num_of_AP = 5
        self.model = node_update(M,N,L,6,self.Pmax,2,64,self.num_of_AP, self.device).to(self.device)      
        self.min_rate = 1
        self.log_interval = 10
        self.log_eval_interval = 500 

        self.training_associate_threshold = 0.1
        self.associate_threshold = 0.1
        self.dup = False                             
        self.sum_UE = np.zeros((self.num_of_AP))

    def train_batch(self):
        self.model.train()
        user_feature, e, user_index, e_dir, user_index_for_testing = self.dataloader.gen_training_data(self.K,self.training_associate_threshold,self.associate_threshold,duplicate=self.dup)
        user_feature = user_feature.to(self.device)
        e = e.to(self.device)
        e_dir = e_dir.to(self.device)
        for AP in range(self.num_of_AP):
            sum_of_ue = np.sum(user_index_for_testing[AP])
            self.sum_UE[AP] += sum_of_ue
            
        self.opt.zero_grad()
        W, theta = self.model(user_feature,e,user_index,e_dir,training=True,duplicate=self.dup)
        loss,sum_rate,rate  = self.dataloader.compute_loss(W,theta,self.Pmax, self.device)
        loss.backward()
        self.opt.step()

        return loss.item(),sum_rate.item(),rate.detach().cpu()  



    def train(self, run_id, out_dir, log_dir, test_sample_val, test_sample_final):

        # Create subfolders for models, arrays, logs
        model_dir = os.path.join(out_dir, "models")
        array_dir = os.path.join(out_dir, "arrays")
        os.makedirs(model_dir, exist_ok=True)
        os.makedirs(array_dir, exist_ok=True)
        os.makedirs(log_dir, exist_ok=True)
        os.makedirs(out_dir, exist_ok=True)  # make results directory if not exists

        writer = SummaryWriter(log_dir=log_dir)
        
        self.opt = torch.optim.Adam(self.model.parameters(),lr=0.0001,weight_decay=1e-6)
        train_loss = []
        train_sum_rate = []
        train_total = []

        val_sum_rate_centralized = []
        val_sum_rate_centralized_discrete = []
        val_sum_rate_centralized_random_phase = []
        val_sum_rate_centralized_random_phase_discrete = []
        val_sum_rate_decentralized = []
        val_sum_rate_decentralized_discrete = []
        val_sum_rate_decentralized_random_phase = []
        val_sum_rate_decentralized_random_phase_discrete = []
        val_EE  = [1]
        val_sigma1 = []
        val_sigma2 = []

        train_losses = []
        sum_rates = []

        best_loss = float("inf")
        best_sum_rate = -float("inf")
        best_loss_model_path = None
        best_sumrate_model_path = None

        for i in range(self.n_iter):
            loss, sum_rate, rate = self.train_batch()
            train_loss.append(loss)
            train_sum_rate.append(sum_rate)

            # TensorBoard
            writer.add_scalar("Train/Loss", loss, i)
            writer.add_scalar("Train/SumRate", sum_rate, i)
            
            for user_id, val in enumerate(rate):
                writer.add_scalar(f"UserRate/User_{user_id+1}", val, i)

            if i% self.log_interval == 0:
                print(f"[Train | {i}/{self.n_iter} ] loss = {np.mean(train_loss):.5f}, sum rate = {(np.mean(train_sum_rate)):.5f}")  #! I added sum_rate printing out
                writer.add_scalar("Train/np.mean(train_loss) Hank's ", np.mean(train_loss), i)
                writer.add_scalar("Train/np.mean(train_sum_rate) Hank's ", np.mean(train_sum_rate), i)
                
                train_total.append(np.mean(train_loss))
                train_loss = []
                train_sum_rate = []

            if (i+1)%self.log_eval_interval==0 and i>=1:
                centralized, centralized_random_phase, decentralized, decentralized_random_phase, centralized_discrete, decentralized_discrete, centralized_random_phase_discrete, decentralized_random_phase_discrete = self.eval(test_sample_val,0,i)  # 400 validation samples

                print(f"[Val Cen. | {i+1}/{self.n_iter} ] sum rate = {(centralized):.5f}")
                print(f"[Val Cen. + Discrete | {i+1}/{self.n_iter} ] sum rate = {(centralized_discrete):.5f}")
                print(f"[Val Cen. + Random Phase | {i+1}/{self.n_iter} ] sum rate = {(centralized_random_phase):.5f}")
                print(f"[Val Cen. + Random Phase + Discrete| {i+1}/{self.n_iter} ] sum rate = {(centralized_random_phase_discrete):.5f}")
                
                print(f"[Val Decen. | {i+1}/{self.n_iter} ] sum rate = {(decentralized):.5f}")
                print(f"[Val Decen. + Discrete | {i+1}/{self.n_iter} ] sum rate = {(decentralized_discrete):.5f}")
                print(f"[Val Decen. + Random Phase | {i+1}/{self.n_iter} ] sum rate = {(decentralized_random_phase):.5f}")
                print(f"[Val Decen. + Random Phase + Discrete | {i+1}/{self.n_iter} ] sum rate = {(decentralized_random_phase_discrete):.5f}")

                val_sum_rate_centralized.append(centralized)
                val_sum_rate_centralized_discrete.append(centralized_discrete)
                val_sum_rate_centralized_random_phase.append(centralized_random_phase)
                val_sum_rate_centralized_random_phase_discrete.append(centralized_random_phase_discrete)
                val_sum_rate_decentralized.append(decentralized)
                val_sum_rate_decentralized_discrete.append(decentralized_discrete)
                val_sum_rate_decentralized_random_phase.append(decentralized_random_phase)
                val_sum_rate_decentralized_random_phase_discrete.append(decentralized_random_phase_discrete)

                
                # Log validation metrics
                writer.add_scalar("Val/Centralized", centralized, i+1)
                writer.add_scalar("Val/Centralized_Discrete", centralized_discrete, i+1)
                writer.add_scalar("Val/Centralized_RandomPhase", centralized_random_phase, i+1)
                writer.add_scalar("Val/Centralized_RandomPhase_Discrete", centralized_random_phase_discrete, i+1)
                writer.add_scalar("Val/Decentralized", decentralized, i+1)
                writer.add_scalar("Val/Decentralized_Discrete", decentralized_discrete, i+1)
                writer.add_scalar("Val/Decentralized_RandomPhase", decentralized_random_phase, i+1)
                writer.add_scalar("Val/Decentralized_RandomPhase_Discrete", decentralized_random_phase_discrete, i+1)

        print("Running FINAL evaluation with more samples...")
        centralized, centralized_random_phase, decentralized, decentralized_random_phase, centralized_discrete, decentralized_discrete, centralized_random_phase_discrete, decentralized_random_phase_discrete = self.eval(test_sample_final, 0, self.n_iter)

        print(f"[Final Eval] Cen. = {centralized:.5f}, "
              f"Cen.+Discrete = {centralized_discrete:.5f}, "
              f"Cen.+Rand = {centralized_random_phase:.5f}, "
              f"Cen.+Rand+Discrete = {centralized_random_phase_discrete:.5f}, "
              f"Decen. = {decentralized:.5f}, "
              f"Decen.+Discrete = {decentralized_discrete:.5f}, "
              f"Decen.+Rand = {decentralized_random_phase:.5f}, "
              f"Decen.+Rand+Discrete = {decentralized_random_phase_discrete:.5f}, ")
        

        # save into arrays for later analysis
        final_eval_results = {
            "centralized": centralized,
            "centralized_discrete": centralized_discrete,
            "centralized_random_phase": centralized_random_phase,
            "centralized_random_phase_discrete": centralized_random_phase_discrete,
            "decentralized": decentralized,
            "decentralized_discrete": decentralized_discrete,
            "decentralized_random_phase": decentralized_random_phase,
            "decentralized_random_phase_discrete": decentralized_random_phase_discrete
        }

        final_dir = os.path.join(out_dir, "final_eval")
        os.makedirs(final_dir, exist_ok=True)

        np.save(os.path.join(final_dir, f"final_eval_run{run_id}.npy"), final_eval_results)
        with open(os.path.join(final_dir, f"final_eval_run{run_id}.txt"), "w") as f:
            for k, v in final_eval_results.items():
                f.write(f"{k}: {v:.5f}\n")


        writer.close()                                                  #! double check if writer.close() is at the right place
        # Save final model
        final_model_path = os.path.join(model_dir, f"model_final_run{run_id}.pt")
        torch.save(self.model.state_dict(), final_model_path)


        # Save training logs
        np.save(os.path.join(array_dir, f"losses_run{run_id}.npy"), np.array(train_losses))
        np.save(os.path.join(array_dir, f"sumrates_run{run_id}.npy"), np.array(sum_rates))

        np.save(os.path.join(array_dir, f"train_total{run_id}.npy"), np.array(train_total))      
        np.save(os.path.join(array_dir, f"val_sum_rate_centralized{run_id}.npy"), np.array(val_sum_rate_centralized))
        np.save(os.path.join(array_dir, f"val_sum_rate_centralized_discrete{run_id}.npy"), np.array(val_sum_rate_centralized_discrete))
        np.save(os.path.join(array_dir, f"val_sum_rate_centralized_random_phase{run_id}.npy"), np.array(val_sum_rate_centralized_random_phase))
        np.save(os.path.join(array_dir, f"val_sum_rate_centralized_random_phase_discrete{run_id}.npy"), np.array(val_sum_rate_centralized_random_phase_discrete))
        np.save(os.path.join(array_dir, f"val_sum_rate_decentralized{run_id}.npy"), np.array(val_sum_rate_decentralized))
        np.save(os.path.join(array_dir, f"val_sum_rate_decentralized_discrete{run_id}.npy"), np.array(val_sum_rate_decentralized_discrete))
        np.save(os.path.join(array_dir, f"val_sum_rate_decentralized_random_phase{run_id}.npy"), np.array(val_sum_rate_decentralized_random_phase))
        np.save(os.path.join(array_dir, f"val_sum_rate_decentralized_random_phase_discrete{run_id}.npy"), np.array(val_sum_rate_decentralized_random_phase_discrete))

        train_total = np.array(train_total)
        val_EE = np.array(val_EE)

    def eval(self,test_sample,sigma,itera):
        
        self.model.eval()
        with torch.no_grad():                                      #! my addition
            iteration = int(test_sample/self.batch_size)
            num_bits = 2
            EE = []

            sum_rate_array_centralized = []
            sum_rate_array_centralized_discrete = []
            sum_rate_array_centralized_random_phase = []
            sum_rate_array_centralized_random_phase_discrete = []

            sum_rate_array_decentralized = []
            sum_rate_array_decentralized_discrete = []
            sum_rate_array_decentralized_random_phase = []
            sum_rate_array_decentralized_random_phase_discrete = []

            over_array = []

            for i in range(iteration):

                user_feature, e, user_index, e_dir, user_index_testing = self.dataloader.gen_training_data(self.K,self.training_associate_threshold,self.associate_threshold,duplicate=self.dup)
                user_feature = user_feature.to(self.device)
                e = e.to(self.device)
                e_dir = e_dir.to(self.device)
                
                # Centralized (C)
                W, theta = self.model(user_feature,e,user_index,e_dir,training=True,duplicate=self.dup)
                loss,sum_rate,rate = self.dataloader.compute_loss(W,theta,self.Pmax, self.device)
                sum_rate_array_centralized.append(sum_rate.item())

                # Centralized (D)
                theta_discrete = discrete_mapping(theta,num_bits)
                loss,sum_rate,rate = self.dataloader.compute_loss(W,theta_discrete,self.Pmax, self.device)
                sum_rate_array_centralized_discrete.append(sum_rate.item())

                # Centralized (C, continuous random)  Note: not included in paper
                theta = torch.rand_like(theta)
                theta = F.normalize(theta, dim=3)
                theta = theta.to(self.device)
                loss,sum_rate,rate = self.dataloader.compute_loss(W,theta,self.Pmax, self.device)
                sum_rate_array_centralized_random_phase.append(sum_rate.item())

                # Centralized (D-R)
                theta_rand_discrete = discrete_mapping(theta,num_bits)
                loss,sum_rate,rate = self.dataloader.compute_loss(W,theta_rand_discrete,self.Pmax, self.device)
                sum_rate_array_centralized_random_phase_discrete.append(sum_rate.item())

                # Decentralized (C)
                user_feature, e, user_index, e_dir = self.dataloader.gen_testing_data(self.K,self.associate_threshold,self.associate_threshold,False)
                mean_ue = torch.Tensor(self.sum_UE/((itera+1)*self.batch_size))
                mean_ue = mean_ue.to(self.device)

                for num_BS in range(len(user_feature)):
                    user_feature[num_BS] = user_feature[num_BS].to(self.device)
                    e[num_BS] = e[num_BS].to(self.device)
                    e_dir[num_BS] = e_dir[num_BS].to(self.device)
                W, theta = self.model(user_feature,e,user_index,e_dir,training=False,mean_ue=mean_ue)
                loss,sum_rate,rate = self.dataloader.compute_loss(W,theta,self.Pmax, self.device)
                sum_rate_array_decentralized.append(sum_rate.item())

                # Decentralized (D)
                theta_discrete = discrete_mapping(theta,num_bits)
                loss,sum_rate,rate = self.dataloader.compute_loss(W,theta_discrete,self.Pmax, self.device)
                sum_rate_array_decentralized_discrete.append(sum_rate.item())

                # Decentralized (C, continuous random)  Note: not included in paper
                theta = torch.rand_like(theta)
                theta = F.normalize(theta, dim=3)
                theta = theta.to(self.device)
                loss,sum_rate,rate = self.dataloader.compute_loss(W,theta,self.Pmax, self.device)
                sum_rate_array_decentralized_random_phase.append(sum_rate.item())

                # Decentralized (D-R)
                theta_rand_discrete = discrete_mapping(theta,num_bits)
                loss,sum_rate,rate = self.dataloader.compute_loss(W,theta_rand_discrete,self.Pmax, self.device)
                sum_rate_array_decentralized_random_phase_discrete.append(sum_rate.item())

            return np.mean(sum_rate_array_centralized), np.mean(sum_rate_array_centralized_random_phase),\
                    np.mean(sum_rate_array_decentralized), np.mean(sum_rate_array_decentralized_random_phase), \
                    np.mean(sum_rate_array_centralized_discrete), np.mean(sum_rate_array_decentralized_discrete), \
                    np.mean(sum_rate_array_centralized_random_phase_discrete), np.mean(sum_rate_array_decentralized_random_phase_discrete)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Trainer arguments")

    parser.add_argument("--M", type=int, default=4, help="Number of antennas per BS")
    parser.add_argument("--N", type=int, default=30, help="Number of RIS elements")
    parser.add_argument("--L", type=int, default=4, help="Number of RIS")
    parser.add_argument("--K", type=int, default=8, help="Number of users")
    parser.add_argument("--Pmax", type=float, default=10.0, help="Power budget")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--runs", type=int, default=5, help="Number of training runs")
    
    parser.add_argument("--bs_file", type=str, default="BS_{i}.txt", help="BS output filename pattern")
    parser.add_argument("--ris_file", type=str, default="RIS_{i}.txt", help="RIS output filename pattern")
    parser.add_argument("--user_file", type=str, default="USER_{i}.txt", help="User output filename pattern")
    parser.add_argument("--out_dir", type=str, default="results", help="Directory to save logs and models")
    parser.add_argument("--test_sample_val", type=int, default=100, help="Samples for periodic validation")
    parser.add_argument("--test_sample_final", type=int, default=3200, help="Samples for final evaluation")

    parser.add_argument("--device", type=str, default="cuda:0", help="Which GPU/CPU to use, e.g., 'cuda:0', 'cuda:1', or 'cpu'")
    args = parser.parse_args()

    exp_name = f"M{args.M}_N{args.N}_L{args.L}_K{args.K}_P{args.Pmax}"

    BS = []
    RIS = []
    user = []
    for i in range(args.runs):
        base_dir = os.path.join(args.out_dir, exp_name, f"run{i}")
        log_dir = os.path.join(base_dir, "logs")
        out_dir = base_dir

        trainer = Trainer(args.M,
                          args.N,
                          args.L,
                          args.K,
                          args.batch_size,
                          i+1, 
                          args.Pmax, 
                          device=args.device)       
        BS.append(trainer.dataloader.BS_Loc_array)
        RIS.append(trainer.dataloader.RIS_Loc_array)

        array_dir = os.path.join(base_dir, "arrays")
        os.makedirs(array_dir, exist_ok=True)

        bs_filename   = os.path.join(array_dir, args.bs_file.format(i=i))
        ris_filename  = os.path.join(array_dir, args.ris_file.format(i=i))

        np.savetxt(bs_filename, trainer.dataloader.BS_Loc_array, fmt='%f')
        np.savetxt(ris_filename, trainer.dataloader.RIS_Loc_array, fmt='%f')

        trainer.train(run_id=i, out_dir=out_dir, log_dir=log_dir, test_sample_val = args.test_sample_val, test_sample_final = args.test_sample_final)
