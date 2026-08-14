#!/bin/bash



##################################### vary M (batch_size 8) ##################################### 
python trainer_2.py --M 1 --N 30 --L 4 --K 8 --Pmax 15 --batch_size 8 --runs 1 --device cuda:0 --out_dir results_batch_8_BS-radius_200_RIS-radius_100_vary_M
sleep 5

python trainer_2.py --M 2 --N 30 --L 4 --K 8 --Pmax 15 --batch_size 8 --runs 1 --device cuda:0 --out_dir results_batch_8_BS-radius_200_RIS-radius_100_vary_M
sleep 5

python trainer_2.py --M 3 --N 30 --L 4 --K 8 --Pmax 15 --batch_size 8 --runs 1 --device cuda:0 --out_dir results_batch_8_BS-radius_200_RIS-radius_100_vary_M
sleep 5

python trainer_2.py --M 4 --N 30 --L 4 --K 8 --Pmax 15 --batch_size 8 --runs 1 --device cuda:0 --out_dir results_batch_8_BS-radius_200_RIS-radius_100_vary_M
sleep 5

python trainer_2.py --M 5 --N 30 --L 4 --K 8 --Pmax 15 --batch_size 8 --runs 1 --device cuda:0 --out_dir results_batch_8_BS-radius_200_RIS-radius_100_vary_M
sleep 5

##################################### vary Pmax (batch_size 8) ##################################### 
python trainer_2.py --M 2 --N 30 --L 4 --K 8 --Pmax 5 --batch_size 8 --runs 1 --device cuda:0 --out_dir results_batch_8_BS-radius_200_RIS-radius_100_vary_Pmax
sleep 5

python trainer_2.py --M 2 --N 30 --L 4 --K 8 --Pmax 10 --batch_size 8 --runs 1 --device cuda:0 --out_dir results_batch_8_BS-radius_200_RIS-radius_100_vary_Pmax
sleep 5

python trainer_2.py --M 2 --N 30 --L 4 --K 8 --Pmax 15 --batch_size 8 --runs 1 --device cuda:0 --out_dir results_batch_8_BS-radius_200_RIS-radius_100_vary_Pmax
sleep 5

python trainer_2.py --M 2 --N 30 --L 4 --K 8 --Pmax 20 --batch_size 8 --runs 1 --device cuda:0 --out_dir results_batch_8_BS-radius_200_RIS-radius_100_vary_Pmax
sleep 5

python trainer_2.py --M 2 --N 30 --L 4 --K 8 --Pmax 25 --batch_size 8 --runs 1 --device cuda:0 --out_dir results_batch_8_BS-radius_200_RIS-radius_100_vary_Pmax
sleep 5

python trainer_2.py --M 2 --N 30 --L 4 --K 8 --Pmax 30 --batch_size 8 --runs 1 --device cuda:0 --out_dir results_batch_8_BS-radius_200_RIS-radius_100_vary_Pmax
sleep 5

python trainer_2.py --M 2 --N 30 --L 4 --K 8 --Pmax 35 --batch_size 8 --runs 1 --device cuda:0 --out_dir results_batch_8_BS-radius_200_RIS-radius_100_vary_Pmax
sleep 5