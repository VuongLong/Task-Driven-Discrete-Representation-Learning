#!/bin/bash
seed_max=5
ns=(3000 10000 30000)
episode=100
walls=("empty")
cb_dim=(50 100 500)
for seed in `seq ${seed_max}`; do
    for wall in "${walls[@]}"; do
        for n in "${ns[@]}"; do

            # continuous 
            python -m train_rep  --tag ${n}_continuous_phi_10_${wall} --walls ${wall}   --save --seed ${seed}  --n_samples ${n} -n 3000 -r 10 -c 10
            python -m train_agent  --tag ${n}_continuous_phi_10_${wall} --walls ${wall}  --phi_path ${n}_continuous_phi_10_${wall}  --save --agent dqn -e ${episode} --seed ${seed} -r 10 -c 10

            for cb in "${cb_dim[@]}"; do

                # VQWAE 
                python -m train_rep --quantize_coef 100 --codebook_size ${cb} --tag ${n}_quantize_phi_10_${cb}_${wall} --walls ${wall} --quantize  --save --seed ${seed} --use_vqwae --n_samples ${n} -n 3000 -r 10 -c 10
                python -m train_agent  --codebook_size ${cb} --tag ${n}_quantize_phi_10_${cb}_${wall} --walls ${wall} --quantize --use_vqwae  \
                        --phi_path ${n}_quantize_phi_10_${cb}_${wall}  --save --agent dqn -e ${episode} --seed ${seed} -r 10 -c 10


                # VQVAE 
                python -m train_rep --quantize_coef 1 --codebook_size ${cb}  --tag ${n}_quantize_novqwae_phi_10_${cb}_${wall} --walls ${wall} --quantize  --save --seed ${seed}  --n_samples ${n} -n 3000 -r 10 -c 10
                python -m train_agent  --codebook_size ${cb}  --tag ${n}_quantize_novqwae_phi_10_${cb}_${wall} --walls ${wall} --quantize   \
                        --phi_path ${n}_quantize_novqwae_phi_10_${cb}_${wall}  --save --agent dqn -e ${episode}  --seed ${seed} -r 10 -c 10


                # VQSAE
                python -m train_rep --quantize_coef 0.1 --codebook_size ${cb} --tag ${n}_quantize_vqsae_phi_10_${cb}_${wall} --walls ${wall} --quantize --use_sqvae --save --seed ${seed}  --n_samples ${n} -n 3000 -r 10 -c 10
                python -m train_agent  --codebook_size ${cb} --tag ${n}_quantize_vqsae_phi_10_${cb}_${wall} --walls ${wall} --quantize  --use_sqvae \
                        --phi_path ${n}_quantize_vqsae_phi_10_${cb}_${wall}  --save --agent dqn -e ${episode}  --seed ${seed} -r 10 -c 10

            done
        done
    done
done

