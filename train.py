import gym
import jsbsim_gym.jsbsim_gym 
from os import path
from jsbsim_gym.LMA_features import StackedLMAFeaturesExtractor
from stable_baselines3 import SAC, PPO # Import both SAC and PPO
import torch 
import argparse 

def main(args_cli):
    if args_cli.cuda_device and torch.cuda.is_available():
        device = "cuda"
        print("Using CUDA device.")
    else:
        device = "cpu"
        print("Using CPU device.")
        if args_cli.cuda_device and not torch.cuda.is_available():
            print("CUDA specified but not available, falling back to CPU.")

    # Common Policy Kwargs for Feature Extractor
    # (Applicable to both SAC and PPO if they use the same feature extractor setup)
    feature_extractor_policy_kwargs = dict(
        features_extractor_class=StackedLMAFeaturesExtractor,
        features_extractor_kwargs=dict(
            lma_embed_dim_d0=64,        
            lma_num_heads_stacking=4,   
            lma_num_heads_latent=4,     
            lma_ff_latent_hidden=128,   
            lma_num_layers=2, # Number of Transformer Blocks in LMA reduced dimensions           
            lma_dropout=0.1,
            lma_bias=True
        )
    )

    env = gym.make("JSBSim-v0")
    log_path = path.join(path.abspath(path.dirname(__file__)), 'logs')
    
    model = None # Initialize model variable

    if args_cli.algorithm.lower() == "sac":
        print("Selected Algorithm: SAC")
        policy_kwargs_sac = {**feature_extractor_policy_kwargs} 
        # Add any SAC specific policy_kwargs if needed, e.g., net_arch for SAC
        # policy_kwargs_sac.update(net_arch=[256, 256]) # Example SAC architecture

        model_save_name = "jsbsim_sac_stacked_lma"
        buffer_save_name = "jsbsim_sac_stacked_lma_buffer"

        try:
            model = SAC(
                'MlpPolicy',
                env,
                verbose=1,
                policy_kwargs=policy_kwargs_sac,
                tensorboard_log=log_path,
                gradient_steps=-1, # As per your original SAC setup
                device=device,
                learning_rate=args_cli.learning_rate, # Use general learning_rate arg
                gamma=args_cli.gamma,
                seed=args_cli.seed,
                # Add other SAC specific hyperparameters from args_cli if needed
                # buffer_size=args_cli.buffer_size_sac, 
                # batch_size=args_cli.batch_size_sac_ppo, # Name changed for clarity
                # ent_coef=args_cli.ent_coef_sac, # if different from PPO's
                # train_freq=args_cli.train_freq_sac,
                # learning_starts=args_cli.learning_starts_sac,
            )
            print(f"Training with SAC on device: {model.device}")
            model.learn(total_timesteps=args_cli.total_timesteps)
        finally:
            if model:
                model.save(f"models/{model_save_name}")
                model.save_replay_buffer(f"models/{buffer_save_name}")
                print(f"SAC Model and Replay Buffer saved to models/{model_save_name}")

    elif args_cli.algorithm.lower() == "ppo":
        print(f"Selected Algorithm: PPO {'(AM-PPO extensions active)' if args_cli.use_am_ppo else ''}")
        
        policy_kwargs_ppo = {**feature_extractor_policy_kwargs}
        # Buffing AM-PPO's value network architecture as SB3 shares policy_kwargs for both actor and critic
        policy_kwargs_ppo.update(net_arch=dict(pi=[64, 64], vf=[128, 64])) # Example PPO architecture
        
        am_ppo_kwargs = {}
        if args_cli.use_am_ppo:
            print("AM-PPO extensions ENABLED for PPO training.")
            am_ppo_kwargs = dict(
                use_am_ppo=True,
                am_ppo_optimizer=args_cli.am_ppo_optimizer,
                am_ppo_alpha_optimizer=args_cli.am_ppo_alpha_optimizer_val,
                dynago_tau=args_cli.dynago_tau,
                dynago_p_star=args_cli.dynago_p_star,
                dynago_kappa=args_cli.dynago_kappa,
                dynago_eta=args_cli.dynago_eta,
                dynago_rho=args_cli.dynago_rho,
                dynago_eps=args_cli.dynago_eps,
                dynago_alpha_min=args_cli.dynago_alpha_min,
                dynago_alpha_max=args_cli.dynago_alpha_max,
                dynago_rho_sat=args_cli.dynago_rho_sat,
                dynago_alpha_A_init=args_cli.dynago_alpha_A_init,
                dynago_prev_sat_A_init=args_cli.dynago_prev_sat_A_init,
                dynago_v_shift=args_cli.dynago_v_shift,
                am_ppo_norm_adv=args_cli.am_ppo_norm_adv,
            )

        model_save_name = "jsbsim_ppo_stacked_lma"
        if args_cli.use_am_ppo:
            model_save_name = "jsbsim_am_ppo_stacked_lma"

        # Sanity Check: Ensure PPO is imported correctly (This currently imports PPO from true stable_baselines3 install)
        import inspect
        print(f"PPO class imported from: {inspect.getfile(PPO)}")
        
        try:
            model = PPO(
                'MlpPolicy',
                env,
                verbose=1,
                policy_kwargs=policy_kwargs_ppo,
                tensorboard_log=log_path,
                device=device,
                learning_rate=args_cli.learning_rate,
                n_steps=args_cli.n_steps_ppo, 
                batch_size=args_cli.batch_size_sac_ppo, # Shared batch size for PPO
                n_epochs=args_cli.n_epochs_ppo, 
                gamma=args_cli.gamma,
                gae_lambda=args_cli.gae_lambda_ppo,
                clip_range=args_cli.clip_range_ppo,
                ent_coef=args_cli.ent_coef_ppo, # PPO specific ent_coef
                vf_coef=args_cli.vf_coef_ppo,
                max_grad_norm=args_cli.max_grad_norm_ppo,
                seed=args_cli.seed,
                **am_ppo_kwargs 
            )
            print(f"Training with PPO on device: {model.device}")
            model.learn(total_timesteps=args_cli.total_timesteps)
        finally:
            if model:
                model.save(f"models/{model_save_name}")
                print(f"PPO Model saved to models/{model_save_name}")
    else:
        print(f"Error: Unknown algorithm '{args_cli.algorithm}'. Choose 'sac' or 'ppo'.")
        return

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train SAC or PPO/AM-PPO on JSBSim.")
    parser.add_argument("--algorithm", type=str, default="ppo", choices=["sac", "ppo"], help="RL algorithm to use (sac or ppo)")
    parser.add_argument("--cuda_device", action="store_true", help="Use CUDA if available")
    parser.add_argument("--total_timesteps", type=int, default=3_000_000)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--learning_rate", type=float, default=2e-4, help="Learning rate for the optimizer")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor")

    # --- SAC specific args (add more as needed, or if they differ significantly from PPO) ---
    # parser.add_argument("--buffer_size_sac", type=int, default=1_000_000, help="Replay buffer size for SAC")
    # parser.add_argument("--ent_coef_sac", type=str, default='auto', help="Entropy coefficient for SAC (auto or float)")
    # parser.add_argument("--train_freq_sac", type=int, default=1, help="Update the model every N steps for SAC")
    # parser.add_argument("--learning_starts_sac", type=int, default=100, help="How many steps of random actions before starting to learn for SAC")
    
    # --- PPO specific args (prefix with _ppo if they might conflict with SAC or general args) ---
    parser.add_argument("--n_steps_ppo", type=int, default=2048, help="Num steps per rollout per env for PPO")
    # Batch size can be shared or specific. Let's make it shared for simplicity now.
    parser.add_argument("--batch_size_sac_ppo", type=int, default=256, help="Minibatch size for SAC updates OR PPO updates (PPO uses it as minibatch, SAC for sampling from buffer)") # Note: PPO's batch_size is often n_steps * num_envs, here it's minibatch_size
    parser.add_argument("--n_epochs_ppo", type=int, default=10, help="Num epochs for PPO update")
    parser.add_argument("--gae_lambda_ppo", type=float, default=0.95)
    parser.add_argument("--clip_range_ppo", type=float, default=0.2)
    parser.add_argument("--ent_coef_ppo", type=float, default=0.0, help="Entropy coefficient for PPO")
    parser.add_argument("--vf_coef_ppo", type=float, default=0.5)
    parser.add_argument("--max_grad_norm_ppo", type=float, default=0.5)

    # --- AM-PPO specific args (only relevant if algorithm is ppo and use_am_ppo is true) ---
    parser.add_argument("--use_am_ppo", action="store_true", help="Enable AM-PPO extensions (only if algorithm is ppo)")
    parser.add_argument("--am_ppo_optimizer", type=str, default="DAG", choices=["Adam", "AlphaGrad", "DAG"])
    parser.add_argument("--am_ppo_alpha_optimizer_val", type=float, default=0.0, help="Alpha for AlphaGrad optimizer")
    parser.add_argument("--dynago_tau", type=float, default=1.25)
    parser.add_argument("--dynago_p_star", type=float, default=0.10)
    parser.add_argument("--dynago_kappa", type=float, default=2.0)
    parser.add_argument("--dynago_eta", type=float, default=0.3)
    parser.add_argument("--dynago_rho", type=float, default=0.1)
    parser.add_argument("--dynago_eps", type=float, default=1e-5)
    parser.add_argument("--dynago_alpha_min", type=float, default=1e-12)
    parser.add_argument("--dynago_alpha_max", type=float, default=1e12)
    parser.add_argument("--dynago_rho_sat", type=float, default=0.98)
    parser.add_argument("--dynago_alpha_A_init", type=float, default=1.0)
    parser.add_argument("--dynago_prev_sat_A_init", type=float, default=0.10)
    parser.add_argument("--dynago_v_shift", type=float, default=0.0)
    parser.add_argument("--am_ppo_norm_adv", type=bool, default=True)

    args = parser.parse_args()

    # SB3 PPO's `batch_size` parameter is actually the minibatch size.
    # The total batch collected per rollout is `n_steps_ppo * num_envs`.
    # For single env, `n_steps_ppo` is the rollout buffer size.
    # Ensure PPO's minibatch_size (args.batch_size_sac_ppo) is less than or equal to n_steps_ppo.
    if args.algorithm.lower() == "ppo" and args.batch_size_sac_ppo > args.n_steps_ppo:
        print(f"Warning: PPO minibatch size ({args.batch_size_sac_ppo}) is greater than n_steps_ppo ({args.n_steps_ppo}). "
              f"Adjusting PPO minibatch size to {args.n_steps_ppo}.")
        args.batch_size_sac_ppo = args.n_steps_ppo


    main(args)