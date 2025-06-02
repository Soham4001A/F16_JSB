import gymnasium as gym
import jsbsim_gym.ils_env # This will run the gym.register in f16_ils_env.py
from jsbsim_gym.features.ils_features import StackedLMA_ILS_FeaturesExtractor # Corrected import

from os import path
from stable_baselines3 import SAC, PPO
import torch
import argparse
import inspect # For AM-PPO check

def main(args_cli):
    if args_cli.cuda_device and torch.cuda.is_available():
        device = "cuda"
        print("Using CUDA device.")
    else:
        device = "cpu"
        print("Using CPU device.")
        if args_cli.cuda_device and not torch.cuda.is_available():
            print("CUDA specified but not available, falling back to CPU.")

    # Common Policy Kwargs for our new ILS Feature Extractor
    feature_extractor_policy_kwargs = dict(
        features_extractor_class=StackedLMA_ILS_FeaturesExtractor, # MODIFIED HERE
        features_extractor_kwargs=dict(
            # These are hyperparameters for StackedLMA_ILS_FeaturesExtractor,
            # which internally passes them to its LMAFeaturesExtractor instance.
            lma_embed_dim_d0=args_cli.lma_d0, # Pass from args
            lma_num_heads_stacking=args_cli.lma_heads_stack,
            lma_num_heads_latent=args_cli.lma_heads_latent,
            lma_ff_latent_hidden=args_cli.lma_ff_hidden,
            lma_num_layers=args_cli.lma_layers,          
            lma_dropout=args_cli.lma_dropout,
            lma_bias=True # Or make this an arg
        )
    )

    # Create the environment using the new env_id
    print(f"Creating environment with ID: {args_cli.env_id}")
    env = gym.make(args_cli.env_id, 
                   render_mode=args_cli.render_mode if args_cli.render_mode else None # Pass render_mode
                  )
    print(f"Environment observation space: {env.observation_space.shape}")
    print(f"Environment action space: {env.action_space.shape}")
    
    log_path = path.join(path.abspath(path.dirname(__file__)), 'logs_ils') # New log path
    model_save_dir = path.join(path.abspath(path.dirname(__file__)), 'models_ils') # New model save dir
    if not path.exists(model_save_dir):
        import os
        os.makedirs(model_save_dir)

    model = None

    if args_cli.algorithm.lower() == "sac":
        print("Selected Algorithm: SAC")
        policy_kwargs_sac = {**feature_extractor_policy_kwargs}
        # policy_kwargs_sac.update(net_arch=[256, 256]) # Example SAC architecture

        model_save_name = f"f16ils_sac_stacked_lma_seed{args_cli.seed}" # MODIFIED HERE
        buffer_save_name = f"f16ils_sac_stacked_lma_buffer_seed{args_cli.seed}" # MODIFIED HERE

        try:
            model = SAC(
                'MlpPolicy', # StackedLMA is a feature extractor, policy is still MlpPolicy
                env,
                verbose=1,
                policy_kwargs=policy_kwargs_sac,
                tensorboard_log=log_path,
                gradient_steps=-1,
                device=device,
                learning_rate=args_cli.learning_rate,
                gamma=args_cli.gamma,
                seed=args_cli.seed,
                buffer_size=args_cli.buffer_size_sac, 
                batch_size=args_cli.batch_size_sac_ppo,
                ent_coef=args_cli.ent_coef_sac,
                train_freq=args_cli.train_freq_sac,
                learning_starts=args_cli.learning_starts_sac,
            )
            print(f"Training with SAC on device: {model.device}")
            model.learn(total_timesteps=args_cli.total_timesteps, log_interval=10) # Added log_interval
        finally:
            if model:
                model.save(path.join(model_save_dir, model_save_name))
                if hasattr(model, 'save_replay_buffer'): # Check as not all SB3 versions might support this for all algos
                    model.save_replay_buffer(path.join(model_save_dir, buffer_save_name))
                print(f"SAC Model and Replay Buffer saved to {model_save_dir}")

    elif args_cli.algorithm.lower() == "ppo":
        print(f"Selected Algorithm: PPO {'(AM-PPO extensions active)' if args_cli.use_am_ppo else ''}")
        
        policy_kwargs_ppo = {**feature_extractor_policy_kwargs}
        # policy_kwargs_ppo.update(net_arch=dict(pi=[64, 64], vf=[128, 64])) # Example PPO architecture

        am_ppo_kwargs = {}
        if args_cli.use_am_ppo:
            print("AM-PPO extensions ENABLED for PPO training.")
            # Populate am_ppo_kwargs from args_cli... (same as your original)
            am_ppo_kwargs = {k: getattr(args_cli, k) for k in dir(args_cli) if k.startswith('dynago_') or k.startswith('am_ppo_')}
            am_ppo_kwargs['use_am_ppo'] = True # Ensure this is set

        model_save_name = f"f16ils_ppo_stacked_lma_seed{args_cli.seed}" # MODIFIED HERE
        if args_cli.use_am_ppo:
            model_save_name = f"f16ils_am_ppo_stacked_lma_seed{args_cli.seed}" # MODIFIED HERE
        
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
                batch_size=args_cli.batch_size_sac_ppo,
                n_epochs=args_cli.n_epochs_ppo, 
                gamma=args_cli.gamma,
                gae_lambda=args_cli.gae_lambda_ppo,
                clip_range=args_cli.clip_range_ppo,
                ent_coef=args_cli.ent_coef_ppo,
                vf_coef=args_cli.vf_coef_ppo,
                max_grad_norm=args_cli.max_grad_norm_ppo,
                seed=args_cli.seed,
                **am_ppo_kwargs 
            )
            print(f"Training with PPO on device: {model.device}")
            model.learn(total_timesteps=args_cli.total_timesteps, log_interval=1) # Added log_interval
        finally:
            if model:
                model.save(path.join(model_save_dir, model_save_name))
                print(f"PPO Model saved to {model_save_dir}")
    else:
        print(f"Error: Unknown algorithm '{args_cli.algorithm}'. Choose 'sac' or 'ppo'.")
        return

    env.close() # Close the environment after training

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train SAC or PPO/AM-PPO on F16ILSEnv.")
    parser.add_argument("--env_id", type=str, default="F16ILSEnv-v0", help="Gymnasium Environment ID") # MODIFIED HERE
    parser.add_argument("--render_mode", type=str, default=None, choices=[None, "human", "rgb_array"], help="Render mode for environment")

    parser.add_argument("--algorithm", type=str, default="sac", choices=["sac", "ppo"])
    parser.add_argument("--cuda_device", action="store_true")
    parser.add_argument("--total_timesteps", type=int, default=1_000_000) # Reduced for quicker testing
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--learning_rate", type=float, default=1e-4) # Adjusted common LR
    parser.add_argument("--gamma", type=float, default=0.99)

    # SAC specific args
    parser.add_argument("--buffer_size_sac", type=int, default=300_000) # Smaller buffer for potentially faster iteration
    parser.add_argument("--ent_coef_sac", type=str, default='auto')
    parser.add_argument("--train_freq_sac", type=int, default=1) # As per original
    parser.add_argument("--learning_starts_sac", type=int, default=10000) # Give more time to explore ILS

    # PPO specific args
    parser.add_argument("--n_steps_ppo", type=int, default=2048)
    parser.add_argument("--batch_size_sac_ppo", type=int, default=64) # Common batch/minibatch size
    parser.add_argument("--n_epochs_ppo", type=int, default=10)
    parser.add_argument("--gae_lambda_ppo", type=float, default=0.95)
    parser.add_argument("--clip_range_ppo", type=float, default=0.2)
    parser.add_argument("--ent_coef_ppo", type=float, default=0.001) # Small entropy for PPO
    parser.add_argument("--vf_coef_ppo", type=float, default=0.5)
    parser.add_argument("--max_grad_norm_ppo", type=float, default=0.5)

    # LMA Feature Extractor Hyperparameters (for StackedLMA_ILS_FeaturesExtractor)
    parser.add_argument("--lma_d0", type=int, default=64, help="LMA embed_dim (D0)")
    parser.add_argument("--lma_heads_stack", type=int, default=4, help="LMA num_heads_stacking")
    parser.add_argument("--lma_heads_latent", type=int, default=4, help="LMA num_heads_latent")
    parser.add_argument("--lma_ff_hidden", type=int, default=128, help="LMA ff_latent_hidden dim")
    parser.add_argument("--lma_layers", type=int, default=2, help="LMA num_layers (Transformer blocks)")
    parser.add_argument("--lma_dropout", type=float, default=0.1, help="LMA dropout rate")

    # AM-PPO specific args (same as your original)
    parser.add_argument("--use_am_ppo", action="store_true")
    parser.add_argument("--am_ppo_optimizer", type=str, default="DAG", choices=["Adam", "AlphaGrad", "DAG"])
    # ... (all other am_ppo and dynago args from your script) ...
    parser.add_argument("--am_ppo_alpha_optimizer_val", type=float, default=0.0)
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

    if args.algorithm.lower() == "ppo" and args.batch_size_sac_ppo > args.n_steps_ppo:
        print(f"Warning: PPO minibatch size ({args.batch_size_sac_ppo}) > n_steps_ppo ({args.n_steps_ppo}). Adjusting.")
        args.batch_size_sac_ppo = args.n_steps_ppo
    
    print("Starting training with arguments:")
    for arg, value in sorted(vars(args).items()):
        print(f"  {arg}: {value}")

    main(args)