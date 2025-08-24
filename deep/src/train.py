#!/usr/bin/env python3
"""
Training script for Social Curiosity deep learning implementation.
Uses PettingZoo environment and stable-baselines3 PPO agent.
"""
import argparse
import os
import sys
import time
from typing import Dict, Any, Optional

# Add the project root to Python path for imports
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Import local modules
from deep.src.environment import make_env
from deep.src.agents import create_multi_ppo_agents, save_multi_agents
from deep.config.default import get_config
from deep.src.utils import setup_wandb, save_config, SocialCuriosityCallback


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Train Social Curiosity agent with PPO')
    
    # Experiment configuration
    parser.add_argument('--run_name', type=str, default='baseline',
                       help='Name of the training run')
    parser.add_argument('--intrinsic_coef', type=float, default=0.0,
                       help='Coefficient for intrinsic motivation reward')
    parser.add_argument('--total_timesteps', type=int, default=1_000_000,
                       help='Total number of timesteps to train for')
    
    
    # Logging configuration
    parser.add_argument('--log_interval', type=int, default=10,
                       help='Log progress every N episodes')
    parser.add_argument('--save_interval', type=int, default=100,
                       help='Save model every N episodes')
    parser.add_argument('--eval_interval', type=int, default=50,
                       help='Evaluate model every N episodes')
    
    # WandB configuration
    parser.add_argument('--wandb_project', type=str, default='social-curiosity-deep',
                       help='WandB project name')
    parser.add_argument('--wandb_entity', type=str, default=None,
                       help='WandB entity/username')
    
    return parser.parse_args()


def setup_directories(config: Dict[str, Any]):
    """Create necessary directories for the experiment."""
    os.makedirs(config['paths']['models_dir'], exist_ok=True)
    os.makedirs(config['paths']['logs_dir'], exist_ok=True)
    os.makedirs(config['paths']['checkpoints_dir'], exist_ok=True)
    os.makedirs(config['logging']['tensorboard_log'], exist_ok=True)


def main():
    """Main training function."""
    # Parse command line arguments
    args = parse_args()
    
    # Get default configuration
    config = get_config()
    
    # Update configuration with command line arguments
    config['training']['total_timesteps'] = args.total_timesteps
    config['training']['log_interval'] = args.log_interval
    config['training']['save_interval'] = args.save_interval
    config['training']['eval_interval'] = args.eval_interval
    
    
    # Set intrinsic coefficient from command line argument or config
    intrinsic_coef = args.intrinsic_coef if args.intrinsic_coef is not None else config['sim']['intrinsic_coef']
    
    config['logging']['wandb_project'] = args.wandb_project
    config['logging']['wandb_entity'] = args.wandb_entity
    
    # Setup directories
    setup_directories(config)
    
    # Save configuration
    config_path = os.path.join(config['paths']['logs_dir'], 'config.json')
    save_config(config, config_path)
    
    # Setup WandB logging
    wandb_run = setup_wandb(config, args.run_name)
    
    # Create training environment
    env = make_env(
        intrinsic_coef=intrinsic_coef
    )
    
    # Create evaluation environment
    eval_env = make_env(
        intrinsic_coef=intrinsic_coef
    )
    
    # Create PPO agents with TensorBoard logging
    ppo_config = config['ppo'].copy()
    ppo_config['tensorboard_log'] = config['logging']['tensorboard_log']
    agents = create_multi_ppo_agents(env, ppo_config, 2)  # Fixed 2 agents for our GridWorld
    
    # Create callback for saving and evaluation
    # Note: For multi-agent training, we'll need to modify the callback or create separate callbacks
    # For now, we'll use the first agent for the callback
    callback = SocialCuriosityCallback(
        save_path=config['paths']['models_dir'],
        save_interval=config['training']['save_interval'],
        eval_env=eval_env,
        eval_interval=config['training']['eval_interval'],
        n_eval_episodes=config['training']['n_eval_episodes'],
        verbose=1,
        model=agents[0],  # Use first agent for callback
        wandb_run=wandb_run
    )
    
    print(f"Starting training run: {args.run_name}")
    print(f"Intrinsic coefficient: {intrinsic_coef}")
    print(f"Total timesteps: {args.total_timesteps:,}")
    print(f"Logging: TensorBoard -> {config['logging']['tensorboard_log']}")
    print(f"Models will be saved to: {config['paths']['models_dir']}")
    print(f"Starting training at: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("-" * 60)
    
    try:
        # Start training for each agent
        # Note: This is a simplified approach where each agent is trained independently
        # A more sophisticated approach would involve coordinated multi-agent training
        for i, agent in enumerate(agents):
            print(f"Training agent {i+1}/{len(agents)}...")
            agent.learn(
                total_timesteps=config['training']['total_timesteps'] // len(agents),
                callback=callback if i == 0 else None,  # Only use callback for first agent
                tb_log_name=f"{args.run_name}_agent_{i}",
                progress_bar=True
            )
        
        # Save final models
        model_paths = [os.path.join(config['paths']['models_dir'], f'final_model_agent_{i}') for i in range(len(agents))]
        save_multi_agents(agents, model_paths)
        print(f"Final models saved to: {config['paths']['models_dir']}")
        
    except KeyboardInterrupt:
        print("\nTraining interrupted by user. Saving current models...")
        model_paths = [os.path.join(config['paths']['models_dir'], f'interrupted_model_agent_{i}') for i in range(len(agents))]
        save_multi_agents(agents, model_paths)
        print(f"Interrupted models saved to: {config['paths']['models_dir']}")
    
    finally:
        # Cleanup
        env.close()
        eval_env.close()
        
        if wandb_run:
            wandb_run.finish()
        
        print("Training completed!")
        print(f"Finished at: {time.strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == '__main__':
    main()