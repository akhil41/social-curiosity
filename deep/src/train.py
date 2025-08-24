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
from deep.src.agents import create_ppo_agent, save_agent
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
    
    # Environment configuration
    parser.add_argument('--num_agents', type=int, default=2,
                       help='Number of agents in the environment')
    parser.add_argument('--local_ratio', type=float, default=0.5,
                       help='Ratio of local reward to global reward')
    
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
    
    config['env']['num_agents'] = args.num_agents
    config['env']['local_ratio'] = args.local_ratio
    
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
        num_agents=config['env']['num_agents'],
        local_ratio=config['env']['local_ratio']
    )
    
    # Create evaluation environment
    eval_env = make_env(
        num_agents=config['env']['num_agents'],
        local_ratio=config['env']['local_ratio']
    )
    
    # Create PPO agent with TensorBoard logging
    ppo_config = config['ppo'].copy()
    ppo_config['tensorboard_log'] = config['logging']['tensorboard_log']
    agent = create_ppo_agent(env, ppo_config)
    
    # Create callback for saving and evaluation
    callback = SocialCuriosityCallback(
        save_path=config['paths']['models_dir'],
        save_interval=config['training']['save_interval'],
        eval_env=eval_env,
        eval_interval=config['training']['eval_interval'],
        n_eval_episodes=config['training']['n_eval_episodes'],
        verbose=1,
        model=agent,
        wandb_run=wandb_run
    )
    
    print(f"Starting training run: {args.run_name}")
    print(f"Intrinsic coefficient: {args.intrinsic_coef}")
    print(f"Total timesteps: {args.total_timesteps:,}")
    print(f"Environment: {config['env']['num_agents']} agents, local_ratio={config['env']['local_ratio']}")
    print(f"Logging: TensorBoard -> {config['logging']['tensorboard_log']}")
    print(f"Models will be saved to: {config['paths']['models_dir']}")
    print(f"Starting training at: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("-" * 60)
    
    try:
        # Start training
        agent.learn(
            total_timesteps=config['training']['total_timesteps'],
            callback=callback,
            tb_log_name=args.run_name,
            progress_bar=True
        )
        
        # Save final model
        final_model_path = os.path.join(config['paths']['models_dir'], 'final_model')
        save_agent(agent, final_model_path)
        print(f"Final model saved to: {final_model_path}")
        
    except KeyboardInterrupt:
        print("\nTraining interrupted by user. Saving current model...")
        interrupted_model_path = os.path.join(config['paths']['models_dir'], 'interrupted_model')
        save_agent(agent, interrupted_model_path)
        print(f"Interrupted model saved to: {interrupted_model_path}")
    
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