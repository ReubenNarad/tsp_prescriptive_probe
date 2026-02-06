import os
import argparse
import pickle
import glob
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from typing import Optional

import torch

# Use relative imports for modules in the same package
from .policy_hooked import HookedAttentionModelPolicy
from .reinforce_clipped import REINFORCEClipped


class TSPModelVisualizer:
    """
    Visualizer for TSP model solutions.
    Loads a model checkpoint and provides methods to decode solutions and visualize them.
    """
    
    def __init__(
        self, 
        run_path: str, 
        checkpoint_epoch: Optional[int] = None,
        device: Optional[str] = None
    ):
        """
        Initialize the TSP model visualizer.
        
        Args:
            run_path: Path to the run directory
            checkpoint_epoch: Specific epoch to load (if None, loads latest)
            device: Device to run model on
        """
        self.run_path = Path(run_path)
        
        # Set device
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        
        print(f"Using device: {self.device}")
        
        # Load environment and model
        self.env = self._load_env()
        self.config = self._load_config()
        self.model = self._load_model(checkpoint_epoch)
    
    def _load_env(self):
        """Load the environment from pickle file"""
        env_path = self.run_path / "env.pkl"
        if not env_path.exists():
            raise FileNotFoundError(f"Environment file not found at {env_path}")
        
        with open(env_path, "rb") as f:
            env = pickle.load(f)
        
        print(f"Loaded environment from {env_path}")
        return env
    
    def _load_config(self):
        """Load the configuration from the run path"""
        config_path = self.run_path / "config.json"
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found at {config_path}")
        
        import json
        with open(config_path, "r") as f:
            config = json.load(f)
        
        print(f"Loaded config from {config_path}")
        return config
    
    def _load_model(self, checkpoint_epoch=None):
        """
        Load the model from a checkpoint.
        
        Args:
            checkpoint_epoch: Specific epoch to load (if None, loads latest)
        
        Returns:
            Loaded model
        """
        checkpoint_dir = self.run_path / "checkpoints"
        
        if checkpoint_epoch is not None:
            # Load specific checkpoint
            checkpoint_path = checkpoint_dir / f"checkpoint_epoch_{checkpoint_epoch}.ckpt"
            if not checkpoint_path.exists():
                raise FileNotFoundError(f"Checkpoint file not found at {checkpoint_path}")
        else:
            # Find latest checkpoint
            checkpoint_files = glob.glob(str(checkpoint_dir / "checkpoint_epoch_*.ckpt"))
            if not checkpoint_files:
                raise FileNotFoundError(f"No checkpoint files found in {checkpoint_dir}")
            
            # Extract epoch numbers and get the latest
            epochs = [int(os.path.basename(f).split("checkpoint_epoch_")[1].split(".ckpt")[0]) 
                      for f in checkpoint_files]
            latest_epoch = max(epochs)
            checkpoint_path = checkpoint_dir / f"checkpoint_epoch_{latest_epoch}.ckpt"
            checkpoint_epoch = latest_epoch
        
        # Create policy model
        policy = HookedAttentionModelPolicy(
            env_name=self.env.name,
            embed_dim=self.config['embed_dim'],
            num_encoder_layers=self.config['n_encoder_layers'],
            num_heads=int(self.config.get("num_heads", 8)),
            temperature=self.config['temperature'],
        )
        
        # Load checkpoint
        model = REINFORCEClipped.load_from_checkpoint(
            checkpoint_path,
            env=self.env,
            policy=policy,
            strict=False
        )
        model = model.to(self.device)
        model.eval()
        
        print(f"Loaded model from checkpoint epoch {checkpoint_epoch}")
        return model
    
    def generate_instances(self, num_instances=1):
        """
        Generate new TSP instances.
        
        Args:
            num_instances: Number of instances to generate
            
        Returns:
            Generated instances
        """
        instances = self.env.reset(batch_size=[num_instances]).to(self.device)
        return instances
    
    def decode_solution(self, instance, greedy=True):
        """
        Decode a solution for the given instance.
        
        Args:
            instance: TSP instance to solve
            greedy: Whether to use greedy decoding (True) or sampling (False)
            
        Returns:
            Solution dict containing actions, reward, etc.
        """
        self.model.eval()
        
        with torch.no_grad():
            decode_type = "greedy" if greedy else "sampling"
            solution = self.model.policy(instance, decode_type=decode_type, return_actions=True)
        
        return solution
    
    def visualize_solution(self, instance, solution=None, instance_idx=0, ax=None, 
                          save_path=None, plot_indices=False, highlight_nodes=None):
        """
        Visualize a TSP instance and its solution.
        
        Args:
            instance: TSP instance
            solution: Solution dict (if None, will be generated)
            instance_idx: Instance index in the batch
            ax: Matplotlib axis (optional)
            save_path: Path to save the figure (optional)
            plot_indices: Whether to plot node indices (default: False)
            highlight_nodes: List of node indices to highlight
            
        Returns:
            Matplotlib axis
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 10))
        
        # If no solution provided, generate one
        if solution is None:
            solution = self.decode_solution(instance)
        
        # Extract node coordinates for this instance
        locs = instance['locs'][instance_idx].cpu().numpy()
        
        # Get the actions (solution)
        actions = solution['actions'][instance_idx].cpu().numpy()
        
        # Extract the tour (node ordering)
        tour = np.concatenate((actions, [actions[0]]))  # Connect back to start
        
        # Plot nodes
        if highlight_nodes is not None:
            # Use default colors for most nodes
            ax.scatter(locs[:, 0], locs[:, 1], s=80, color='blue', zorder=2)
            
            # Highlight specific nodes
            highlight_locs = locs[highlight_nodes]
            ax.scatter(highlight_locs[:, 0], highlight_locs[:, 1], s=120, color='red', 
                      edgecolors='black', linewidths=2, zorder=3)
        else:
            # All nodes same color
            ax.scatter(locs[:, 0], locs[:, 1], s=80, color='blue', zorder=2)
        
        # Plot indices next to nodes if requested
        if plot_indices:
            for i, (x, y) in enumerate(locs):
                ax.text(x, y, str(i), fontsize=8, ha='center', va='center', 
                       color='white', fontweight='bold', zorder=4)
        
        # Plot the tour using arrows to show direction
        tour_points = locs[tour]
        
        # Using quiver for arrows
        dx, dy = np.diff(tour_points[:, 0]), np.diff(tour_points[:, 1])
        ax.quiver(
            tour_points[:-1, 0], tour_points[:-1, 1], 
            dx, dy, 
            scale_units='xy', angles='xy', scale=1, 
            color='black', alpha=0.7, width=0.005, zorder=1
        )
        
        # Add title and labels
        reward = solution['reward'][instance_idx].item()
        # Negative because reward is negative of distance
        distance = -reward
        
        ax.set_title(f'TSP Solution (Distance: {distance:.4f})')
        ax.set_xlabel('X Coordinate')
        ax.set_ylabel('Y Coordinate')
        
        # Set equal aspect ratio and limits
        ax.set_aspect('equal')
        ax.set_xlim(-0.05, 1.05)
        ax.set_ylim(-0.05, 1.05)
        
        # Save the figure if a path is provided
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Saved visualization to {save_path}")
        
        return ax
    
    def visualize_solutions_batch(self, num_instances=5, output_dir=None, greedy=True):
        """
        Generate and visualize solutions for multiple instances.
        
        Args:
            num_instances: Number of instances to generate
            output_dir: Directory to save visualizations
            greedy: Whether to use greedy decoding
            
        Returns:
            List of solution visualizations
        """
        # Generate instances
        instances = self.generate_instances(num_instances)
        
        # Decode solutions
        solutions = self.decode_solution(instances, greedy=greedy)
        
        # Create output directory if needed
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        
        # Visualize each solution
        figures = []
        for i in range(num_instances):
            fig, ax = plt.subplots(figsize=(10, 10))
            
            if output_dir:
                save_path = os.path.join(output_dir, f"solution_{i}.png")
            else:
                save_path = None
            
            self.visualize_solution(
                instances, 
                solution=solutions,
                instance_idx=i,
                ax=ax,
                save_path=save_path
            )
            
            figures.append((fig, ax))
            
            if save_path is None:
                plt.show()
            
            # Close the figure if saved
            if save_path:
                plt.close(fig)
        
        return figures
    
    def visualize_activations(self, instance, instance_idx=0, output_dir=None):
        """
        Visualize model's encoder activations for a given instance.
        
        Args:
            instance: Instance to visualize
            instance_idx: Index of instance in batch
            output_dir: Output directory for saving visualizations
            
        Returns:
            Dictionary of activation visualizations
        """
        self.model.eval()
        policy = self.model.policy
        
        # Run forward pass to collect activations
        with torch.no_grad():
            # Clear existing activations
            policy.clear_cache()
            
            # Run forward pass with the instance
            solution = policy(instance, decode_type="greedy", return_actions=True)
            
            # Get activation cache
            activations = policy.activation_cache
        
        # Create output directory if needed
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        
        # Process layer activations
        layer_imgs = {}
        for layer_name, activation in activations.items():
            if not layer_name.startswith('encoder_layer_'):
                continue
                
            # Get activations for this layer
            # Shape: [batch_size, num_nodes, embed_dim]
            layer_acts = activation
            
            # Compute the norm of each node's embedding to get a scalar value
            act_norms = torch.norm(layer_acts, dim=2)[instance_idx].cpu().numpy()
            
            # Create visualization
            fig, ax = plt.subplots(figsize=(10, 10))
            
            # Extract node coordinates for this instance
            locs = instance['locs'][instance_idx].cpu().numpy()
            
            # Plot nodes with colors based on activation norms
            scatter = ax.scatter(
                locs[:, 0], 
                locs[:, 1], 
                c=act_norms,
                cmap='viridis',
                s=100,
                alpha=0.8,
                edgecolors='black'
            )
            
            # Add colorbar
            plt.colorbar(scatter, ax=ax, label=f'{layer_name} Activation Magnitude')
            
            # Add title and labels
            ax.set_title(f'TSP Node Activations - {layer_name}')
            ax.set_xlabel('X Coordinate')
            ax.set_ylabel('Y Coordinate')
            
            # Set equal aspect ratio and limits
            ax.set_aspect('equal')
            ax.set_xlim(-0.05, 1.05)
            ax.set_ylim(-0.05, 1.05)
            
            # Save if output directory provided
            if output_dir:
                save_path = os.path.join(output_dir, f"{layer_name}_activations.png")
                plt.savefig(save_path, dpi=150, bbox_inches='tight')
                print(f"Saved {layer_name} visualization to {save_path}")
                plt.close(fig)
            else:
                plt.show()
            
            layer_imgs[layer_name] = fig
        
        # Also visualize the solution for reference
        solution_fig, ax = plt.subplots(figsize=(10, 10))
        
        if output_dir:
            save_path = os.path.join(output_dir, "solution.png")
        else:
            save_path = None
            
        self.visualize_solution(
            instance, 
            solution=solution,
            instance_idx=instance_idx,
            ax=ax,
            save_path=save_path
        )
        
        if output_dir:
            plt.close(solution_fig)
        
        # Return the figures
        return layer_imgs
    
    def visualize_attention(self, instance, layer_idx=0, head_idx=0, instance_idx=0, output_dir=None):
        """
        Visualize attention maps for a specific layer and head.
        
        NOTE: This requires modifying the HookedAttentionModelPolicy to capture attention weights.
        This is a placeholder for future extension.
        
        Args:
            instance: Instance to visualize
            layer_idx: Index of attention layer
            head_idx: Index of attention head
            instance_idx: Index of instance in batch
            output_dir: Output directory for saving visualizations
            
        Returns:
            Attention visualization figure
        """
        # This is a placeholder - to implement this fully, you would need to:
        # 1. Modify the HookedAttentionModelPolicy to capture attention weights
        # 2. Register hooks on the MultiHeadAttention layers 
        # 3. Extract and visualize the attention weights
        
        print("Attention visualization requires modifying the HookedAttentionModelPolicy to capture attention weights.")
        print("This functionality needs to be implemented in the policy class first.")
        
        # For illustration, just visualize the solution
        solution_fig, ax = plt.subplots(figsize=(10, 10))
        solution = self.decode_solution(instance)
        
        if output_dir:
            save_path = os.path.join(output_dir, "solution.png")
        else:
            save_path = None
            
        self.visualize_solution(
            instance, 
            solution=solution,
            instance_idx=instance_idx,
            ax=ax,
            save_path=save_path
        )
        
        if output_dir and save_path:
            plt.close(solution_fig)
            
        return solution_fig


def main():
    parser = argparse.ArgumentParser(description="Visualize TSP model solutions")
    
    parser.add_argument("--run_path", type=str, required=True,
                       help="Path to the main TSP run directory")
    parser.add_argument("--checkpoint_epoch", type=int, default=None,
                       help="Specific epoch to load (if None, loads latest)")
    parser.add_argument("--num_instances", type=int, default=5,
                       help="Number of instances to generate and visualize")
    parser.add_argument("--output_dir", type=str, default=None,
                       help="Directory to save visualizations")
    parser.add_argument("--device", type=str, default=None,
                       help="Device to run on (default: auto-select)")
    parser.add_argument("--show_activations", action="store_true",
                       help="Visualize encoder layer activations")
    parser.add_argument("--sampling", action="store_true",
                       help="Use sampling instead of greedy decoding")
    
    args = parser.parse_args()
    
    # Create visualizer
    visualizer = TSPModelVisualizer(
        run_path=args.run_path,
        checkpoint_epoch=args.checkpoint_epoch,
        device=args.device
    )
    
    # Handle activation visualization
    if args.show_activations:
        # Generate a single instance for activation visualization
        instance = visualizer.generate_instances(1)
        
        # Create output subdirectory for activations
        if args.output_dir:
            activation_dir = os.path.join(args.output_dir, "activations")
        else:
            activation_dir = None
            
        visualizer.visualize_activations(instance, output_dir=activation_dir)
    else:
        # Visualize solutions
        visualizer.visualize_solutions_batch(
            num_instances=args.num_instances,
            output_dir=args.output_dir,
            greedy=not args.sampling
        )


if __name__ == "__main__":
    main()
