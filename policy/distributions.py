from torch import tensor, zeros, randperm, rand
import torch
from torch.distributions import Uniform
import math
import warnings

class Uniform:
    """Samples TSP instances with node coordinates uniformly distributed from (0,0) to (1,1)
    """
    def __init__(self, num_loc=20):
        self.num_loc = num_loc

    def sample(self, size):
        return torch.rand(size[0], self.num_loc, 2)


class RandomUniform:
    """Samples TSP instances where node coordinates are uniformly distributed
    within a rectangle. For each instance:
    - the bottom-left (min_x, min_y) corner is sampled from [0, 0.5] x [0, 0.5]
    - the top-right (max_x, max_y) corner is sampled from [0.5, 1] x [0.5, 1]
    These corners define the sampling rectangle for that instance's nodes.
    """
    def __init__(self, num_loc=20, min_coord_range=(0.0, 0.5), max_coord_range=(0.5, 1.0)):
        self.num_loc = num_loc
        # Define the fixed ranges for sampling corners
        self.min_coord_range = min_coord_range
        self.max_coord_range = max_coord_range

    def sample(self, batch_size_arg):
        """Samples batch_size instances using the defined corner sampling logic.

        Returns:
            locs: Tensor of shape [batch_size, num_loc, 2]
        """
        # Determine the integer batch size from the input argument
        if isinstance(batch_size_arg, (list, tuple)):
            current_batch_size = batch_size_arg[0]
        elif torch.is_tensor(batch_size_arg):
            current_batch_size = batch_size_arg[0].item()
        elif isinstance(batch_size_arg, int):
            current_batch_size = batch_size_arg
        else:
            raise TypeError(f"Unsupported batch_size_arg type: {type(batch_size_arg)}")

        # Determine device
        if torch.is_tensor(batch_size_arg):
             device = batch_size_arg.device
        else:
             # Default device if batch_size_arg is int or list
             device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Default ranges if attributes don't exist (for backward compatibility)
        min_coord_range = getattr(self, 'min_coord_range', (0.0, 0.5))
        max_coord_range = getattr(self, 'max_coord_range', (0.5, 1.0))

        # Sample bottom-left corners (min_x, min_y) from [0, 0.5] x [0, 0.5]
        # Shape: [batch_size, 2]
        min_coords = torch.rand(current_batch_size, 2, device=device) * (min_coord_range[1] - min_coord_range[0]) + min_coord_range[0]

        # Sample top-right corners (max_x, max_y) from [0.5, 1] x [0.5, 1]
        # Shape: [batch_size, 2]
        max_coords = torch.rand(current_batch_size, 2, device=device) * (max_coord_range[1] - max_coord_range[0]) + max_coord_range[0]

        # Expand dims for broadcasting with num_loc
        # Shape: [batch_size, 1, 2]
        min_coords = min_coords.unsqueeze(1)
        max_coords = max_coords.unsqueeze(1)

        # Calculate width and height for each instance's box
        # Shape: [batch_size, 1, 2] -> [batch_size, 1, width/height]
        box_dims = max_coords - min_coords
        # Prevent division by zero or negative dimensions (shouldn't happen with these ranges)
        box_dims = torch.clamp(box_dims, min=1e-6)

        # Sample points uniformly within [0, 1) relative to the box dimensions
        # Shape: [batch_size, num_loc, 2]
        relative_coords = torch.rand(current_batch_size, self.num_loc, 2, device=device)

        # Scale relative coordinates by box dimensions and shift by min coordinates
        locs = relative_coords * box_dims + min_coords

        return locs


class DualUniform:
    """
    Samples 50/50 from two uniform distributions defined by their corner coordinates.
    
    Args:
        corner1_min: (x_min, y_min) for first distribution
        corner1_max: (x_max, y_max) for first distribution
        corner2_min: (x_min, y_min) for second distribution
        corner2_max: (x_max, y_max) for second distribution
    """
    def __init__(self, corner1_min=(0.0, 0.0), corner1_max=(0.4, 0.4),
                 corner2_min=(0.6, 0.6), corner2_max=(1.0, 1.0)):
        super().__init__()
        self.corner1_min = tensor(corner1_min)
        self.corner1_max = tensor(corner1_max)
        self.corner2_min = tensor(corner2_min)
        self.corner2_max = tensor(corner2_max)

    def sample(self, size):
        batch_size, num_loc, _ = size
        
        # Create empty tensor for coordinates
        coords = zeros(batch_size, num_loc, 2)
        
        # For each batch
        for i in range(batch_size):
            # Sample half the points from first distribution
            n_first = num_loc // 2
            coords[i, :n_first] = rand(n_first, 2) * (self.corner1_max - self.corner1_min) + self.corner1_min
            
            # Sample remaining points from second distribution
            coords[i, n_first:] = rand(num_loc - n_first, 2) * (self.corner2_max - self.corner2_min) + self.corner2_min
            
            # Shuffle the points
            idx = randperm(num_loc)
            coords[i] = coords[i][idx]

        return coords

class NRandomUniform:
    """
    Draws a random number (1-5) -> n uniform distributions, then draws their boundaries from a uniform distribution.
    Each distribution is elongated/skinny with a random orientation.
    """
    def __init__(self, min_loc=0.0, max_loc=1.0, length_factor=5.0, min_clusters=1, max_clusters=5):
        super().__init__()
        self.min_loc = min_loc
        self.max_loc = max_loc
        self.length_factor = length_factor
        self.min_clusters = min_clusters
        self.max_clusters = max_clusters

    def sample(self, size):
        batch_size, num_loc, _ = size
        
        # Get the device from the input size tensor if it's a tensor, otherwise default to CPU
        device = size.device if torch.is_tensor(size) else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Create tensors directly on the correct device
        coords = torch.zeros(batch_size, num_loc, 2, device=device)
        
        # Generate all random numbers at once on the device
        num_clusters = torch.randint(self.min_clusters, self.max_clusters + 1, (batch_size,), device=device)
        max_points_per_cluster = (num_loc + num_clusters.max().item() - 1) // num_clusters.max().item()
        
        # Pre-generate all random numbers we'll need
        centers = torch.rand(batch_size, num_clusters, 2, device=device)
        lengths = torch.rand(batch_size, num_clusters, device=device) * 0.7 + 0.4  # Length between 0.4 and 1.1
        widths = lengths / self.length_factor  # Much smaller width
        
        point_idx = 0
        for i in range(batch_size):
            points_per_cluster = num_loc // num_clusters
            
            for c in range(num_clusters):
                cluster_points = points_per_cluster if c < num_clusters - 1 else (num_loc - point_idx)
                
                # Generate points in a rectangle - no rotation
                x_offsets = (torch.rand(cluster_points, device=device) - 0.5) * lengths[i, c]
                y_offsets = (torch.rand(cluster_points, device=device) - 0.5) * widths[i, c]
                
                coords[i, point_idx:point_idx + cluster_points, 0] = centers[i, c, 0] + x_offsets
                coords[i, point_idx:point_idx + cluster_points, 1] = centers[i, c, 1] + y_offsets
                
                point_idx += cluster_points
            point_idx = 0  # Reset for next batch
            
        return coords * (self.max_loc - self.min_loc) + self.min_loc


class FuzzyCircle:
    """
    Samples points in a circular pattern with normally distributed radial noise.
    For each batch, the radius mean and standard deviation are themselves
    sampled from uniform distributions.
    
    Args:
        radius_mean_lower: Lower bound for sampling the mean radius
        radius_mean_upper: Upper bound for sampling the mean radius
        radius_std_lower: Lower bound for sampling the radius standard deviation
        radius_std_upper: Upper bound for sampling the radius standard deviation
        center: (x, y) coordinates of circle center, defaults to (0.5, 0.5)
        random_center: If True, ignores the center parameter and generates a random
                     center for each batch within the unit square [0,1]×[0,1]
        center_x_range: (min_x, max_x) range for random center x-coordinate
        center_y_range: (min_y, max_y) range for random center y-coordinate
    """
    def __init__(self, radius_mean_lower=0.3, radius_mean_upper=0.5, 
                radius_std_lower=0.02, radius_std_upper=0.05, center=(0.5, 0.5),
                random_center=False, center_x_range=(0.2, 0.8), center_y_range=(0.2, 0.8)):
        super().__init__()
        self.radius_mean_lower = radius_mean_lower
        self.radius_mean_upper = radius_mean_upper
        self.radius_std_lower = radius_std_lower
        self.radius_std_upper = radius_std_upper
        self.center = torch.tensor(center)
        self.random_center = random_center
        self.center_x_range = center_x_range
        self.center_y_range = center_y_range

    def sample(self, size):
        batch_size, num_loc, _ = size
        
        # Get the device from the input size tensor if it's a tensor, otherwise default to CPU
        device = size.device if torch.is_tensor(size) else torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        random_center = getattr(self, "random_center", False)
        center_tensor = getattr(self, "center", torch.tensor((0.5, 0.5)))
        center_tensor = torch.as_tensor(center_tensor, dtype=torch.float, device=device)
        center_x_range = getattr(self, "center_x_range", (0.2, 0.8))
        center_y_range = getattr(self, "center_y_range", (0.2, 0.8))
        
        # Create empty tensor for coordinates
        coords = torch.zeros(batch_size, num_loc, 2, device=device)
        
        for i in range(batch_size):
            # Sample the radius mean and std for this batch
            radius_mean = torch.empty(1, device=device).uniform_(self.radius_mean_lower, self.radius_mean_upper)
            radius_std = torch.empty(1, device=device).uniform_(self.radius_std_lower, self.radius_std_upper)
            
            # Sample a random center if specified
            if random_center:
                center_x = torch.empty(1, device=device).uniform_(center_x_range[0], center_x_range[1])
                center_y = torch.empty(1, device=device).uniform_(center_y_range[0], center_y_range[1])
                center = torch.tensor([center_x.item(), center_y.item()], device=device)
            else:
                center = center_tensor
            
            # Sample uniform angles
            angles = 2 * math.pi * torch.rand(num_loc, device=device)
            
            # Using scalar values from the tensors
            radius_mean_val = radius_mean.item()
            radius_std_val = radius_std.item()
            distances = torch.normal(
                mean=radius_mean_val, 
                std=radius_std_val, 
                size=(num_loc,), 
                device=device
            )
            
            # Convert to cartesian coordinates
            coords[i, :, 0] = distances * torch.cos(angles) + center[0]
            coords[i, :, 1] = distances * torch.sin(angles) + center[1]
            
        return coords


class Clusters:
    """
    Samples from a random number of gaussians, each with random mean and cov.
    Creates new clusters for each sample() call with tight, compact clusters.
    Allows points to extend beyond the [0,1] range.
    """
    def __init__(self, min_clusters=8, max_clusters=15):
        super().__init__()
        self.min_clusters = min_clusters
        self.max_clusters = max_clusters

    def sample(self, size):
        batch_size, num_loc, _ = size
        
        # Get device from the input size if it's a tensor
        device = size.device if torch.is_tensor(size) else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Create empty tensor for coordinates
        coords = torch.zeros(batch_size, num_loc, 2, device=device)
        
        # For each batch, create new clusters
        for i in range(batch_size):
            # Generate new clusters for this batch - more clusters now
            n_clusters = torch.randint(self.min_clusters, self.max_clusters + 1, (1,)).item()
            clusters = []
            
            for j in range(n_clusters):
                # Define the cluster's mean (center)
                mean = 0.05 + 0.95 * torch.rand(2, device=device)
                
                # Create a proper random covariance matrix with random orientation
                # Generate random standard deviations for x and y directions
                # Smaller values for tighter clusters
                std_x = (0.001 + 0.0005 * torch.rand(1, device=device)).sqrt()
                std_y = (0.001 + 0.0005 * torch.rand(1, device=device)).sqrt()
                
                # Generate random rotation angle (0 to 2π)
                theta = 2 * math.pi * torch.rand(1, device=device)
                
                # Create rotation matrix
                c, s = torch.cos(theta), torch.sin(theta)
                rotation = torch.tensor([[c, -s], [s, c]], device=device).squeeze()
                
                # Create diagonal matrix with variance
                scales = torch.diag(torch.tensor([std_x**2, std_y**2], device=device).squeeze())
                
                # Create covariance matrix: R * S * R^T
                cov = rotation @ scales @ rotation.T
                
                clusters.append((mean, cov))
            
            # Calculate how many points to sample from each cluster
            points_per_cluster = torch.full((n_clusters,), num_loc // n_clusters, device=device)
            # Distribute remaining points
            remainder = num_loc % n_clusters
            if remainder > 0:
                points_per_cluster[:remainder] += 1
            
            point_idx = 0
            # Sample from each cluster
            for c, (mean, cov) in enumerate(clusters):
                n_points = points_per_cluster[c].item()
                
                # Generate samples from multivariate normal distribution
                z = torch.randn(n_points, 2, device=device)
                
                # We need the Cholesky decomposition of the covariance matrix
                try:
                    L = torch.linalg.cholesky(cov)
                    # Transform: x = μ + Lz
                    samples = torch.matmul(z, L.T) + mean
                except:
                    # Fallback to simpler approach if cholesky fails
                    std = torch.sqrt(torch.diag(cov))
                    samples = torch.randn(n_points, 2, device=device) * std + mean
                
                coords[i, point_idx:point_idx + n_points] = samples
                point_idx += n_points
            
            # Shuffle the points for this batch
            idx = torch.randperm(num_loc, device=device)
            coords[i] = coords[i][idx]
        
        # Removed the clamping to allow points outside [0,1]
        return coords

class SimpleClusters:
    """
    A faster version of Clusters that samples from axis-aligned Gaussians.
    Each cluster has a random mean and random x/y standard deviations,
    but no rotation. Generates new clusters for each sample() call.
    Allows points to extend beyond the [0,1] range.
    """
    def __init__(self, min_clusters=8, max_clusters=15, std_dev_range=(0.01, 0.05)):
        super().__init__()
        self.min_clusters = min_clusters
        self.max_clusters = max_clusters
        # Use a range for standard deviation directly
        self.std_dev_min = std_dev_range[0]
        self.std_dev_max = std_dev_range[1]

    def sample(self, size):
        batch_size, num_loc, _ = size
        
        # Get device from the input size if it's a tensor
        device = size.device if torch.is_tensor(size) else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Create empty tensor for coordinates
        coords = torch.zeros(batch_size, num_loc, 2, device=device)
        
        # For each batch, create new clusters
        for i in range(batch_size):
            # Generate new clusters for this batch
            n_clusters = torch.randint(self.min_clusters, self.max_clusters + 1, (1,)).item()
            cluster_params = []
            
            for _ in range(n_clusters):
                # Define the cluster's mean (center)
                mean = 0.05 + 0.90 * torch.rand(2, device=device) # Slightly reduced range
                
                # Sample standard deviations directly for x and y
                std_x = torch.empty(1, device=device).uniform_(self.std_dev_min, self.std_dev_max)
                std_y = torch.empty(1, device=device).uniform_(self.std_dev_min, self.std_dev_max)
                
                cluster_params.append((mean, std_x, std_y))
            
            # Calculate how many points to sample from each cluster
            points_per_cluster = torch.full((n_clusters,), num_loc // n_clusters, dtype=torch.long, device=device)
            # Distribute remaining points
            remainder = num_loc % n_clusters
            if remainder > 0:
                points_per_cluster[:remainder] += 1
            
            point_idx = 0
            # Sample from each cluster
            for c, (mean, std_x, std_y) in enumerate(cluster_params):
                n_points = points_per_cluster[c].item()
                
                # Generate samples from axis-aligned normal distribution
                # Sample standard normal variables
                z = torch.randn(n_points, 2, device=device)
                
                # Scale and shift: x = μ + σ*z
                samples = torch.zeros_like(z)
                samples[:, 0] = mean[0] + std_x * z[:, 0]
                samples[:, 1] = mean[1] + std_y * z[:, 1]
                
                coords[i, point_idx:point_idx + n_points] = samples
                point_idx += n_points
            
            # Shuffle the points for this batch
            idx = torch.randperm(num_loc, device=device)
            coords[i] = coords[i][idx]
        
        return coords

class HybridSampler:
    """
    A hybrid sampler that draws from multiple distributions with equal or specified probabilities.
    
    Args:
        distributions: List of distribution objects that implement a sample method
        probabilities: Optional list of probabilities corresponding to each distribution.
                      If None, samples evenly from all distributions.
    """
    def __init__(
        self, 
        distributions=None,
        probabilities=None
    ):
        super().__init__()
        
        # Set default distributions if none provided
        if distributions is None:
            distributions = [
                Uniform()
            ]
        
        self.distributions = distributions
        
        # Validate and normalize probabilities if provided
        if probabilities is not None:
            if len(probabilities) != len(distributions):
                raise ValueError("Length of probabilities must match length of distributions")
            
            # Convert to tensor and normalize
            self.probabilities = torch.tensor(probabilities, dtype=torch.float)
            self.probabilities = self.probabilities / self.probabilities.sum()
        else:
            # Equal probabilities for all distributions
            n_dists = len(distributions)
            self.probabilities = torch.ones(n_dists) / n_dists

    def sample(self, size):
        batch_size, num_loc, _ = size
        
        # Get the device from the input size tensor
        device = size.device if torch.is_tensor(size) else torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Some legacy checkpoints stored sub-distributions as dist1, dist2, ...
        if (not hasattr(self, "distributions")) or self.distributions is None or len(self.distributions) == 0:
            recovered = []
            recovered_names = []
            for name, value in sorted(self.__dict__.items()):
                if name.startswith("dist") and hasattr(value, "sample"):
                    recovered.append(value)
                    recovered_names.append(name)
            if recovered:
                warnings.warn(
                    f"HybridSampler recovered embedded distributions: {', '.join(recovered_names)}",
                    RuntimeWarning,
                )
                self.distributions = recovered
            else:
                raise AttributeError("HybridSampler has no distributions defined or recoverable. "
                                     "Please instantiate with explicit distributions.")

        # Ensure we have probabilities matching the number of distributions
        if not hasattr(self, "probabilities") or self.probabilities is None:
            self.probabilities = torch.ones(len(self.distributions), dtype=torch.float)
        probabilities = torch.as_tensor(self.probabilities, dtype=torch.float)
        if probabilities.numel() != len(self.distributions):
            probabilities = torch.ones(len(self.distributions), dtype=torch.float)
        probabilities = probabilities / probabilities.sum()
        self.probabilities = probabilities
        
        # Move probabilities to the same device
        probabilities = probabilities.to(device)
        
        # Create tensor to store results
        coords = torch.zeros(batch_size, num_loc, 2, device=device)
        
        # For each instance in the batch, decide which distribution to use
        for i in range(batch_size):
            # Sample distribution index based on probabilities
            dist_idx = torch.multinomial(probabilities, 1).item()
            
            # Sample from the selected distribution
            coords[i] = self.distributions[dist_idx].sample((1, num_loc, 2))[0]
        
        return coords


# Demonstration with quick plot
if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import seaborn as sns
    sns.set_theme(style="whitegrid")

    # Initialize the sampler
    # sampler = Uniform(num_loc=100)
    # sampler = Clusters(min_clusters=10, max_clusters=15)
    sampler = FuzzyCircle()

    # Sample a batch of 1 instance
    coords = sampler.sample([1, 100, 2])  # 1 instance, 100 points per instance, 2D

    # Create a single plot
    fig, ax = plt.subplots(1, 1, figsize=(8, 8))

    # Plot the single instance
    coords_np = coords[0].cpu().numpy()
    ax.scatter(coords_np[:, 0], coords_np[:, 1], alpha=.7, s=50, color="#6f42c1")
    # Set fixed plot limits from -0.2 to 1.2
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    plt.tight_layout()
    plt.savefig("fuzzy_circle_distribution.png")
