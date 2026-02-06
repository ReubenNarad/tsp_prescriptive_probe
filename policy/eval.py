import matplotlib.pyplot as plt
import os, argparse, pickle
import imageio.v2 as imageio
import numpy as np


def main(args):
    run_path = f'runs/{args.run_name}'
    n_epochs = args.num_epochs
    plot_instance = 1

    env = pickle.load(open(f'{run_path}/env.pkl', 'rb'))
    instance = pickle.load(open(f'{run_path}/val_td.pkl', 'rb'))

    # 1) Load the solved baseline
    with open(f'{run_path}/baseline.pkl', 'rb') as f:
        baseline = pickle.load(f)

    # Render the baseline (optimal) solution
    baseline_dist_for_instance = -baseline["rewards"][0][plot_instance].item()
    fig = env.render(instance[plot_instance], baseline["actions"][0][plot_instance])
    plt.title(f"Optimal Distance: {baseline_dist_for_instance:.2f}")
    plt.savefig(f'{run_path}/optimal_solution_{plot_instance}.png')
    plt.close()

    # 2) Load the training results for each epoch and render solutions
    os.makedirs(f'{run_path}/renders', exist_ok=True)
    results = []
    step = 2
    
    for i in range(0, n_epochs, step):
        # Load results for this epoch
        with open(f'{run_path}/results/results_epoch_{i}.pkl', 'rb') as f:
            td_epoch = pickle.load(f)
        results.append(td_epoch)
        
        # Compute distance for this epoch (recall: distance = -reward)
        distance = -td_epoch["rewards"][plot_instance].item()

        # Render solution for this epoch
        fig = env.render(instance[plot_instance], td_epoch["actions"][plot_instance])
        plt.title(f"Epoch {i} — Distance: {distance:.2f}")
        plt.savefig(f'{run_path}/renders/epoch_{i}_{plot_instance}.png')
        plt.close()

    # 2.5) Create a single GIF from the saved PNG files using imageio
    gif_path = f"{run_path}/animation_{plot_instance}.gif"
    # Read all images first
    images = []
    for i in range(0, n_epochs, step):
        filename = f'{run_path}/renders/epoch_{i}_{plot_instance}.png'
        images.append(imageio.imread(filename))

    # Write GIF with fps parameter
    imageio.mimsave(gif_path, images, fps=args.fps)
    print(f"GIF saved at {gif_path}")

    # 3) Get the baseline's average distance
    baseline_distance = -baseline["rewards"][0].mean().item()

    # 4) Compute the average distance for each epoch
    epoch_distances = []
    for i in range(0, n_epochs, step):
        with open(f'{run_path}/results/results_epoch_{i}.pkl', 'rb') as f:
            td_epoch = pickle.load(f)
        avg_dist = -td_epoch["rewards"][0].mean().item() if isinstance(td_epoch["rewards"], list) else -td_epoch["rewards"].mean().item()
        epoch_distances.append(avg_dist)

    # 5) Plot training progress (without exponential decay curve)
    plt.figure(figsize=(8, 5))
    
    # Use all available data points
    epochs = np.array(range(0, n_epochs, step))
    distances = np.array(epoch_distances)
    
    print(f"Current best distance: {min(epoch_distances):.3f}")
    
    plt.plot(epochs, epoch_distances, marker="o", markersize=2, label="Avg Distance per Epoch")
    plt.axhline(y=baseline_distance, color="red", linestyle="--", label="Baseline (Optimal)")
    plt.xlabel("Epoch")
    plt.ylabel("Distance")
    plt.ylim(baseline_distance / 1.1, max(epoch_distances))
    plt.yscale('log')
    plt.title(f"Average Solution Distance on Validation Set")
    plt.legend()
    plt.savefig(f'{run_path}/train_plot.png')
    plt.close()

    # 6) Create edge overlap plot
    plt.figure(figsize=(8, 5))
    
    # Convert baseline route to edges
    baseline_route = baseline["actions"][0][plot_instance]
    optimal_edges = set(
        tuple(sorted((int(baseline_route[i].item()), int(baseline_route[(i+1) % len(baseline_route)].item()))))
        for i in range(len(baseline_route))
    )
    
    edge_overlaps = []
    
    # Calculate edge overlap for each epoch
    for i in range(0, n_epochs, step):
        with open(f'{run_path}/results/results_epoch_{i}.pkl', 'rb') as f:
            td_epoch = pickle.load(f)
            
        # Convert route to edges
        current_route = td_epoch["actions"][plot_instance]
        current_edges = set(
            tuple(sorted((int(current_route[i].item()), int(current_route[(i+1) % len(current_route)].item()))))
            for i in range(len(current_route))
        )
        
        overlap = len(optimal_edges.intersection(current_edges)) / len(optimal_edges)
        edge_overlaps.append(overlap)
    
    plt.plot(range(0, n_epochs, step), edge_overlaps, marker="o", markersize=2, 
             color='purple', label="Edge Overlap with Optimal")
    plt.xlabel("Epoch")
    plt.ylabel("Edge Overlap Ratio")
    plt.ylim(0, 1)
    plt.title(f"Edge Overlap with Optimal Solution (every {step} epochs)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.savefig(f'{run_path}/edge_overlap_plot.png')
    plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_name", type=str, default=None)
    parser.add_argument("--num_epochs", type=int, default=100)
    parser.add_argument("--fps", type=int, default=20)
    args = parser.parse_args()

    main(args)


# Optionally, remove the individual PNGs to keep things clean
# import glob
# for png_file in glob.glob(f"{run_path}/renders/*.png"):
#     if os.path.basename(png_file).startswith("epoch_"):
#         os.remove(png_file)

