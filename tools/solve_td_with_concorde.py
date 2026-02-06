import contextlib
import io
import torch
import subprocess, os, pickle, argparse, re
import tempfile
from pathlib import Path

@contextlib.contextmanager
def _torch_unpickle_on_cpu():
    """
    Ensure tensors inside pickled objects load onto CPU even if they were saved from CUDA.
    This repo uses `pickle.dump` on objects containing tensors, which can fail on CPU-only
    machines unless we force a CPU map_location during unpickling.
    """
    orig = torch.storage._load_from_bytes

    def _load_from_bytes_cpu(b: bytes):
        return torch.load(io.BytesIO(b), map_location=torch.device("cpu"), weights_only=False)

    torch.storage._load_from_bytes = _load_from_bytes_cpu
    try:
        yield
    finally:
        torch.storage._load_from_bytes = orig

def write_tsplib_file(filename, coords):
    """
    Write a single TSP instance to file in TSPLIB format.

    Args:
        filename (str): Path to file to be written.
        coords (Tensor): Float tensor of shape [num_locs, 2].
                         Each row is (x, y) in [0,1] or some range.
    """
    with open(filename, 'w') as f:
        f.write("NAME: td_tsp\n")
        f.write("TYPE: TSP\n")
        f.write(f"DIMENSION: {coords.shape[0]}\n")
        f.write("EDGE_WEIGHT_TYPE: EUC_2D\n")
        f.write("NODE_COORD_SECTION\n")
        for idx, (x, y) in enumerate(coords):
            # TSPLIB node indices start at 1
            # Scale coordinates by 100 for concorde stability (should not affect solution)
            f.write(f"{idx + 1} {float(x*100):.4f} {float(y*100):.4f}\n")
        f.write("EOF\n")


def run_concorde(instance_filename, solution_filename, *, cwd: str | None = None):
    """
    Run Concorde on the given TSP instance file and save solution to solution_filename.
    Returns the number of branch-and-bound nodes used.

    Args:
        instance_filename (str): Path to .tsp file in TSPLIB format.
        solution_filename (str): Desired output .sol or .txt file from Concorde.

    Returns:
        int: Number of branch-and-bound nodes, or -1 if not found.
    """
    base_path = os.path.splitext(os.path.basename(instance_filename))[0]
    bb_nodes = -1 # Default value if not found
    try:
        # Capture stdout to parse node count
        result = subprocess.run(
            ['concorde', '-o', solution_filename, instance_filename],
            check=True,
            text=True,
            stdout=subprocess.PIPE, # Capture standard output
            stderr=subprocess.PIPE,
            cwd=cwd,
        )

        # Print the stdout for debugging
        print(f"Concorde stdout for {instance_filename}:")
        print(result.stdout)

        # Use the specific regex for "Number of bbnodes:"
        match = re.search(r"Number of bbnodes:\s*(\d+)", result.stdout)
        if match:
            bb_nodes = int(match.group(1))
        else:
            # Also check stderr if not found in stdout
             match_err = re.search(r"Number of bbnodes:\s*(\d+)", result.stderr)
             if match_err:
                 bb_nodes = int(match_err.group(1))

    except subprocess.CalledProcessError as e:
        print(f"Error executing Concorde for {instance_filename}:")
        print(e.stderr) # Keep printing the error message itself

        # Check stderr for node count as well
        match = re.search(r"Number of bbnodes:\s*(\d+)", e.stderr)
        if match:
            bb_nodes = int(match.group(1))
        # raise # Optionally re-raise the exception
    return bb_nodes


def read_solution_file(filename):
    """
    Concorde's solution file expects lines with city indices (1-based)
    separated by whitespace, ending with -1. This function reads them
    and returns a route in 0-based indexing.

    Args:
        filename (str): Path to the concorde solution file.

    Returns:
        tour (list): The route as a list of node indices (0-based).
    """
    tour = []
    with open(filename, 'r') as f:
        lines = f.readlines()
    for i, line in enumerate(lines):
        # The first line is often a second copy of dimension,
        # so we skip it or parse carefully
        if i == 0:
            # Could parse dimension from here if needed, but skip
            continue
        # Each line can contain multiple nodes, separated by space, ending with -1
        # e.g. "1 3 4 -1"
        numbers = [int(x) for x in line.strip().split()]
        # Exclude the trailing -1 (if present)
        for num in numbers:
            if num == -1:
                break
            tour.append(num)
    return tour


def compute_tour_length(coords, route):
    """
    Compute the total Euclidean tour length for a route that visits
    all nodes in 'route' order and returns to the start.

    coords: [num_l locs, 2]  FloatTensor
    route: [num_locs]  1D long tensor with the visiting order (0-based).

    Returns:
        Scalar float for total distance of the route (closed tour).
    """
    # Gather the coordinates in the visiting order
    route_coords = coords[route]

    # Compute pairwise distances, including the wrap from last to first
    # roll(-1) shifts everything by one to line up neighbors
    dist_vec = (route_coords - route_coords.roll(-1, dims=0)).pow(2).sum(-1).sqrt()
    return dist_vec.sum()


def main(args):
    run_path = f'runs/{args.run_name}'
    input_td = os.path.join(run_path, 'val_td.pkl')
    output_td = os.path.join(run_path, 'baseline.pkl')

    with _torch_unpickle_on_cpu():
        with open(input_td, 'rb') as f:
            td = pickle.load(f)
    
    locs = td["locs"]  # shape: [batch_size, num_locs, 2]
    batch_size = locs.shape[0]
    
    all_solutions = []
    all_rewards = []  # We'll store -tour_length here, consistent with how TSPEnv does it
    all_bb_nodes = [] # Store branch-and-bound node counts

    print(f"Solving {batch_size} instances...")
    run_path_p = Path(run_path)
    with tempfile.TemporaryDirectory(prefix="concorde_", dir=str(run_path_p)) as tmp:
        scratch = Path(tmp)

        for i in range(batch_size):
            coords_i = locs[i]  # [num_locs, 2]
            tsp_filename = scratch / f"instance_{i}.tsp"
            sol_filename = scratch / f"solution_{i}.sol"

            write_tsplib_file(str(tsp_filename), coords_i)

            # Run Concorde in the scratch dir so it cannot litter the repo.
            bb_nodes = run_concorde(tsp_filename.name, sol_filename.name, cwd=str(scratch))

            # Read the solution route
            route_0based = read_solution_file(str(sol_filename))

            # Convert route to a torch tensor
            route_t = torch.tensor(route_0based, dtype=torch.long)

            # Compute the TSP reward like in TSPEnv (reward = -tour_length)
            tour_length = compute_tour_length(coords_i, route_t)
            reward = -tour_length.clone().detach()

            all_solutions.append(route_t)
            all_rewards.append(reward)
            all_bb_nodes.append(bb_nodes) # Append the node count

            # Clean up Concorde artifacts as we go (defensive; tempdir will be removed anyway).
            for base_stub in (f"instance_{i}", f"solution_{i}"):
                for ext in (".pul", ".sav", ".res", ".mas", ".pix", ".sol", ".tsp"):
                    for prefix in ("", "O"):
                        p = scratch / f"{prefix}{base_stub}{ext}"
                        try:
                            if p.exists():
                                p.unlink()
                        except Exception:
                            pass

    # Convert to tensors
    optimal_routes = torch.stack(all_solutions, dim=0)      # [batch_size, num_locs]
    optimal_rewards = torch.stack(all_rewards, dim=0)       # [batch_size]
    optimal_bb_nodes = torch.tensor(all_bb_nodes, dtype=torch.long) # [batch_size]

    # Store similarly to how training code stores actions/rewards
    results = {
        'actions': [optimal_routes],  # same shape as your model's output
        'rewards': [optimal_rewards], # match the style: a list with one tensor
        'bb_nodes': [optimal_bb_nodes] # Add the node counts
    }

    avg_nodes = optimal_bb_nodes[optimal_bb_nodes != -1].float().mean().item() if (optimal_bb_nodes != -1).any() else 'N/A'
    print(f"Optimally solved {batch_size} instances with avg distance {- optimal_rewards.mean().item():.4f}, avg B&B nodes: {avg_nodes}")

    # 5) Save the results
    with open(output_td, 'wb') as f:
        pickle.dump(results, f)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_name", type=str, default=None)
    args = parser.parse_args()
    main(args)
