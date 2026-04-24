import torch
from torch_geometric.data import Data
import subprocess
import tempfile
import os


def load_aig_as_graph(path, cache_dir=None):
    """
    Converts AIG → PyG graph with caching support
    Features: [const, input, and, output, fanout]
    """

    # ----------- CACHE (VERY IMPORTANT) -----------
    if cache_dir is not None:
        os.makedirs(cache_dir, exist_ok=True)
        cache_path = os.path.join(
            cache_dir,
            os.path.basename(path).replace(".aig", ".pt")
        )
        if os.path.exists(cache_path):
            return torch.load(cache_path)

    with tempfile.NamedTemporaryFile(suffix=".aag", delete=False) as tmp:
        aag_path = tmp.name

    try:
        subprocess.run(
            ['utils/aiger/aigtoaig', path, aag_path],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
        )

        with open(aag_path, 'r') as f:
            lines = [
                line.strip()
                for line in f
                if line.strip() and not line.startswith('c')
            ]

        header = lines[0].split()
        M, I, L, O, A = map(int, header[1:6])

        # ----------- NODE MAPPING -----------
        node_map = {}
        current_idx = 0

        def get_node(v):
            nonlocal current_idx
            if v not in node_map:
                node_map[v] = current_idx
                current_idx += 1
            return node_map[v]

        edges = []
        edge_attrs = []
        fanout = {}

        line_idx = 1

        const_node = get_node(0)

        # ----------- INPUTS -----------
        for _ in range(I):
            lit = int(lines[line_idx])
            get_node(lit // 2)
            line_idx += 1

        # skip latches
        line_idx += L

        # ----------- OUTPUTS -----------
        outputs = []
        for _ in range(O):
            lit = int(lines[line_idx])
            outputs.append(lit)
            get_node(lit // 2)
            line_idx += 1

        # ----------- AND GATES -----------
        and_nodes = []
        for _ in range(A):
            lhs, r1, r2 = map(int, lines[line_idx].split())

            lhs_n = get_node(lhs // 2)
            r1_n = get_node(r1 // 2)
            r2_n = get_node(r2 // 2)

            and_nodes.append(lhs_n)

            edges.append([r1_n, lhs_n])
            edge_attrs.append([1.0 if r1 % 2 else 0.0])

            edges.append([r2_n, lhs_n])
            edge_attrs.append([1.0 if r2 % 2 else 0.0])

            fanout[r1_n] = fanout.get(r1_n, 0) + 1
            fanout[r2_n] = fanout.get(r2_n, 0) + 1

            line_idx += 1

        num_nodes = current_idx

        # ----------- NODE FEATURES -----------
        x = torch.zeros((num_nodes, 5), dtype=torch.float)

        # const
        x[const_node][0] = 1.0

        # inputs
        line_idx = 1
        for _ in range(I):
            lit = int(lines[line_idx])
            x[get_node(lit // 2)][1] = 1.0
            line_idx += 1

        # outputs
        for lit in outputs:
            x[get_node(lit // 2)][3] = 1.0

        # AND
        for n in and_nodes:
            x[n][2] = 1.0

        # fanout (normalized)
        max_fanout = max(fanout.values()) if fanout else 1.0
        for n in range(num_nodes):
            x[n][4] = fanout.get(n, 0) / max_fanout

        # ----------- EDGES -----------
        if edges:
            edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
            edge_attr = torch.tensor(edge_attrs, dtype=torch.float)
        else:
            edge_index = torch.empty((2, 0), dtype=torch.long)
            edge_attr = torch.empty((0, 1), dtype=torch.float)

        data = Data(
            x=x,
            edge_index=edge_index,
            edge_attr=edge_attr
        )

        # ----------- SAVE CACHE -----------
        if cache_dir is not None:
            torch.save(data, cache_path)

        return data

    finally:
        if os.path.exists(aag_path):
            os.remove(aag_path)