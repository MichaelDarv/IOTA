
import jax
import numpy as np
import jax.numpy as jnp


def extract_network_snapshot(network, input_ids, output_ids):
    input_ids = set(map(int, input_ids))
    output_ids = set(map(int, output_ids))

    nodes = []

    # Map neuron IDs to their topological layer.
    layer_lookup = {}

    for layer_index, layer in enumerate(network["topo_layers"]):
        for node_id in layer:
            layer_lookup[int(node_id)] = layer_index

    useful_nodes = set(map(int, network["useful_nodes"]))

    for node_id, attrs in network["nodes"].items():
        node_id = int(node_id)

        if node_id in input_ids:
            role = "input"
        elif node_id in output_ids:
            role = "output"
        else:
            role = "hidden"

        nodes.append({
            "id": node_id,
            "type": role,
            "layer": layer_lookup.get(node_id, 0),
            "useful": node_id in useful_nodes,
            "bias": float(np.asarray(jax.device_get(attrs["bias"]))),
            "response": float(np.asarray(jax.device_get(attrs["res"]))),
            "activation": str(attrs["act"]),
            "aggregation": str(attrs["agg"]),
        })

    connections = []

    for (source, target), attrs in network["conns"].items():
        connections.append({
            "source": int(source),
            "target": int(target),
            "weight": float(
                np.asarray(jax.device_get(attrs["weight"]))
            ),
        })

    return {
        "nodes": nodes,
        "connections": connections,
        "topo_order": list(map(int, network["topo_order"])),
        "topo_layers": [
            list(map(int, layer))
            for layer in network["topo_layers"]
        ],
        "useful_nodes": list(useful_nodes),
        "node_count": len(nodes),
        "connection_count": len(connections),
    }