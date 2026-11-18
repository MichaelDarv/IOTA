
import tkinter as tk
import random

from network3dvisualizer import Network3DVisualizer


def make_test_network(generation):
    """Generate a synthetic network snapshot."""

    nodes = [
        {
            "id": 0,
            "type": "input",
            "layer": 0,
            "bias": 0.0,
            "response": 1.0,
            "activation": "identity",
            "aggregation": "sum",
            "useful": True,
        },
        {
            "id": 1,
            "type": "input",
            "layer": 0,
            "bias": 0.0,
            "response": 1.0,
            "activation": "identity",
            "aggregation": "sum",
            "useful": True,
        },
        {
            "id": 2,
            "type": "hidden",
            "layer": 1,
            "bias": random.uniform(-1, 1),
            "response": 1.0,
            "activation": "sigmoid",
            "aggregation": "sum",
            "useful": True,
        },
        {
            "id": 3,
            "type": "hidden",
            "layer": 1,
            "bias": random.uniform(-1, 1),
            "response": 1.0,
            "activation": "tanh",
            "aggregation": "sum",
            "useful": True,
        },
        {
            "id": 4,
            "type": "output",
            "layer": 2,
            "bias": 0.1,
            "response": 1.0,
            "activation": "sigmoid",
            "aggregation": "sum",
            "useful": True,
        },
    ]

    connections = [
        {
            "source": 0,
            "target": 2,
            "weight": random.uniform(-2, 2),
        },
        {
            "source": 1,
            "target": 2,
            "weight": random.uniform(-2, 2),
        },
        {
            "source": 0,
            "target": 3,
            "weight": random.uniform(-2, 2),
        },
        {
            "source": 2,
            "target": 4,
            "weight": random.uniform(-2, 2),
        },
        {
            "source": 3,
            "target": 4,
            "weight": random.uniform(-2, 2),
        },
    ]

    # Introduce a new neuron after generation 5.
    if generation >= 5:
        nodes.append({
            "id": 5,
            "type": "hidden",
            "layer": 1,
            "bias": 0.25,
            "response": 1.0,
            "activation": "relu",
            "aggregation": "sum",
            "useful": True,
        })

        connections.extend([
            {
                "source": 1,
                "target": 5,
                "weight": 1.4,
            },
            {
                "source": 5,
                "target": 4,
                "weight": -0.8,
            },
        ])

    layers = [
        [0, 1],
        [2, 3] if generation < 5 else [2, 3, 5],
        [4],
    ]

    return {
        "nodes": nodes,
        "connections": connections,
        "topo_order": [0, 1, 2, 3, 4],
        "topo_layers": layers,
        "useful_nodes": [0, 1, 2, 3, 4],
        "node_count": len(nodes),
        "connection_count": len(connections),
    }


def main():
    root = tk.Tk()
    root.withdraw()

    visualizer = Network3DVisualizer(root)

    generation = [0]

    def update():
        generation[0] += 1

        snapshot = make_test_network(generation[0])

        visualizer.add_history(
            snapshot,
            generation[0],
            random.uniform(0, 1),
        )

        root.after(1000, update)

    update()
    root.mainloop()


if __name__ == "__main__":
    main()
