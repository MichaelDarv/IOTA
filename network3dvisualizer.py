import tkinter as tk
from tkinter import ttk
from collections import defaultdict

import queue
import threading

import numpy as np

from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import (
    FigureCanvasTkAgg,
    NavigationToolbar2Tk,
)
from mpl_toolkits.mplot3d.art3d import Line3DCollection
from mpl_toolkits.mplot3d import proj3d
from matplotlib.colors import to_rgba


class Network3DVisualizer:
    """
    Interactive 3D TensorNEAT network visualizer.

    Input neurons are fixed to one plane.
    Output neurons are fixed to the opposite plane.
    Hidden neurons are force-directed, then uniformly scaled
    to fit between those planes.

    Layout calculations run in a background worker.
    Matplotlib and Tkinter operations remain on the GUI thread.
    """

    NODE_COLOURS = {
        "input": "#3498db",
        "hidden": "#2ecc71",
        "output": "#f1c40f",
    }

    # ---------------------------------------------------------
    # Layout configuration
    # ---------------------------------------------------------

    INPUT_X = -50.0
    OUTPUT_X = 50.0

    # Hidden neurons fill a sphere centred between the planes.
    # The sphere radius is half the plane gap minus this margin.
    HIDDEN_MARGIN = 5.0

    # Input/output disc radius as a fraction of the hidden sphere radius.
    PLANE_RADIUS_FRACTION = 0.7

    # Hidden neurons never sit closer to the centre than this
    # fraction of the sphere radius, which keeps the middle open.
    INNER_RADIUS_FRACTION = 0.3

    # Values above 1 bias neuron directions toward the YZ plane,
    # so the cloud spreads sideways instead of along X.
    YZ_WEIGHT = 1.6


    # Edge appearance
    EDGE_WIDTH_BASE = 0.4
    EDGE_WIDTH_PER_WEIGHT = 0.3     # extra width per unit |weight| (capped at 5)
    EDGE_ALPHA_MIN = 0.05           # weakest edge, nothing selected
    EDGE_ALPHA_MAX = 0.35           # strongest edge, nothing selected
    EDGE_ALPHA_DIMMED = 0.03        # edges outside the selected chain
    EDGE_ALPHA_LIT = 0.9            # edges inside the selected chain
    EDGE_LIT_WIDTH_BOOST = 1.6
    NODE_ALPHA_DIMMED = 0.2         # nodes outside the selected chain

    PICK_RADIUS_PIXELS = 18


    # Force simulation configuration.
    LAYOUT_ITERATIONS = 70
    IDEAL_DISTANCE_BASE = 2.0

    # Rendering thresholds. Labels are expensive in 3D Matplotlib.
    LABEL_NODE_LIMIT = 60
    NODE_SIZE = 90
    SELECTED_NODE_SIZE = 180

    def __init__(self, parent, title="Network 3D Visualizer"):
        self.parent = parent

        self.window = tk.Toplevel(parent)
        self.window.title(title)
        self.window.geometry("1150x850")
        self.window.minsize(800, 600)

        self.snapshot = None
        self.positions = {}

        self.previous_node_ids = set()
        self.previous_connections = set()

        self.selected_node = None
        self.selected_connection = None

        self.history = []
        self.history_index = -1
        self.live_mode = True

        self.base_view_limit = 1.0
        self.current_view_limit = self.base_view_limit
        self.zoom_out_margin = 0.85
        self.zoom_out_factor = 1.2

        self._closed = False

        self._build_ui()

        # -----------------------------------------------------
        # Background layout processing
        # -----------------------------------------------------

        self.layout_requests = queue.Queue(maxsize=1)
        self.layout_results = queue.Queue(maxsize=1)

        self.layout_request_id = 0

        self.layout_thread = threading.Thread(
            target=self._layout_worker,
            daemon=True,
        )

        self.layout_thread.start()

        self.window.protocol(
            "WM_DELETE_WINDOW",
            self._on_close,
        )

        self.window.after(30, self._poll_layout_results)

    # ---------------------------------------------------------
    # UI
    # ---------------------------------------------------------

    def _build_ui(self):
        self.window.columnconfigure(0, weight=1)
        self.window.rowconfigure(1, weight=1)

        header = ttk.Frame(self.window, padding=8)
        header.grid(row=0, column=0, sticky="ew")

        self.title_label = ttk.Label(
            header,
            text="No network loaded",
            font=("Segoe UI", 12, "bold")
        )
        self.title_label.pack(side="left")

        self.mode_var = tk.StringVar(value="Current champion")

        ttk.Combobox(
            header,
            textvariable=self.mode_var,
            values=["Current champion", "All-time champion"],
            state="readonly",
            width=20,
        ).pack(side="right")

        content = ttk.Frame(self.window)
        content.grid(row=1, column=0, sticky="nsew", padx=8, pady=4)

        content.columnconfigure(0, weight=4)
        content.columnconfigure(1, weight=1)
        content.rowconfigure(0, weight=1)

        # 3D plotting area
        plot_frame = ttk.Frame(content)
        plot_frame.grid(row=0, column=0, sticky="nsew")

        self.figure = Figure(
            figsize=(8, 7),
            dpi=100,
            facecolor="#151923",
        )

        self.ax = self.figure.add_subplot(
            111,
            projection="3d",
        )

        self._style_axes()

        self.canvas = FigureCanvasTkAgg(
            self.figure,
            master=plot_frame,
        )

        self.canvas.get_tk_widget().pack(
            fill="both",
            expand=True,
        )

        toolbar = NavigationToolbar2Tk(
            self.canvas,
            plot_frame,
            pack_toolbar=False,
        )
        toolbar.update()
        toolbar.pack(side="bottom", fill="x")

        # Information panel
        info = ttk.LabelFrame(
            content,
            text="Inspection",
            padding=10,
        )
        info.grid(
            row=0,
            column=1,
            sticky="nsew",
            padx=(8, 0),
        )

        self.info_text = tk.Text(
            info,
            width=28,
            wrap="word",
            state="disabled",
            font=("Consolas", 9),
        )
        self.info_text.pack(fill="both", expand=True)

        # History controls
        footer = ttk.Frame(self.window, padding=8)
        footer.grid(row=2, column=0, sticky="ew")

        self.history_label = ttk.Label(
            footer,
            text="Generation: --",
        )
        self.history_label.pack(side="left")

        self.history_slider = ttk.Scale(
            footer,
            from_=0,
            to=0,
            orient="horizontal",
            command=self._on_history_change,
        )
        self.history_slider.pack(
            side="left",
            fill="x",
            expand=True,
            padx=12,
        )

        self.live_button = ttk.Button(
            footer,
            text="Return to Live",
            command=self.return_to_live,
        )
        self.live_button.pack(side="right")

        self.canvas.mpl_connect("button_press_event", self._on_plot_press)
        self.canvas.mpl_connect("button_release_event", self._on_plot_click)

    # ---------------------------------------------------------
    # Window management
    # ---------------------------------------------------------

    def _on_close(self):
        self._closed = True

        try:
            self.window.destroy()
        except tk.TclError:
            pass

    # ---------------------------------------------------------
    # Plot styling
    # ---------------------------------------------------------

    def _style_axes(self):
        bg = "#151923"
        self.ax.set_facecolor(bg)

        # Dark, nearly invisible panes instead of the default grey.
        for axis in (self.ax.xaxis, self.ax.yaxis, self.ax.zaxis):
            axis.set_pane_color((0.09, 0.11, 0.16, 1.0))
            axis._axinfo["grid"]["color"] = (1, 1, 1, 0.06)

        self.ax.tick_params(colors="#666e7d", labelsize=7)

        self.ax.set_xlabel("Input  →  Output", color="#9aa3b2")
        self.ax.set_ylabel("Y", color="#9aa3b2")
        self.ax.set_zlabel("Z", color="#9aa3b2")

    def _chain_for(self, node_id, connections):
        """Return (nodes, edges) reachable upstream and downstream of node_id."""
        downstream = defaultdict(list)
        upstream = defaultdict(list)

        for edge in connections:
            s, t = int(edge["source"]), int(edge["target"])
            downstream[s].append(t)
            upstream[t].append(s)

        chain_nodes = {node_id}
        chain_edges = set()

        for adjacency, forward in ((downstream, True), (upstream, False)):
            seen = {node_id}
            stack = [node_id]
            while stack:
                current = stack.pop()
                for nxt in adjacency[current]:
                    chain_edges.add((current, nxt) if forward else (nxt, current))
                    if nxt not in seen:
                        seen.add(nxt)
                        stack.append(nxt)
            chain_nodes |= seen

        return chain_nodes, chain_edges

    # ---------------------------------------------------------
    # Hidden cluster scaling
    # ---------------------------------------------------------

    def _fit_hidden_region(self, positions, node_lookup=None):
        """
        Spread the hidden neurons through a sphere centred between the
        input and output planes.

        Each neuron keeps its direction from the cluster centre, but its
        distance from the centre is set by rank, so the neurons fill the
        volume evenly instead of clumping around the middle.
        """

        if not positions:
            return positions

        hidden_ids = list(positions.keys())
        count = len(hidden_ids)

        centre_x = (self.INPUT_X + self.OUTPUT_X) / 2.0

        sphere_radius = (
                (self.OUTPUT_X - self.INPUT_X) / 2.0
                - self.HIDDEN_MARGIN
        )

        if count == 1:
            return {hidden_ids[0]: np.array([centre_x, 0.0, 0.0])}

        coords = np.asarray(
            [positions[node_id] for node_id in hidden_ids],
            dtype=float,
        )

        coords = coords - coords.mean(axis=0)

        # Standardise each axis. An axis with no spread gets a
        # deterministic per-neuron value so it cannot collapse the cloud.
        std = coords.std(axis=0)
        for axis in range(3):
            if std[axis] > 1e-4:
                coords[:, axis] /= std[axis]
            else:
                coords[:, axis] = [
                    np.sin(int(node_id) * (12.9898 + 4.1 * axis))
                    for node_id in hidden_ids
                ]

        # Bias directions toward YZ.
        coords[:, 1:] *= self.YZ_WEIGHT

        # Unit direction of each neuron from the centre.
        radii = np.linalg.norm(coords, axis=1)
        directions = np.zeros_like(coords)

        valid = radii > 1e-9
        directions[valid] = coords[valid] / radii[valid, np.newaxis]

        # Neurons sitting exactly at the centre get a deterministic direction.
        for i in np.where(~valid)[0]:
            angle = int(hidden_ids[i]) * 2.399963
            directions[i] = (0.0, np.cos(angle), np.sin(angle))

        # Radius by rank. The cube root gives equal volume per neuron,
        # so the neurons are uniformly distributed through the sphere.
        ranks = np.argsort(np.argsort(radii))
        fraction = (ranks + 1) / count

        inner = self.INNER_RADIUS_FRACTION
        final_radii = sphere_radius * (
                inner + (1.0 - inner) * fraction ** (1.0 / 3.0)
        )

        coords = directions * final_radii[:, np.newaxis]
        coords[:, 0] += centre_x

        return {
            node_id: coords[i]
            for i, node_id in enumerate(hidden_ids)
        }

    def _fix_input_output_planes(self, positions, node_lookup):
        """
        Force input and output neurons onto their respective
        planes after force simulation.
        """

        for node_id, node in node_lookup.items():

            if node_id not in positions:
                continue

            x, y, z = positions[node_id]

            if node["type"] == "input":
                positions[node_id] = (
                    self.INPUT_X,
                    y,
                    z,
                )

            elif node["type"] == "output":
                positions[node_id] = (
                    self.OUTPUT_X,
                    y,
                    z,
                )

        return positions

    # ---------------------------------------------------------
    # Position generation
    # 3D force-directed positioning
    # ---------------------------------------------------------

    def _calculate_positions(self, snapshot, old_positions=None):
        """
        Calculate a 3D force-directed layout.

        Neurons attract connected neurons.
        Neurons repel other neurons.
        Existing positions provide temporal stability.

        Input/output planes and hidden-cluster scaling are
        applied after the force simulation.
        """

        nodes = snapshot["nodes"]
        connections = snapshot["connections"]

        if not nodes:
            return {}

        node_ids = [
            int(node["id"])
            for node in nodes
        ]

        node_lookup = {
            int(node["id"]): node
            for node in nodes
        }

        index = {
            node_id: i
            for i, node_id in enumerate(node_ids)
        }

        count = len(node_ids)

        # Do not read GUI state from the worker thread.
        old_positions = dict(old_positions or {})

        # -----------------------------------------------------
        # Initialise positions
        # -----------------------------------------------------

        positions = np.zeros((count, 3), dtype=float)

        layer_lookup = {}

        for layer_index, layer in enumerate(
            snapshot.get("topo_layers", [])
        ):
            for node_id in layer:
                layer_lookup[int(node_id)] = layer_index

        # Build adjacency for initial placement.
        neighbours = {
            node_id: []
            for node_id in node_ids
        }

        for edge in connections:
            source = int(edge["source"])
            target = int(edge["target"])

            if source in neighbours and target in neighbours:
                neighbours[source].append(target)
                neighbours[target].append(source)

        for i, node_id in enumerate(node_ids):

            # Preserve existing neuron coordinates.
            if node_id in old_positions:
                positions[i] = old_positions[node_id]
                continue

            node = node_lookup[node_id]

            # Place new neurons near their existing neighbours.
            known_neighbours = [
                old_positions[n]
                for n in neighbours[node_id]
                if n in old_positions
            ]

            # Deterministic 3D offset based on neuron ID.
            seed = node_id * 12.9898

            jitter = np.array([
                np.sin(seed),
                np.cos(seed * 1.37),
                np.sin(seed * 0.73),
            ]) * 0.8

            if known_neighbours:
                centre = np.mean(
                    known_neighbours,
                    axis=0,
                )

                positions[i] = centre + jitter

            else:
                layer = layer_lookup.get(node_id, 0)

                positions[i] = np.array([
                    (layer - 1) * 2.0,
                    0.0,
                    0.0,
                ]) + jitter

        # -----------------------------------------------------
        # Build connection index arrays
        # -----------------------------------------------------

        edge_sources = []
        edge_targets = []

        for edge in connections:
            source = int(edge["source"])
            target = int(edge["target"])

            if source not in index or target not in index:
                continue

            edge_sources.append(index[source])
            edge_targets.append(index[target])

        edge_sources = np.asarray(
            edge_sources,
            dtype=int,
        )

        edge_targets = np.asarray(
            edge_targets,
            dtype=int,
        )

        # -----------------------------------------------------
        # Identify input/output nodes
        # -----------------------------------------------------

        input_indices = []
        output_indices = []

        for i, node_id in enumerate(node_ids):
            role = node_lookup[node_id]["type"]

            if role == "input":
                input_indices.append(i)

            elif role == "output":
                output_indices.append(i)

        # -----------------------------------------------------
        # Temporal stability
        # -----------------------------------------------------

        old_anchor = np.zeros((count, 3), dtype=float)
        has_old_position = np.zeros(count, dtype=bool)

        for i, node_id in enumerate(node_ids):
            if node_id in old_positions:
                old_anchor[i] = old_positions[node_id]
                has_old_position[i] = True

        # -----------------------------------------------------
        # Force-directed simulation
        # -----------------------------------------------------

        ideal_distance = max(
            self.IDEAL_DISTANCE_BASE,
            (count / 8.0) ** (1.0 / 3.0) * 2.0,
        )

        # Force repulsion is O(N^2). Reduce the number of simulation
        # passes as the topology grows to keep updates responsive.
        if count <= 100:
            iterations = self.LAYOUT_ITERATIONS
        elif count <= 300:
            iterations = 45
        elif count <= 750:
            iterations = 25
        else:
            iterations = 15

        for iteration in range(iterations):

            forces = np.zeros_like(positions)

            # ---------------------------------------------
            # 1. Repulsion between every pair of neurons
            # ---------------------------------------------

            delta = (
                    positions[:, np.newaxis, :]
                    - positions[np.newaxis, :, :]
            )

            distance_squared = np.sum(
                delta * delta,
                axis=2,
            )

            # Find different neurons occupying identical positions.
            coincident = distance_squared < 1e-8
            np.fill_diagonal(coincident, False)

            # Give coincident neurons a small deterministic separation.
            for i, j in zip(*np.where(np.triu(coincident, k=1))):
                direction = np.array([
                    1.0,
                    (j - i) * 0.37,
                    (i + j + 1) * 0.19,
                ], dtype=float)

                direction /= np.linalg.norm(direction)

                delta[i, j] = direction * 0.001
                delta[j, i] = -direction * 0.001

            # Recalculate distances using the adjusted displacement vectors.
            distance_squared = np.sum(
                delta * delta,
                axis=2,
            )

            # Prevent self-repulsion and any remaining division by zero.
            np.fill_diagonal(distance_squared, np.inf)
            distance_squared = np.maximum(distance_squared, 1e-8)

            repulsion = (
                    ideal_distance ** 2
                    / distance_squared
            )

            repulsion_force = (
                    delta * repulsion[:, :, np.newaxis]
            )

            forces += np.sum(
                repulsion_force,
                axis=1,
            )

            # ---------------------------------------------
            # 2. Attraction between connected neurons
            # ---------------------------------------------

            if len(edge_sources):
                edge_delta = (
                    positions[edge_targets]
                    - positions[edge_sources]
                )

                edge_distance = np.linalg.norm(
                    edge_delta,
                    axis=1,
                )

                edge_distance = np.maximum(
                    edge_distance,
                    0.001,
                )

                attraction = (
                    edge_delta
                    * (
                        edge_distance / ideal_distance
                    )[:, np.newaxis]
                )

                np.add.at(
                    forces,
                    edge_sources,
                    attraction,
                )

                np.add.at(
                    forces,
                    edge_targets,
                    -attraction,
                )

            # ---------------------------------------------
            # 3. Temporal stability
            # ---------------------------------------------

            stability = 0.12

            forces[has_old_position] += (
                old_anchor[has_old_position]
                - positions[has_old_position]
            ) * stability

            # ---------------------------------------------
            # 4. Weak input/output spatial organisation
            # ---------------------------------------------

            # These are only soft forces during simulation.
            # Exact plane positions are enforced afterwards.

            if input_indices:
                forces[input_indices, 0] += (
                    self.INPUT_X
                    - positions[input_indices, 0]
                ) * 0.06

            if output_indices:
                forces[output_indices, 0] += (
                    self.OUTPUT_X
                    - positions[output_indices, 0]
                ) * 0.06

            # ---------------------------------------------
            # 5. Apply movement with cooling
            # ---------------------------------------------

            progress = iteration / iterations

            temperature = (
                0.12 * (1.0 - progress)
                + 0.015
            )

            force_length = np.linalg.norm(
                forces,
                axis=1,
                keepdims=True,
            )

            force_length = np.maximum(
                force_length,
                0.001,
            )

            forces = forces / force_length

            positions += forces * temperature

        # -----------------------------------------------------
        # Separate neurons by role
        # -----------------------------------------------------

        input_ids = []
        hidden_ids = []
        output_ids = []

        for node_id, node in node_lookup.items():

            if node["type"] == "input":
                input_ids.append(node_id)

            elif node["type"] == "output":
                output_ids.append(node_id)

            else:
                hidden_ids.append(node_id)

        # -----------------------------------------------------
        # Extract and fit hidden positions first
        # -----------------------------------------------------

        hidden_positions = {
            node_id: positions[index[node_id]].copy()
            for node_id in hidden_ids
        }

        hidden_positions = self._fit_hidden_region(
            hidden_positions
        )

        # -----------------------------------------------------
        # Input/output plane radius
        # -----------------------------------------------------

        # Planes are discs sized relative to the hidden sphere,
        # rather than copying the (previously tiny) hidden YZ spread.
        hidden_sphere_radius = (
                (self.OUTPUT_X - self.INPUT_X) / 2.0
                - self.HIDDEN_MARGIN
        )

        plane_radius = max(
            3.0,
            hidden_sphere_radius * self.PLANE_RADIUS_FRACTION,
        )

        input_positions = self._distribute_on_plane(
            input_ids,
            self.INPUT_X,
            radius=plane_radius,
        )

        output_positions = self._distribute_on_plane(
            output_ids,
            self.OUTPUT_X,
            radius=plane_radius,
        )

        # -----------------------------------------------------
        # Combine all positions
        # -----------------------------------------------------

        result = {}

        result.update(input_positions)
        result.update(hidden_positions)
        result.update(output_positions)

        return {
            node_id: tuple(float(v) for v in position)
            for node_id, position in result.items()
        }

    def _distribute_on_plane(self, node_ids, x_position, radius=3.0):
        """Distribute neurons evenly across a circular YZ plane.

        A golden-angle spiral gives a more even area distribution than
        placing every neuron around the circumference. The last point
        reaches the requested radius.
        """

        count = len(node_ids)

        if count == 0:
            return {}

        positions = {}
        golden_angle = np.pi * (3.0 - np.sqrt(5.0))

        for i, node_id in enumerate(node_ids):
            if count == 1:
                y = 0.0
                z = 0.0
            else:
                radial_distance = radius * np.sqrt((i + 1) / count)
                angle = golden_angle * i
                y = radial_distance * np.cos(angle)
                z = radial_distance * np.sin(angle)

            positions[node_id] = (
                float(x_position),
                float(y),
                float(z),
            )

        return positions

    # ---------------------------------------------------------
    # Asynchronous layout worker
    # ---------------------------------------------------------

    def _replace_queue_item(self, target_queue, item):
        """Keep only the newest pending item."""

        while True:
            try:
                target_queue.put_nowait(item)
                return

            except queue.Full:
                try:
                    target_queue.get_nowait()
                except queue.Empty:
                    pass

    def _layout_worker(self):
        """Calculate network positions away from the GUI thread."""

        while True:
            request = self.layout_requests.get()

            if request is None:
                break

            (
                request_id,
                snapshot,
                old_positions,
                generation,
                fitness,
            ) = request

            try:
                positions = self._calculate_positions(
                    snapshot,
                    old_positions,
                )

                result = (
                    request_id,
                    snapshot,
                    positions,
                    generation,
                    fitness,
                    None,
                )

            except Exception as exc:
                result = (
                    request_id,
                    snapshot,
                    None,
                    generation,
                    fitness,
                    exc,
                )

            self._replace_queue_item(
                self.layout_results,
                result,
            )

    def _poll_layout_results(self):
        """Receive completed layouts on the Tkinter main thread."""

        if self._closed:
            return

        try:
            while True:
                result = self.layout_results.get_nowait()

                (
                    request_id,
                    snapshot,
                    positions,
                    generation,
                    fitness,
                    error,
                ) = result

                # Ignore outdated calculations.
                if request_id != self.layout_request_id:
                    continue

                if error is not None:
                    print(f"Network layout error: {error}")
                    continue

                self._render_network(
                    snapshot,
                    positions,
                    generation,
                    fitness,
                )

        except queue.Empty:
            pass

        if not self._closed:
            try:
                self.window.after(
                    30,
                    self._poll_layout_results,
                )
            except tk.TclError:
                self._closed = True

    # ---------------------------------------------------------
    # Rendering
    # ---------------------------------------------------------

    def _render_network(
        self,
        snapshot,
        positions,
        generation=None,
        fitness=None,
    ):
        """Render a completed layout on the Tkinter main thread."""

        if self._closed:
            return

        self.snapshot = snapshot
        self.positions = positions

        new_node_ids = {
            int(node["id"])
            for node in snapshot["nodes"]
        }

        new_connections = {
            (int(edge["source"]), int(edge["target"]))
            for edge in snapshot["connections"]
        }

        added_nodes = new_node_ids - self.previous_node_ids

        removed_nodes = (
            self.previous_node_ids - new_node_ids
        )

        added_connections = (
            new_connections - self.previous_connections
        )

        # Preserve the user's current camera and zoom across generation redraws.
        # ax.clear() resets Matplotlib's limits and camera, so capture them first.
        had_existing_view = getattr(self, "_view_initialized", False)

        if had_existing_view:
            saved_xlim = self.ax.get_xlim()
            saved_ylim = self.ax.get_ylim()
            saved_zlim = self.ax.get_zlim()
            saved_elev = self.ax.elev
            saved_azim = self.ax.azim
            saved_roll = getattr(self.ax, "roll", 0)

        self.ax.clear()
        self._style_axes()

        # ---- Selection chain -------------------------------------------
        chain_nodes = chain_edges = None
        if self.selected_node in new_node_ids:
            chain_nodes, chain_edges = self._chain_for(
                self.selected_node, snapshot["connections"]
            )

        # ---- Edges: dim ones first so lit ones draw on top --------------
        dim_segs, dim_cols, dim_widths = [], [], []
        lit_segs, lit_cols, lit_widths = [], [], []

        for edge in snapshot["connections"]:
            source = int(edge["source"])
            target = int(edge["target"])

            if source not in self.positions or target not in self.positions:
                continue

            segment = [self.positions[source], self.positions[target]]
            weight = float(edge["weight"])
            magnitude = min(abs(weight), 5.0)
            strength = magnitude / 5.0
            colour = "#4aa3ff" if weight >= 0 else "#ff5c5c"
            width = self.EDGE_WIDTH_BASE + magnitude * self.EDGE_WIDTH_PER_WEIGHT

            if chain_edges is None:
                alpha = self.EDGE_ALPHA_MIN + strength * (
                        self.EDGE_ALPHA_MAX - self.EDGE_ALPHA_MIN
                )
                dim_segs.append(segment)
                dim_cols.append(to_rgba(colour, alpha))
                dim_widths.append(width)

            elif (source, target) in chain_edges:
                lit_segs.append(segment)
                lit_cols.append(
                    to_rgba(colour, 0.5 + 0.4 * strength)
                )
                lit_widths.append(width * self.EDGE_LIT_WIDTH_BOOST)

            else:
                dim_segs.append(segment)
                dim_cols.append(to_rgba(colour, self.EDGE_ALPHA_DIMMED))
                dim_widths.append(self.EDGE_WIDTH_BASE)

        for segs, cols, widths in (
                (dim_segs, dim_cols, dim_widths),
                (lit_segs, lit_cols, lit_widths),
        ):
            if segs:
                self.ax.add_collection3d(
                    Line3DCollection(segs, colors=cols, linewidths=widths)
                )

        # ---- Nodes ------------------------------------------------------
        nodes_by_role = defaultdict(list)
        for node in snapshot["nodes"]:
            node_id = int(node["id"])
            if node_id in self.positions:
                nodes_by_role[node["type"]].append(
                    (node_id, self.positions[node_id])
                )

        for role, role_nodes in nodes_by_role.items():
            coords = np.asarray([p for _, p in role_nodes], dtype=float)

            colours, edge_colours, sizes = [], [], []
            for node_id, _ in role_nodes:
                in_chain = chain_nodes is None or node_id in chain_nodes
                alpha = 1.0 if in_chain else self.NODE_ALPHA_DIMMED

                base = (
                    "#ffffff" if node_id in added_nodes
                    else self.NODE_COLOURS.get(role, "#aaaaaa")
                )
                colours.append(to_rgba(base, alpha))
                edge_colours.append(to_rgba("#ffffff", alpha * 0.8))
                sizes.append(
                    self.SELECTED_NODE_SIZE if node_id == self.selected_node
                    else self.NODE_SIZE
                )

            self.ax.scatter(
                coords[:, 0], coords[:, 1], coords[:, 2],
                s=sizes,
                c=colours,
                edgecolors=edge_colours,
                linewidths=0.8,
                depthshade=False,  # we handle dimming ourselves
            )

        # ---- Labels -----------------------------------------------------
        show_all_labels = len(new_node_ids) <= self.LABEL_NODE_LIMIT
        for node_id, (x, y, z) in self.positions.items():
            in_chain = chain_nodes is None or node_id in chain_nodes
            if node_id == self.selected_node or (show_all_labels and in_chain):
                self.ax.text(
                    x, y, z + 0.08, str(node_id),
                    color="white", fontsize=8, ha="center",
                )

        # Establish limits only on the first render. On subsequent generations,
        # restore the exact limits the user currently has (including toolbar zoom).
        # This prevents redraws from automatically zooming out as the network grows.
        if had_existing_view:
            self.ax.set_xlim(saved_xlim)
            self.ax.set_ylim(saved_ylim)
            self.ax.set_zlim(saved_zlim)
            self.ax.view_init(
                elev=saved_elev,
                azim=saved_azim,
                roll=saved_roll,
            )
        else:
            limit = max(abs(self.INPUT_X), abs(self.OUTPUT_X)) * 1.1
            self.ax.set_xlim(-limit, limit)
            self.ax.set_ylim(-limit, limit)
            self.ax.set_zlim(-limit, limit)
            self.ax.set_box_aspect((1, 1, 1))
            self._view_initialized = True

        self.previous_node_ids = new_node_ids
        self.previous_connections = new_connections

        if generation is not None:
            self.history_label.configure(
                text=f"Generation: {generation}"
            )

        if fitness is not None:
            self.title_label.configure(
                text=f"Generation {generation} | Fitness: {fitness:.6f}"
            )

        self._update_inspection()
        self.canvas.draw_idle()

    def update_network(self, snapshot, generation=None, fitness=None):
        """Request an asynchronous network layout."""

        if self._closed:
            return

        self.layout_request_id += 1

        request = (
            self.layout_request_id,
            snapshot,
            self.positions.copy(),
            generation,
            fitness,
        )

        self._replace_queue_item(
            self.layout_requests,
            request,
        )

    # ---------------------------------------------------------
    # Selection
    # ---------------------------------------------------------

    def _on_plot_press(self, event):
        self._press_xy = (event.x, event.y)

    def _on_plot_click(self, event):
        """Treat a press-release with little movement as a click."""
        if event.inaxes != self.ax or not self.positions:
            return

        press = getattr(self, "_press_xy", None)
        if press is None:
            return
        if abs(event.x - press[0]) > 4 or abs(event.y - press[1]) > 4:
            return  # it was a rotate/pan drag

        proj = self.ax.get_proj()
        closest_id = None
        closest_distance = self.PICK_RADIUS_PIXELS ** 2

        for node_id, (x, y, z) in self.positions.items():
            x2, y2, _ = proj3d.proj_transform(x, y, z, proj)
            sx, sy = self.ax.transData.transform((x2, y2))
            distance = (sx - event.x) ** 2 + (sy - event.y) ** 2

            if distance < closest_distance:
                closest_distance = distance
                closest_id = node_id

        # Clicking empty space clears the selection.
        self.selected_node = closest_id
        self.selected_connection = None

        self._render_network(self.snapshot, self.positions)

    def _update_inspection(self):
        self.info_text.configure(state="normal")
        self.info_text.delete("1.0", "end")

        if self.snapshot is None:
            self.info_text.insert(
                "end",
                "Waiting for network..."
            )

        elif self.selected_node is not None:
            node = next(
                (
                    item
                    for item in self.snapshot["nodes"]
                    if int(item["id"]) == self.selected_node
                ),
                None,
            )

            if node is not None:
                self.info_text.insert(
                    "end",
                    f"NEURON {node['id']}\n\n"
                    f"Type: {node['type']}\n"
                    f"Bias: {node['bias']:.6f}\n"
                    f"Response: {node['response']:.6f}\n"
                    f"Activation: {node['activation']}\n"
                    f"Aggregation: {node['aggregation']}\n"
                    f"Layer: {node['layer']}\n"
                    f"Useful: {node['useful']}\n"
                )

        self.info_text.configure(state="disabled")

    # ---------------------------------------------------------
    # Automatic zoom
    # ---------------------------------------------------------

    def _update_auto_zoom(self, positions):
        if not positions:
            return

        coords = np.asarray(
            list(positions.values()),
            dtype=float,
        )

        # Find the furthest node from the centre on any axis.
        required_limit = (
            float(np.max(np.abs(coords)))
            / self.zoom_out_margin
        )

        # Only zoom out. Never automatically zoom back in.
        if required_limit > self.current_view_limit:
            while self.current_view_limit < required_limit:
                self.current_view_limit *= self.zoom_out_factor

        limit = self.current_view_limit

        self.ax.set_xlim(-limit, limit)
        self.ax.set_ylim(-limit, limit)
        self.ax.set_zlim(-limit, limit)
        self.ax.set_box_aspect((1, 1, 1))

    # ---------------------------------------------------------
    # Generation history
    # ---------------------------------------------------------

    def add_history(self, snapshot, generation, fitness):
        self.history.append({
            "snapshot": snapshot,
            "generation": generation,
            "fitness": fitness,
        })

        self.history_slider.configure(
            to=max(0, len(self.history) - 1)
        )

        if self.live_mode:
            self.history_index = len(self.history) - 1

            self.update_network(
                snapshot,
                generation,
                fitness,
            )

    def _on_history_change(self, value):
        if not self.history:
            return

        index = int(float(value))

        index = max(
            0,
            min(index, len(self.history) - 1),
        )

        self.live_mode = (
            index == len(self.history) - 1
        )

        self.history_index = index

        entry = self.history[index]

        self.update_network(
            entry["snapshot"],
            entry["generation"],
            entry["fitness"],
        )

    def return_to_live(self):
        if not self.history:
            return

        self.live_mode = True

        self.history_index = len(self.history) - 1

        self.history_slider.set(
            self.history_index
        )

        entry = self.history[self.history_index]

        self.update_network(
            entry["snapshot"],
            entry["generation"],
            entry["fitness"],
        )
