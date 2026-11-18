"""
TensorNEAT Tkinter dashboard
=============================

Run with:
    python tensorneat_gui.py

Requirements:
    pip install custom TensorNEAT dependencies already used by your project

This file expects crossovers.py alongside this script, defining:
    NudgeCrossover(alpha=0.2)

The GUI runs TensorNEAT in a worker thread and updates Tkinter via a queue.
"""
import pickle
import queue
import threading
import time
import traceback
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
from tkinter.scrolledtext import ScrolledText
from pathlib import Path
import importlib.util

import jax
import numpy as np

from tensorneat.pipeline import Pipeline
from tensorneat import algorithm, genome, problem
from tensorneat.common import ACT

import crossovers

import jax.numpy as jnp
from tensorneat.problem.func_fit import FuncFit

import inspect

from tensorneat.genome import BaseCrossover, DefaultMutation
from tensorneat.genome.operations import DefaultCrossover

import runpy

from network3dvisualizer import Network3DVisualizer
from network_visualizer import extract_network_snapshot

MAX_NETWORK_HISTORY = 1


class ReportingPipeline(Pipeline):
    def __init__(
        self,
        *args,
        report_queue,
        stop_event,
        initial_genome=None,
        neat_dict_fn,
        events,
        num_inputs,
        num_outputs,
        **kwargs
    ):
        super().__init__(*args, **kwargs)

        self.generation_count = 0
        self.events = events
        self.report_queue = report_queue
        self.stop_event = stop_event

        self.num_inputs = num_inputs
        self.num_outputs = num_outputs

        self.input_ids = set(range(num_inputs))
        self.output_ids = set(
            range(num_inputs, num_inputs + num_outputs)
        )

        self.initial_genome = initial_genome

        self.generation_best_genome = None
        self.generation_best_fitness = float("-inf")

        self.best_genome = None
        self.best_fitness = float("-inf")

        self.current_genome = None
        self.current_network = None
        self.best_network = None

        self.network_dict_fn = neat_dict_fn
        self.generation_timestamp = None

    def setup(self, state=None):
        from tensorneat.common import State
        if state is None:
            state = State()
        state = state.register(randkey=jax.random.PRNGKey(self.seed))
        state = self.algorithm.setup(state)
        if self.initial_genome is not None:
            state = self._seed_population(state)
        state = self.problem.setup(state)
        return state

    def _seed_population(self, state):
        from tensorneat.common import State
        nodes, conns = (jnp.asarray(a) for a in self.initial_genome)

        if (nodes.shape != state.pop_nodes.shape[1:]
                or conns.shape != state.pop_conns.shape[1:]):
            raise ValueError(
                "Saved genome doesn't fit the current settings.\n"
                f"Saved nodes {nodes.shape}, conns {conns.shape}; "
                f"expected {state.pop_nodes.shape[1:]}, "
                f"{state.pop_conns.shape[1:]}.\n"
                "Match Maximum nodes (and the fitness file's inputs/outputs) "
                "to the saved run."
            )

        pop = state.pop_nodes.shape[0]
        patched = []

        def patch(st, path=""):
            updates = {}
            for key, val in st.state_dict.items():
                full = f"{path}{key}"
                if isinstance(val, State):
                    updates[key] = patch(val, full + ".")
                    continue
                shape = getattr(val, "__dict__", None) is not None and None
                shape = getattr(val, "shape", None) if hasattr(val, "dtype") else None
                if shape is None or len(shape) < 2:
                    continue
                if full == "pop_nodes":
                    updates[key] = jnp.broadcast_to(nodes, (pop, *nodes.shape))
                elif full == "pop_conns":
                    updates[key] = jnp.broadcast_to(conns, (pop, *conns.shape))
                elif "node" in key and shape[1:] == nodes.shape:
                    updates[key] = val.at[0].set(nodes)
                    patched.append(full)
                elif "conn" in key and shape[1:] == conns.shape:
                    updates[key] = val.at[0].set(conns)
                    patched.append(full)
            return st.update(**updates) if updates else st

        state = patch(state)
        return state

    def analysis(self, state, pop, fitnesses):
        """Capture generation statistics and champion network snapshots."""

        fitnesses = np.asarray(jax.device_get(fitnesses))
        valid = np.isfinite(fitnesses)

        if not np.any(valid):
            return

        safe_fitnesses = np.where(valid, fitnesses, -np.inf)

        # Select this generation's champion.
        best_idx = int(np.argmax(safe_fitnesses))
        current_fitness = float(safe_fitnesses[best_idx])

        current_genome = (
            pop[0][best_idx],
            pop[1][best_idx],
        )

        self.current_genome = current_genome

        # Extract the actual TensorNEAT network and convert it
        # into a plain Python snapshot for the visualiser.
        try:
            network = self.network_dict_fn(state, *current_genome)

            current_snapshot = extract_network_snapshot(
                network,
                self.input_ids,
                self.output_ids,
            )

            visualizer_error = None

        except Exception as exc:
            current_snapshot = None
            visualizer_error = str(exc)
            traceback.print_exc()

        self.current_network = current_snapshot

        # Update the all-time champion only on a new record.
        new_record = current_fitness > self.best_fitness

        if new_record:
            self.best_fitness = current_fitness
            self.best_genome = current_genome
            self.best_network = current_snapshot

        # Generation statistics.
        valid_fitnesses = fitnesses[valid]

        max_f = float(np.max(valid_fitnesses))
        min_f = float(np.min(valid_fitnesses))
        mean_f = float(np.mean(valid_fitnesses))
        std_f = float(np.std(valid_fitnesses))

        self.generation_count += 1

        elapsed = (
            time.time() - self.generation_timestamp
            if self.generation_timestamp is not None
            else 0.0
        )

        # Send one dictionary event, matching _poll_events().
        self.events.put({
            "type": "generation",

            "generation": self.generation_count,

            "best": max_f,
            "best_fitness": self.best_fitness,
            "mean": mean_f,
            "min": min_f,
            "std": std_f,

            "valid": int(np.sum(valid)),
            "elapsed": elapsed,

            "current_fitness": current_fitness,
            "current_network": current_snapshot,

            "new_record": new_record,
            "best_network": self.best_network,

            "visualizer_error": visualizer_error,
        })


class UserFitnessProblem(FuncFit):
    """TensorNEAT problem backed by the user's editable fitness file."""

    def __init__(self, fitness_file):
        self.fitness_file = Path(fitness_file)

        spec = importlib.util.spec_from_file_location(
            "user_fitness_function",
            self.fitness_file,
        )
        if spec is None or spec.loader is None:
            raise ImportError(
                f"Cannot load fitness function: {self.fitness_file}"
            )

        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        self.module = module

        for name in ("INPUTS", "TARGETS", "fitness"):
            if not hasattr(module, name):
                raise ValueError(
                    f"{self.fitness_file.name} must define {name}"
                )

        self._inputs = jnp.asarray(module.INPUTS, dtype=jnp.float32)

        targets = module.TARGETS
        if targets is None:
            targets = module.TARGETS

        self._targets = (
            None if targets is None
            else jnp.asarray(targets, dtype=jnp.float32)
        )

        self.fitness_fn = module.fitness

        if self._inputs.ndim != 2:
            raise ValueError("INPUTS must be a 2D array.")

        if self._targets.ndim != 2:
            raise ValueError("TARGETS must be a 2D array.")

        if self._inputs.shape[0] != self._targets.shape[0]:
            raise ValueError(
                "INPUTS and TARGETS must have the same number of rows."
            )

        if self._inputs.shape[0] == 0:
            raise ValueError("At least one training example is required.")

        super().__init__(error_method="mse")

    @property
    def inputs(self):
        return self._inputs

    @property
    def targets(self):
        return self._targets

    @property
    def input_shape(self):
        return self.inputs.shape

    @property
    def output_shape(self):
        return self._targets.shape

    def evaluate(self, state, randkey, act_func, params):
        import jax.numpy as jnp
        from jax import vmap

        predictions = vmap(
            act_func,
            in_axes=(None, None, 0),
        )(state, params, self.inputs)

        if self.targets is not None:
            reference = jnp.asarray(self.targets)
        else:
            reference = jnp.asarray(self.inputs)

        return self.fitness_fn(predictions, reference)

    @inputs.setter
    def inputs(self, value):
        self._inputs = value

class ScrollableFrame(ttk.Frame):
    """A vertically scrollable container. Put child widgets in `.inner`."""

    def __init__(self, parent, width=290):
        super().__init__(parent)
        bg = ttk.Style().lookup("TFrame", "background") or "#ececec"

        self.canvas = tk.Canvas(
            self, width=width, background=bg,
            highlightthickness=0, borderwidth=0,
        )
        self.scrollbar = ttk.Scrollbar(
            self, orient="vertical", command=self.canvas.yview
        )
        self.canvas.configure(yscrollcommand=self.scrollbar.set)

        self.scrollbar.pack(side="right", fill="y")
        self.canvas.pack(side="left", fill="both", expand=True)

        self.inner = ttk.Frame(self.canvas, padding=(0, 0, 12, 0))
        self._window = self.canvas.create_window(
            (0, 0), window=self.inner, anchor="nw"
        )

        # Keep the scroll region and inner width in sync with the layout.
        self.inner.bind("<Configure>", self._on_inner_configure)
        self.canvas.bind("<Configure>", self._on_canvas_configure)

        # One global wheel binding; the handler checks where the pointer is,
        # so the editor and results text boxes keep their own scrolling.
        self.bind_all("<MouseWheel>", self._on_mousewheel, add="+")
        self.bind_all("<Button-4>", self._on_mousewheel, add="+")   # Linux
        self.bind_all("<Button-5>", self._on_mousewheel, add="+")   # Linux

    def _on_inner_configure(self, _event):
        self.canvas.configure(scrollregion=self.canvas.bbox("all"))

    def _on_canvas_configure(self, event):
        self.canvas.itemconfigure(self._window, width=event.width)

    def _pointer_inside(self, event):
        widget = self.winfo_containing(event.x_root, event.y_root)
        while widget is not None:
            if widget is self:
                return True
            widget = widget.master
        return False

    def _on_mousewheel(self, event):
        if not self._pointer_inside(event):
            return
        # Nothing to scroll if the content already fits.
        if self.inner.winfo_reqheight() <= self.canvas.winfo_height():
            return
        if event.num == 4:
            step = -1
        elif event.num == 5:
            step = 1
        else:
            step = -1 if event.delta > 0 else 1
        self.canvas.yview_scroll(step * 3, "units")

class TensorNEATGui:
    def __init__(self, root):
        self.root = root
        self.root.title("TensorNEAT Evolution Studio")
        self.root.geometry("1080x760")
        self.root.minsize(900, 650)

        self.visualizer = None
        self.network_history = []

        self.latest_current_snapshot = None
        self.latest_best_snapshot = None


        self.fitness_file = Path(__file__).resolve().with_name(
            "fitness_function.py"
        )
        self._fitness_save_job = None

        self.events = queue.Queue()
        self.stop_event = threading.Event()
        self.worker = None
        self.winner = None
        self.running = False
        self.best_history = []
        self.mean_history = []
        self.current_config = None
        self.current_problem = None

        self.live_lock = threading.Lock()
        self.pending_live = None
        self.entries = {}
        self.change_marks = []

        self.vars = {
            "pop_size": tk.StringVar(value="500"),
            "species_size": tk.StringVar(value="10"),
            "survival_threshold": tk.StringVar(value="0.2"),
            "generation_limit": tk.StringVar(value="300"),
            "fitness_target": tk.StringVar(value="-0.001"),
            "seed": tk.StringVar(value="42"),
            "initial_bonus_nodes": tk.StringVar(value="16"),
            "max_nodes": tk.StringVar(value="10"),
            "max_conns": tk.StringVar(value="1024"),
            "growth": tk.StringVar(value="0.2"),
            "deterrent": tk.StringVar(value="0.2"),
            "alpha": tk.StringVar(value="0.2"),
            "target_gens": tk.StringVar(value="100"),
            "p_default": tk.StringVar(value="0.5"),
        }
        self.continue_var = tk.BooleanVar(value=False)

        self._configure_style()
        self._build_ui()
        self.root.after(100, self._poll_events)
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

    def _configure_style(self):
        style = ttk.Style()
        try:
            style.theme_use("clam")
        except tk.TclError:
            pass
        style.configure("Title.TLabel", font=("Segoe UI", 19, "bold"))
        style.configure("Metric.TLabel", font=("Segoe UI", 15, "bold"))
        style.configure("Muted.TLabel", foreground="#687386")
        style.configure("Accent.TButton", font=("Segoe UI", 10, "bold"))

    def _build_ui(self):
        outer = ttk.Frame(self.root, padding=14)
        outer.pack(fill="both", expand=True)

        header = ttk.Frame(outer)
        header.pack(fill="x", pady=(0, 12))
        ttk.Label(header, text="TensorNEAT Evolution Studio",
                  style="Title.TLabel").pack(side="left")
        self.status_var = tk.StringVar(value="Ready")
        ttk.Label(header, textvariable=self.status_var,
                  style="Muted.TLabel").pack(side="right", padx=8)

        main = ttk.Panedwindow(outer, orient="horizontal")
        main.pack(fill="both", expand=True)

        settings_scroll = ScrollableFrame(main, width=290)
        settings = settings_scroll.inner  # every existing widget keeps using `settings`
        visual = ttk.Frame(main)
        main.add(settings_scroll, weight=0)
        main.add(visual, weight=1)

        ttk.Label(settings, text="Experiment settings",
                  font=("Segoe UI", 12, "bold")).pack(anchor="w", pady=(0, 8))

        fields = [
            ("Population size", "pop_size"),
            ("Species size", "species_size"),
            ("Survival threshold", "survival_threshold"),
            ("Generation limit", "generation_limit"),
            ("Fitness target", "fitness_target"),
            ("Random seed", "seed"),
            ("Growth", "growth"),
            ("Deterioration", "deterrent"),
            ("Maximum nodes", "max_nodes"),
            ("Maximum connections", "max_conns"),
            ("Initial Nodes", "initial_bonus_nodes"),
            ("Nudge alpha", "alpha"),
            ("Target Gens", "target_gens"),
        ]
        for label, key in fields:
            row = ttk.Frame(settings)
            row.pack(fill="x", pady=4)
            live = key in LIVE_KEYS
            ttk.Label(row, text=label + (" •" if live else ""), width=19).pack(side="left")
            entry = ttk.Entry(row, textvariable=self.vars[key], width=11)
            entry.pack(side="right")
            self.entries[key] = entry
            if live:
                entry.bind("<Return>", self.apply_live)
                entry.bind("<FocusOut>", self.apply_live)

        ttk.Label(settings, text="• editable during training",
                  style="Muted.TLabel").pack(anchor="w")
        ttk.Button(settings, text="Apply changes now",
                   command=lambda: self.apply_live(None)).pack(fill="x", pady=(6, 0))

        # Crossover selector
        ttk.Label(
            settings,
            text="Crossover operator",
            font=("Segoe UI", 10, "bold"),
        ).pack(anchor="w", pady=(10, 4))

        self.crossover_var = tk.StringVar(value="NudgeCrossover")

        self.crossover_combo = ttk.Combobox(
            settings,
            textvariable=self.crossover_var,
            values=sorted(discover_crossovers().keys()),
            state="readonly",
            width=25,
        )
        self.crossover_combo.pack(fill="x", pady=(0, 6))

        self.crossover_combo.bind("<<ComboboxSelected>>", self._on_crossover_selected)

        ttk.Button(
            settings,
            text="Refresh crossover list",
            command=self._refresh_crossovers,
        ).pack(fill="x", pady=(0, 8))

        # Dynamic problem and crossover information
        self.problem_info_var = tk.StringVar(value="Loading problem info...")

        self.problem_info_label = ttk.Label(
            settings,
            textvariable=self.problem_info_var,
            style="Muted.TLabel",
            justify="left",
            wraplength=240,
        )
        self.problem_info_label.pack(anchor="w", pady=(10, 12))

        self.crossover_combo.bind(
            "<<ComboboxSelected>>",
            self._update_problem_info,
        )

        self._update_problem_info()

        self.continue_check = ttk.Checkbutton(
            settings,
            text="Continue from latest genome",
            variable=self.continue_var,
        )
        self.continue_check.pack(anchor="w", pady=(0, 4))

        self.start_btn = ttk.Button(
            settings, text="▶  Start training", style="Accent.TButton",
            command=self.start_training
        )
        self.start_btn.pack(fill="x", pady=3)
        self.stop_btn = ttk.Button(
            settings, text="■  Stop after current generation",
            command=self.stop_training, state="disabled"
        )
        self.stop_btn.pack(fill="x", pady=3)

        self.save_btn = ttk.Button(
            settings, text=" Save latest winner",
            command=self.save_latest, state="disabled"
        )
        self.save_btn.pack(fill="x", pady=3)

        self.load_btn = ttk.Button(
            settings, text=" Load data",
            command=self.load_data, state="normal"
        )
        self.load_btn.pack(fill="x", pady=3)

        self.visualizer_btn = ttk.Button(
            settings,
            text="3D Network Visualiser",
            command=self.open_network_visualizer,
        )
        self.visualizer_btn.pack(fill="x", pady=3)


        ttk.Separator(settings).pack(fill="x", pady=12)
        ttk.Label(settings, text="Live metrics",
                  font=("Segoe UI", 12, "bold")).pack(anchor="w", pady=(0, 8))

        self.metric_vars = {
            "generation": tk.StringVar(value="—"),
            "best": tk.StringVar(value="—"),
            "mean": tk.StringVar(value="—"),
            "valid": tk.StringVar(value="—"),
            "elapsed": tk.StringVar(value="—"),
        }
        metric_labels = [
            ("Generation", "generation"),
            ("Best fitness", "best"),
            ("Mean fitness", "mean"),
            ("Valid genomes", "valid"),
            ("Generation time", "elapsed"),
        ]
        for label, key in metric_labels:
            box = ttk.Frame(settings, padding=(0, 4))
            box.pack(fill="x")
            ttk.Label(box, text=label, style="Muted.TLabel").pack(anchor="w")
            ttk.Label(box, textvariable=self.metric_vars[key],
                      style="Metric.TLabel").pack(anchor="w")

        ttk.Label(visual, text="Fitness over generations",
                  font=("Segoe UI", 12, "bold")).pack(anchor="w")
        self.chart = tk.Canvas(
            visual, height=270, background="#ffffff",
            highlightthickness=1, highlightbackground="#d9dee7"
        )
        self.chart.pack(fill="x", pady=(8, 12))
        self.chart.bind("<Configure>", lambda _event: self._draw_chart())

        # Horizontal section beneath the fitness chart
        results_pane = ttk.Panedwindow(visual, orient="horizontal")
        results_pane.pack(fill="both", expand=True, pady=(8, 0))

        # left: User text input
        input_frame = ttk.Frame(results_pane, padding=(8, 0, 0, 0))
        results_pane.add(input_frame, weight=1)

        editor_header = ttk.Frame(input_frame)
        editor_header.pack(fill="x")

        ttk.Label(
            editor_header,
            text="Fitness function",
            font=("Segoe UI", 12, "bold")
        ).pack(side="left")

        self.fitness_status = tk.StringVar(value="Loading...")
        ttk.Label(
            editor_header,
            textvariable=self.fitness_status,
            style="Muted.TLabel"
        ).pack(side="right")

        self.text_input = ScrolledText(
            input_frame,
            height=13,
            wrap="none",
            font=("Consolas", 10),
            undo=True
        )
        self.text_input.pack(fill="both", expand=True, pady=(8, 0))

        # Load the existing file or create a starter function.
        default_code = '''def fitness(net, genome, config):
            """
            Evaluate an evolved network.
            Return a numeric fitness score.
            """
            inputs = [0.0, 1.0]
            outputs = net.activate(inputs)

            expected = [1.0]
            error = sum(
                (actual - target) ** 2
                for actual, target in zip(outputs, expected)
            )

            return 1.0 - error
        '''

        try:
            if self.fitness_file.exists():
                source = self.fitness_file.read_text(encoding="utf-8")
            else:
                source = default_code
                self.fitness_file.write_text(source, encoding="utf-8")

            self.text_input.insert("1.0", source)
            self.fitness_status.set(f"Synced: {self.fitness_file.name}")

        except OSError as exc:
            self.text_input.insert("1.0", default_code)
            self.fitness_status.set(f"File error: {exc}")

        # Reset the modified flag after initial loading.
        self.text_input.edit_modified(False)

        # Automatically schedule saves when the user edits the text.
        self.text_input.bind(
            "<<Modified>>",
            self._on_fitness_text_modified
        )

        # right: Best network and predictions
        results_frame = ttk.Frame(results_pane, padding=(0, 0, 8, 0))
        results_pane.add(results_frame, weight=1)

        ttk.Label(
            results_frame,
            text="Best network and predictions",
            font=("Segoe UI", 12, "bold")
        ).pack(anchor="w")

        self.results = ScrolledText(
            results_frame,
            height=13,
            wrap="word",
            font=("Consolas", 10),
            state="disabled"
        )
        self.results.pack(fill="both", expand=True, pady=(8, 0))

        self._set_results(
            "Configure the experiment and press Start training.\n\n"
            "Generation statistics and the final evolved network will appear here."
        )

    def _update_problem_info(self, _event=None):
        crossover_name = self.crossover_var.get()

        try:
            num_inputs, num_outputs = read_fitness_dimensions(
                self.fitness_file
            )

            self.problem_info_var.set(
                "Problem: User-defined fitness\n"
                f"Inputs: {num_inputs}\n"
                f"Outputs: {num_outputs}\n"
                f"Min nodes: {num_inputs + num_outputs}\n"
                f"Min connections: {num_inputs * num_outputs}\n"
                f"Crossover: {crossover_name}"
            )

        except Exception as exc:
            self.problem_info_var.set(
                "Problem: User-defined fitness\n"
                f"Crossover: {crossover_name}\n"
                f"Dimensions unavailable: {exc}"
            )

    def _refresh_crossovers(self):
        """Refresh the dropdown from the current crossovers module."""
        try:
            import importlib

            importlib.invalidate_caches()
            importlib.reload(crossovers)

            names = sorted(discover_crossovers().keys())
            self.crossover_combo["values"] = names

            if self.crossover_var.get() not in names:
                self.crossover_var.set(
                    "NudgeCrossover" if "NudgeCrossover" in names
                    else names[0] if names
                    else ""
                )

            self.status_var.set(
                f"Found {len(names)} crossover operators"
            )

        except Exception as exc:
            messagebox.showerror(
                "Crossover discovery error",
                str(exc),
            )

    def _on_fitness_text_modified(self, _event=None):
        """Debounce editor changes before writing the file."""
        if not self.text_input.edit_modified():
            return

        # Reset the flag so future edits trigger this handler.
        self.text_input.edit_modified(False)

        if self._fitness_save_job is not None:
            self.root.after_cancel(self._fitness_save_job)

        self.fitness_status.set("Unsaved changes...")

        self._fitness_save_job = self.root.after(
            700,
            self._save_fitness_file
        )

    def _save_fitness_file(self, source=None):
        """Write the current editor contents to the fitness file."""
        self._fitness_save_job = None
        if source is None:
            source = self.text_input.get("1.0", "end-1c")

        if isinstance(source, bytes):
            source = source.decode("utf-8")

        try:
            self.fitness_file.write_text(
                source,
                encoding="utf-8"
            )
            self.fitness_status.set(
                f"Saved: {self.fitness_file.name}"
            )
            self._update_problem_info()

        except OSError as exc:
            self.fitness_status.set("Save failed")
            messagebox.showerror(
                "Fitness function save error",
                f"Could not save the fitness function:\n{exc}"
            )

    def _flush_fitness_file(self):
        """Ensure pending edits are saved before training starts."""
        if self._fitness_save_job is not None:
            self.root.after_cancel(self._fitness_save_job)
            self._fitness_save_job = None

        self._save_fitness_file()

    def _read_config(self):
        try:
            config = {
                "pop_size": int(self.vars["pop_size"].get()),
                "species_size": int(self.vars["species_size"].get()),
                "survival_threshold": float(self.vars["survival_threshold"].get()),
                "generation_limit": int(self.vars["generation_limit"].get()),
                "fitness_target": float(self.vars["fitness_target"].get()),
                "seed": int(self.vars["seed"].get()),
                "max_nodes": int(self.vars["max_nodes"].get()),
                "max_conns": int(self.vars["max_conns"].get()),
                "initial_bonus_nodes": int(self.vars["initial_bonus_nodes"].get()),
                "alpha": float(self.vars["alpha"].get()),
                "target_gens": float(self.vars["target_gens"].get()),
                "growth": float(self.vars["growth"].get()),
                "deterrent": float(self.vars["deterrent"].get()),
                "p_default": float(self.vars["p_default"].get()),
            }
        except ValueError as exc:
            raise ValueError("Please enter valid numeric values in all settings.") from exc

        if config["pop_size"] < 2:
            raise ValueError("Population size must be at least 2.")
        if config["species_size"] < 1:
            raise ValueError("Species size must be at least 1.")
        if not 0 < config["survival_threshold"] <= 1:
            raise ValueError("Survival threshold must be in (0, 1].")
        if config["generation_limit"] < 1:
            raise ValueError("Generation limit must be at least 1.")
        if config["max_nodes"] < 1:
            raise ValueError("Maximum nodes must be at least 1.")
        if config["max_conns"] < 1:
            raise ValueError("Maximum connections must be at least 1.")
        if not 0 <= config["alpha"] <= 1:
            raise ValueError("Nudge alpha must be between 0 and 1.")
        if not 0 <= config["p_default"] <= 1:
            raise ValueError("P default must be between 0 and 1.")
        return config

    def open_network_visualizer(self):
        """Open the 3D visualiser and populate its generation history."""

        if self.visualizer is not None:
            try:
                if self.visualizer.window.winfo_exists():
                    self.visualizer.window.deiconify()
                    self.visualizer.window.lift()
                    self.visualizer.window.focus_force()
                    return
            except (tk.TclError, AttributeError):
                self.visualizer = None

        self.visualizer = Network3DVisualizer(self.root)

        # Populate generations that occurred before the window was opened.
        for entry in self.network_history:
            snapshot = entry["snapshot"]

            if snapshot is None:
                continue

            self.visualizer.add_history(
                snapshot,
                entry["generation"],
                entry["fitness"],
            )

    def start_training(self):
        if self.running:
            return
        try:
            config = self._read_config()
        except ValueError as exc:
            messagebox.showerror("Invalid settings", str(exc))
            return
        config["continue"] = self.continue_var.get()
        if config["continue"] and self.winner is None:
            messagebox.showerror(
                "No genome",
                "There's no genome to continue from. Train first or load a saved one.",
            )
            return

        config["crossover"] = self.crossover_var.get()

        self.current_config = config

        self._flush_fitness_file()

        self.latest_current_snapshot = None
        self.latest_best_snapshot = None

        self.network_history.clear()

        self.running = True
        self.change_marks.clear()
        self._set_locked_fields(True)
        self.stop_event.clear()
        self.best_history.clear()
        self.mean_history.clear()
        self._draw_chart()
        self._set_results("Initialising TensorNEAT...\n")
        self.status_var.set("Initialising")
        self.start_btn.configure(state="disabled")
        self.stop_btn.configure(state="normal")
        self.save_btn.configure(state="normal")
        self.worker = threading.Thread(
            target=self._training_worker, args=(config,), daemon=True
        )
        self.worker.start()

    def stop_training(self):
        if self.running:
            self.stop_event.set()
            self.status_var.set("Stopping after current generation…")
            self.stop_btn.configure(state="disabled")

    def _training_worker(self, cfg):
        try:
            cfg = dict(cfg)  # private copy: the GUI thread edits self.current_config

            num_inputs, num_outputs = read_fitness_dimensions(self.fitness_file)

            min_nodes = num_inputs + num_outputs
            min_conns = num_inputs * num_outputs
            if cfg["max_nodes"] < min_nodes:
                raise ValueError(
                    f"Maximum nodes ({cfg['max_nodes']}) must be at least "
                    f"inputs + outputs = {min_nodes}."
                )
            if cfg["max_conns"] < min_conns:
                raise ValueError(
                    f"Maximum connections ({cfg['max_conns']}) must be at least "
                    f"inputs x outputs = {min_conns}, the initial wiring."
                )

            neat = algorithm.NEAT(
                pop_size=cfg["pop_size"],
                species_size=cfg["species_size"],
                survival_threshold=cfg["survival_threshold"],
                genome=genome.DefaultGenome(
                    num_inputs=num_inputs,
                    num_outputs=num_outputs,
                    init_hidden_layers=(cfg["initial_bonus_nodes"],),
                    max_nodes=cfg["max_nodes"],
                    max_conns=cfg["max_conns"],
                    output_transform=ACT.sigmoid,
                    crossover=make_crossover(cfg),
                    mutation=make_mutation(cfg),
                ),
            )
            custom_problem = UserFitnessProblem(self.fitness_file)
            self.current_problem = custom_problem

            pipeline = ReportingPipeline(
                algorithm=neat,
                events=self.events,
                neat_dict_fn=neat.genome.network_dict,
                problem=custom_problem,
                generation_limit=cfg["generation_limit"],
                fitness_target=cfg["fitness_target"],
                seed=cfg["seed"],
                report_queue=self.events,
                stop_event=self.stop_event,
                initial_genome=self.winner if cfg.get("continue") else None,
                num_inputs=num_inputs,
                num_outputs=num_outputs,
            )

            state = pipeline.setup()

            def compile_step(st):
                # Fresh closure + cleared caches so JAX can't reuse a stale trace
                # that still has the old operator constants baked in.
                jax.clear_caches()

                def step_fn(s):
                    return pipeline.step(s)

                return jax.jit(step_fn).lower(st).compile()

            self.events.put({"type": "status", "text": "Compiling JAX generation step…"})
            compiled_step = compile_step(state)

            reason = "Generation limit reached"
            gens_run = 0
            while True:
                # --- pick up live edits between generations -------------------
                with self.live_lock:
                    pending, self.pending_live = self.pending_live, None

                if pending:
                    previous = dict(cfg)
                    cfg.update(pending)
                    note = "Applied live: " + ", ".join(pending)

                    if any(k in OPERATOR_KEYS for k in pending):
                        self.events.put({"type": "status",
                                         "text": "Recompiling with new settings…"})
                        old_ops = (neat.genome.crossover, neat.genome.mutation)
                        try:
                            neat.genome.crossover = make_crossover(cfg)
                            neat.genome.mutation = make_mutation(cfg)
                            compiled_step = compile_step(state)
                        except Exception as exc:
                            neat.genome.crossover, neat.genome.mutation = old_ops
                            cfg = previous
                            compiled_step = compile_step(state)
                            note = f"Live change failed, reverted: {exc}"

                    self.events.put({"type": "applied", "text": note})

                # --- stop conditions -------------------------------------------
                if gens_run >= cfg["generation_limit"]:
                    break
                if self.stop_event.is_set():
                    reason = "Stopped by user"
                    break

                pipeline.generation_timestamp = time.time()
                state, previous_pop, fitnesses = compiled_step(state)
                fitnesses = jax.device_get(fitnesses)
                pipeline.analysis(state, previous_pop, fitnesses)
                gens_run += 1

                if np.nanmax(np.asarray(fitnesses)) >= cfg["fitness_target"]:
                    reason = "Fitness target reached"
                    break

            best = pipeline.best_genome
            self.winner = best
            if best is not None:
                extra = ""
                mod = custom_problem.module
                if hasattr(mod, "evaluate") and hasattr(mod, "INPUTS_TEST"):
                    transformed = neat.genome.transform(state, *best)
                    run = jax.vmap(neat.forward, in_axes=(None, None, 0))
                    pred_train = np.asarray(run(state, transformed, custom_problem.inputs))
                    pred_test = np.asarray(run(state, transformed,
                                               jnp.asarray(mod.INPUTS_TEST, dtype=jnp.float32)))
                    ev = mod.evaluate(pred_train, pred_test)
                    extra = "Held-out evaluation:\n" + "\n".join(
                        f"  {k:5} soft {v['soft']:.4f}  hard {v['hard']:.4f}  F1 {v['f1']:.3f}"
                        for k, v in ev.items()) + "\n\n"

                network_repr = neat.genome.repr(state, *best)
                predictions = []
                for inputs in custom_problem.inputs[:10]:
                    transformed = neat.genome.transform(state, *best)
                    output = neat.forward(
                        state, transformed,
                        np.asarray(inputs, dtype=np.float32)
                    )
                    predictions.append((inputs, np.asarray(jax.device_get(output)).reshape(-1)))

                network_dict = neat.genome.network_dict(state, *best)


                report = (
                    f"{reason}\n"
                    f"{extra}\n"
                    f"Best fitness: {pipeline.best_fitness:.8f}\n\n"
                    "Training predictions:\n"
                    + "\n".join(
                        f"  {inputs} -> {np.array2string(output, precision=6)}"
                        for inputs, output in predictions
                    )
                    # + "\n\nNetwork representation:\n"
                    # + str(network_repr)
                    # + "\n\nNetwork dictionary:\n"
                    # + str(network_dict)
                )
            else:
                report = f"{reason}\nNo valid best genome was recorded."

            self.events.put({"type": "finished", "report": report})
        except Exception:
            self.events.put({
                "type": "error",
                "message": traceback.format_exc(),
            })

    def _poll_events(self):
        try:
            while True:
                event = self.events.get_nowait()
                kind = event["type"]

                if kind == "generation":
                    self.best_history.append(event["best"])
                    self.mean_history.append(event["mean"])
                    self.metric_vars["generation"].set(str(event["generation"]))
                    self.metric_vars["best"].set(self._fmt(event["best"]))
                    self.metric_vars["mean"].set(self._fmt(event["mean"]))
                    self.metric_vars["valid"].set(str(event["valid"]))
                    self.metric_vars["elapsed"].set(f'{event["elapsed"] * 1000:.1f} ms')
                    self.status_var.set("Training")
                    self._draw_chart()
                    self.latest_current_snapshot = event["current_network"]
                    self.latest_best_snapshot = event["best_network"]

                    # Store the generation champion for later visualisation.
                    snapshot = event["current_network"]

                    entry = {
                        "generation": event["generation"],
                        "fitness": event["current_fitness"],
                        "snapshot": snapshot,
                    }

                    self.network_history.append(entry)

                    if len(self.network_history) > MAX_NETWORK_HISTORY:
                        self.network_history.pop(0)


                    # Update the visualiser if it is already open.
                    if self.visualizer is not None and snapshot is not None:
                        try:
                            self.visualizer.add_history(
                                snapshot,
                                event["generation"],
                                event["current_fitness"],
                            )
                        except tk.TclError:
                            self.visualizer = None

                elif kind == "status":
                    self.status_var.set(event["text"])

                elif kind == "finished":
                    self._set_results(event["report"])
                    self._finish_ui("Completed")

                elif kind == "error":
                    self._set_results("Training failed:\n\n" + event["message"])
                    self._finish_ui("Error")
                    messagebox.showerror(
                        "TensorNEAT error",
                        "Training failed. See the results panel for the traceback."
                    )
                elif kind == "applied":
                    self.change_marks.append(len(self.best_history))
                    self.status_var.set(event["text"])
                    self._draw_chart()
        except queue.Empty:
            pass

        self.root.after(100, self._poll_events)

    @staticmethod
    def _fmt(value):
        try:
            value = float(value)
            return f"{value:.6g}" if np.isfinite(value) else str(value)
        except (TypeError, ValueError):
            return str(value)

    def _finish_ui(self, status):
        self.running = False
        self._set_locked_fields(False)
        self.start_btn.configure(state="normal")
        self.stop_btn.configure(state="disabled")
        self.status_var.set(status)

    def _set_results(self, text):
        self.results.configure(state="normal")
        self.results.delete("1.0", "end")
        self.results.insert("1.0", text)
        self.results.configure(state="disabled")

    def _on_crossover_selected(self, _event=None):
        self._update_problem_info()
        self.apply_live(None)

    def _set_locked_fields(self, locked):
        for key, entry in self.entries.items():
            if key not in LIVE_KEYS:
                entry.configure(state="disabled" if locked else "normal")
        self.continue_check.configure(state="disabled" if locked else "normal")

    def apply_live(self, event=None):
        """Queue edited live settings for the worker. Silent when triggered by focus/Enter."""
        if not self.running or self.current_config is None:
            return
        try:
            new = self._read_config()
        except ValueError as exc:
            if event is None:
                messagebox.showerror("Invalid settings", str(exc))
            else:
                self.status_var.set(f"Not applied: {exc}")
            return
        new["crossover"] = self.crossover_var.get()

        changed = {
            k: new[k] for k in LIVE_KEYS + ("crossover",)
            if new.get(k) != self.current_config.get(k)
        }
        if not changed:
            return

        self.current_config.update(changed)  # so "Save" stores the latest values
        with self.live_lock:
            self.pending_live = {**(self.pending_live or {}), **changed}
        self.status_var.set("Queued: " + ", ".join(changed))

    def _draw_chart(self):
        canvas = self.chart
        canvas.delete("all")

        width = max(canvas.winfo_width(), 300)
        height = max(canvas.winfo_height(), 180)

        left, right, top, bottom = 65, 18, 28, 38
        plot_w = width - left - right
        plot_h = height - top - bottom

        best = [
            float(v) for v in self.best_history
            if np.isfinite(v)
        ]
        mean = [
            float(v) for v in self.mean_history
            if np.isfinite(v)
        ]

        if not best and not mean:
            canvas.create_text(
                width / 2, height / 2,
                text="Fitness data will appear when training starts",
                fill="#687386"
            )
            return

        # --------------------------------------------------
        # Configuration
        # --------------------------------------------------

        WINDOW = 100  # Number of recent generations displayed
        GRID_LINES = 6
        MIN_PADDING = 0.001

        # Align both series to their generation indices.
        # Do not independently compress their X coordinates.
        all_best = list(self.best_history)
        all_mean = list(self.mean_history)

        visible_count = max(
            len(all_best),
            len(all_mean)
        )

        start_index = max(0, visible_count - WINDOW)

        best = all_best[start_index:]
        mean = all_mean[start_index:]

        # --------------------------------------------------
        # Calculate Y-axis range
        # --------------------------------------------------

        values = [
            float(v)
            for series in (best, mean)
            for v in series
            if np.isfinite(v)
        ]

        if not values:
            return

        lo = min(values)
        hi = max(values)

        spread = hi - lo

        if spread < 1e-12:
            # Give identical values a sensible viewing range.
            scale = max(abs(lo), 0.01)
            pad = max(scale * 0.05, MIN_PADDING)
        else:
            pad = max(spread * 0.15, MIN_PADDING)

        lo -= pad
        hi += pad

        # --------------------------------------------------
        # Chart labels
        # --------------------------------------------------

        canvas.create_text(
            13, top + plot_h / 2,
            anchor="center",
            text="Fitness",
            angle=90,
            fill="#687386",
            font=("Segoe UI", 9)
        )

        canvas.create_line(
            left, top,
            left, height - bottom,
            fill="#b8c0cc"
        )

        canvas.create_line(
            left, height - bottom,
                  width - right, height - bottom,
            fill="#b8c0cc"
        )

        # --------------------------------------------------
        # Grid and Y-axis labels
        # --------------------------------------------------

        for i in range(GRID_LINES):
            frac = i / (GRID_LINES - 1)

            y = top + frac * plot_h
            val = hi - frac * (hi - lo)

            canvas.create_line(
                left, y,
                width - right, y,
                fill="#edf0f4"
            )

            canvas.create_text(
                left - 8, y,
                text=f"{val:.5g}",
                anchor="e",
                fill="#687386",
                font=("Segoe UI", 8)
            )

        # --------------------------------------------------
        # Generate graph points
        # --------------------------------------------------

        def points(series):
            if not series:
                return []

            result = []
            denominator = max(min(visible_count - 1, WINDOW), 1)

            for i, value in enumerate(series):
                if value is None or not np.isfinite(value):
                    result.append(None)
                    continue

                # Position relative to the visible window,
                # NOT the original generation number.
                x = left + (i / denominator) * plot_w

                y = top + (
                        (hi - float(value)) / (hi - lo)
                ) * plot_h

                result.append((x, y))

            return result

        # --------------------------------------------------
        # Draw lines, preserving gaps in the data
        # --------------------------------------------------

        def draw_series(series, colour, line_width):
            coords = points(series)
            segment = []

            def flush():
                if len(segment) >= 4:
                    canvas.create_line(
                        *segment,
                        fill=colour,
                        width=line_width,
                        smooth=False
                    )
                elif len(segment) == 2:
                    x, y = segment
                    canvas.create_oval(
                        x - 2, y - 2,
                        x + 2, y + 2,
                        fill=colour,
                        outline=""
                    )

                segment.clear()

            for point in coords:
                if point is None:
                    flush()
                else:
                    segment.extend(point)

            flush()

        draw_series(best, "#2878d0", 2)
        draw_series(mean, "#e28a2b", 2)

        # --------------------------------------------------
        # X-axis labels
        # --------------------------------------------------

        first_gen = start_index + 1

        canvas.create_text(
            left, height - 13,
            text=f"Generation {first_gen}",
            anchor="w",
            fill="#687386",
            font=("Segoe UI", 8)
        )

        canvas.create_text(
            width - right, height - 13,
            text=f"Generation {visible_count}",
            anchor="e",
            fill="#687386",
            font=("Segoe UI", 8)
        )

        # --------------------------------------------------
        # Legend
        # --------------------------------------------------

        canvas.create_line(
            left + 5, 12,
            left + 23, 12,
            fill="#2878d0",
            width=2
        )

        canvas.create_text(
            left + 28, 12,
            text="Best",
            anchor="w",
            fill="#687386",
            font=("Segoe UI", 8)
        )

        canvas.create_line(
            left + 80, 12,
            left + 98, 12,
            fill="#e28a2b",
            width=2
        )

        canvas.create_text(
            left + 103, 12,
            text="Mean",
            anchor="w",
            fill="#687386",
            font=("Segoe UI", 8)
        )

        # Display viewing window
        canvas.create_text(
            width - right, 12,
            text=f"Last {min(visible_count, WINDOW)} generations",
            anchor="e",
            fill="#687386",
            font=("Segoe UI", 8)
        )

    def _apply_config(self, config):
        """Push a saved config dict back into the settings UI."""
        if not config:
            return

        for key, var in self.vars.items():
            if key not in config:
                continue
            value = config[key]
            # target_gens is stored as a float (100.0); show it as 100.
            if isinstance(value, float) and value.is_integer() and key in (
                    "target_gens",
            ):
                value = int(value)
            var.set(str(value))

        crossover = config.get("crossover")
        if crossover:
            names = sorted(discover_crossovers().keys())
            if crossover in names:
                self.crossover_var.set(crossover)
            else:
                messagebox.showwarning(
                    "Crossover not found",
                    f"The saved crossover '{crossover}' isn't available. "
                    "Click 'Refresh crossover list' if you've added it, "
                    "otherwise the current selection is kept.",
                )

        self._update_problem_info()

    def save_latest(self):
        if self.winner is None:
            return
        path = filedialog.asksaveasfilename(
            defaultextension=".pkl",
            initialfile="winner.pkl",
            filetypes=[("Pickle", "*.pkl"), ("All files", "*.*")],
        )

        with open(self.fitness_file, "rb") as fh:
            fitness_text = fh.read()

        if path:
            bundle = {
                "genome": self.winner,
                "config_text": self.current_config,
                "fitness_source": fitness_text,
            }
            with open(path, "wb") as fh:
                pickle.dump(bundle, fh)

    def load_data(self):
        if self.running:
            messagebox.showinfo("Busy", "Stop training before loading a winner.")
            return
        path = filedialog.askopenfilename(
            filetypes=[("Pickle", "*.pkl"), ("All files", "*.*")])
        if not path:
            return

        try:
            # Only load pickles you trust: unpickling can execute arbitrary code.
            with open(path, "rb") as fh:
                data = pickle.load(fh)
        except Exception as exc:
            messagebox.showerror("Load failed", f"Could not read {path}:\n{exc!r}")
            return

        loaded_genome = data["genome"]
        config = data.get("config_text")
        source = data.get("fitness_source")

        if isinstance(source, bytes):
            source = source.decode("utf-8")
        if source is not None:
            source = source.replace("\r\n", "\n").replace("\r", "\n")
            self.text_input.delete("1.0", tk.END)
            self.text_input.insert("1.0", source)
            self._save_fitness_file(source)

        self.winner = loaded_genome
        self.current_config = config
        self._apply_config(config)
        self.save_btn.configure(state="normal")


    def _on_close(self):
        if self.running:
            if not messagebox.askyesno(
                "Close application",
                "Training is running. Request a stop and close?"
            ):
                return
            self.stop_event.set()
        self.root.destroy()


# Settings that can be edited while training is running.
LIVE_KEYS = ("growth", "deterrent", "alpha", "target_gens", "p_default",
             "generation_limit", "fitness_target")
# Subset of those (plus the crossover choice) that requires a recompile.
OPERATOR_KEYS = ("growth", "deterrent", "alpha", "target_gens", "p_default",
                 "crossover")


def create_crossover(name, **settings):
    """Instantiate a crossover, passing only the settings its __init__ accepts."""
    available = discover_crossovers()
    if name not in available:
        raise ValueError(f"Unknown crossover: {name}")
    cls = available[name]
    params = inspect.signature(cls).parameters
    return cls(**{k: v for k, v in settings.items() if k in params})


def make_crossover(cfg):
    return create_crossover(
        cfg["crossover"],
        alpha=float(cfg["alpha"]),
        target_gens=int(cfg["target_gens"]),
        p_default=float(cfg["p_default"]),
    )


def make_mutation(cfg):
    return DefaultMutation(
        node_add=min(1, cfg["growth"]),
        node_delete=min(1, cfg["deterrent"]),
        conn_add=min(1, cfg["growth"] * 2),
        conn_delete=min(1, cfg["deterrent"] * 2),
    )

def discover_crossovers():
    """Find available built-in and user-defined crossover classes."""
    available = {
        "DefaultCrossover": DefaultCrossover,
    }

    for name, cls in inspect.getmembers(crossovers, inspect.isclass):
        if (
            cls is not BaseCrossover
            and issubclass(cls, BaseCrossover)
            and cls.__module__ == crossovers.__name__
        ):
            available[name] = cls

    return available


def read_fitness_dimensions(fitness_file):
    """
    Read the network dimensions from fitness_function.py.

    INPUTS:
        A 2D array/list: examples x input features.

    TARGETS or TARGET:
        A 2D array/list: examples x output features.
        A 1D array/list is treated as a single output per example.

    num_outputs:
        Used if neither TARGETS nor TARGET is defined.
    """
    namespace = runpy.run_path(str(fitness_file))

    inputs = namespace.get("INPUTS")
    if inputs is None:
        raise ValueError(
            "fitness_function.py must define an INPUTS array."
        )

    def feature_count(array, name):
        if len(array) == 0:
            raise ValueError(f"{name} cannot be empty.")

        # NumPy/JAX arrays expose their dimensions through .shape.
        shape = getattr(array, "shape", None)

        if shape is not None:
            if len(shape) >= 2:
                count = int(shape[1])
            elif len(shape) == 1:
                count = 1
            else:
                raise ValueError(f"{name} must contain at least one dimension.")
        else:
            first = array[0]

            try:
                count = len(first)
            except TypeError:
                # A scalar target means one output.
                count = 1

        if count < 1:
            raise ValueError(f"{name} must have at least one feature.")

        return count

    num_inputs = feature_count(inputs, "INPUTS")

    targets = namespace.get("TARGETS")
    if targets is None:
        targets = namespace.get("TARGET")

    if targets is not None:
        num_outputs = feature_count(targets, "TARGETS/TARGET")
    else:
        num_outputs = namespace.get("num_outputs")

        if num_outputs is None:
            raise ValueError(
                "Define TARGETS, TARGET, or num_outputs "
                "in fitness_function.py to determine the output size."
            )

        try:
            num_outputs = int(num_outputs)
        except (TypeError, ValueError):
            raise ValueError("num_outputs must be a positive integer.")

        if num_outputs < 1:
            raise ValueError("num_outputs must be a positive integer.")

    return num_inputs, num_outputs

def main():
    root = tk.Tk()
    TensorNEATGui(root)
    root.mainloop()


if __name__ == "__main__":
    main()
