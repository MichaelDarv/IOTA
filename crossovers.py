import jax
import jax.numpy as jnp
from jax import vmap

from tensorneat.common import I_INF
from tensorneat.genome import BaseCrossover, BaseGene
from tensorneat.genome.utils import extract_gene_attrs, set_gene_attrs

class DynamicNudgeHybridCrossover(BaseCrossover):
    """Hybrid nudge crossover with linear schedules over generations."""

    def __init__(
            self, target_gens, alpha=0.2, alpha_end=1,
            default_start=1.0, default_end=0.0,
    ):
        super().__init__()
        self.total_gens = target_gens
        self.alpha_start, self.alpha_end = alpha, alpha_end
        self.default_start, self.default_end = default_start, default_end

    def _schedule(self, state):
        # Verify the generation counter's name in your TensorNEAT state.
        gen = getattr(state, "generation", 0)
        t = jnp.clip(gen / self.total_gens, 0.0, 0.85)
        alpha = self.alpha_start + (self.alpha_end - self.alpha_start) * t
        p_default = self.default_start + (self.default_end - self.default_start) * t
        return alpha, p_default

    def __call__(self, state, genome, randkey, nodes1, conns1, nodes2, conns2):
        alpha, p_default = self._schedule(state)
        randkey1, randkey2 = jax.random.split(randkey, 2)
        node_randkeys = jax.random.split(randkey1, genome.max_nodes)
        conn_randkeys = jax.random.split(randkey2, genome.max_conns)

        node_continuous = jnp.array([True, True, False, False])
        conn_continuous = jnp.array([True])
        axes = (None, 0, None, 0, 0, None, None, None, None)

        n_fixed = len(genome.node_gene.fixed_attrs)
        node_attrs1 = vmap(extract_gene_attrs, in_axes=(None, 0))(genome.node_gene, nodes1)
        node_attrs2 = vmap(extract_gene_attrs, in_axes=(None, 0))(genome.node_gene, nodes2)
        node_hom = find_homologous_indices(nodes1[:, :n_fixed], nodes2[:, :n_fixed])
        new_node_attrs = vmap(scheduled_crossover_with_index, in_axes=axes)(
            state, node_randkeys, genome.node_gene, node_attrs1, node_hom,
            node_attrs2, alpha, p_default, node_continuous,
        )
        new_nodes = vmap(set_gene_attrs, in_axes=(None, 0, 0))(
            genome.node_gene, nodes1, new_node_attrs
        )

        c_fixed = len(genome.conn_gene.fixed_attrs)
        conn_attrs1 = vmap(extract_gene_attrs, in_axes=(None, 0))(genome.conn_gene, conns1)
        conn_attrs2 = vmap(extract_gene_attrs, in_axes=(None, 0))(genome.conn_gene, conns2)
        conn_hom = find_homologous_indices(conns1[:, :c_fixed], conns2[:, :c_fixed])
        new_conn_attrs = vmap(scheduled_crossover_with_index, in_axes=axes)(
            state, conn_randkeys, genome.conn_gene, conn_attrs1, conn_hom,
            conn_attrs2, alpha, p_default, conn_continuous,
        )
        new_conns = vmap(set_gene_attrs, in_axes=(None, 0, 0))(
            genome.conn_gene, conns1, new_conn_attrs
        )
        return new_nodes, new_conns


def scheduled_crossover_with_index(
        state, randkey, gene, attrs1, homologous_idx, all_attrs2,
        alpha, p_default, continuous,
):
    def use_winner():
        return attrs1

    def do_crossover():
        attrs2 = all_attrs2[homologous_idx]
        k_pick, k_mix = jax.random.split(randkey)

        picked = gene.crossover(state, k_pick, attrs1, attrs2)
        nudged = attrs2 + alpha * (attrs1 - attrs2)

        # Default branch: loser-picked continuous values are nudged.
        default_result = jnp.where(continuous & (picked != attrs1), nudged, picked)

        # Otherwise nudge every continuous attribute.
        use_default = jax.random.uniform(k_mix, attrs1.shape) < p_default
        return jnp.where(continuous & ~use_default, nudged, default_result)

    return jax.lax.cond(homologous_idx == I_INF, use_winner, do_crossover)


class HybridNudgeCrossover(BaseCrossover):
    """Keep the loser's structure. For each matching attribute, flip a coin:
    with probability `prob`, nudge the loser's value toward the winner's;
    otherwise keep the loser's value unchanged.
    """
    def __init__(self, alpha=0.2):
        super().__init__()
        self.alpha = alpha

    def __call__(self, state, genome, randkey, nodes1, conns1, nodes2, conns2):
        """
        use genome1 and genome2 to generate a new genome
        notice that genome1 should have higher fitness than genome2 (genome1 is winner!)
        """
        randkey1, randkey2 = jax.random.split(randkey, 2)
        node_randkeys = jax.random.split(randkey1, genome.max_nodes)
        conn_randkeys = jax.random.split(randkey2, genome.max_conns)

        # Node custom attributes: [bias, response, aggregation, activation]
        node_continuous = jnp.array([True, True, False, False])
        # Connection custom attributes: [weight]
        conn_continuous = jnp.array([True])

        # crossover nodes
        node_keys1 = nodes1[:, 0: len(genome.node_gene.fixed_attrs)]
        node_keys2 = nodes2[:, 0: len(genome.node_gene.fixed_attrs)]
        node_attrs1 = vmap(extract_gene_attrs, in_axes=(None, 0))(
            genome.node_gene, nodes1
        )
        node_attrs2 = vmap(extract_gene_attrs, in_axes=(None, 0))(
            genome.node_gene, nodes2
        )

        node_hom = find_homologous_indices(node_keys1, node_keys2)
        new_node_attrs = vmap(
            nudge_crossover_with_index,
            in_axes=(None, 0, None, 0, 0, None, None, None),
        )(
            state, node_randkeys, genome.node_gene,
            node_attrs1, node_hom, node_attrs2,
            self.alpha, node_continuous,
        )

        new_nodes = vmap(set_gene_attrs, in_axes=(None, 0, 0))(
            genome.node_gene, nodes1, new_node_attrs
        )

        # crossover connections
        conn_keys1 = conns1[:, 0: len(genome.conn_gene.fixed_attrs)]
        conn_keys2 = conns2[:, 0: len(genome.conn_gene.fixed_attrs)]
        conn_attrs1 = vmap(extract_gene_attrs, in_axes=(None, 0))(
            genome.conn_gene, conns1
        )
        conn_attrs2 = vmap(extract_gene_attrs, in_axes=(None, 0))(
            genome.conn_gene, conns2
        )

        conn_hom = find_homologous_indices(conn_keys1, conn_keys2)
        new_conn_attrs = vmap(
            nudge_crossover_with_index,
            in_axes=(None, 0, None, 0, 0, None, None, None),
        )(
            state, conn_randkeys, genome.conn_gene,
            conn_attrs1, conn_hom, conn_attrs2,
            self.alpha, conn_continuous,
        )

        new_conns = vmap(set_gene_attrs, in_axes=(None, 0, 0))(
            genome.conn_gene, conns1, new_conn_attrs
        )

        return new_nodes, new_conns


class NudgeCrossover(BaseCrossover):
    """Keep the loser's structure and nudge matching parameters
    toward the winner.
    """

    def __init__(self, alpha=0.2):
        super().__init__()
        self.alpha = alpha

    def __call__(
        self, state, genome, randkey,
        nodes1, conns1, nodes2, conns2
    ):
        # TensorNEAT's crossover convention: genome 1 is the winner.
        winner_nodes, winner_conns = nodes1, conns1
        loser_nodes, loser_conns = nodes2, conns2

        # Match node genes by their fixed/identity attributes.
        winner_node_keys = winner_nodes[
            :, :len(genome.node_gene.fixed_attrs)
        ]
        loser_node_keys = loser_nodes[
            :, :len(genome.node_gene.fixed_attrs)
        ]

        winner_node_attrs = vmap(
            extract_gene_attrs, in_axes=(None, 0)
        )(genome.node_gene, winner_nodes)

        loser_node_attrs = vmap(
            extract_gene_attrs, in_axes=(None, 0)
        )(genome.node_gene, loser_nodes)

        node_hom = find_homologous_indices(
            loser_node_keys, winner_node_keys
        )

        # Node custom attributes are [bias, response, aggregation, activation].
        def nudge_node(randkey, loser_attrs, winner_idx, winner_attrs):
            del randkey

            def keep_loser():
                return loser_attrs

            def blend():
                target = winner_attrs[winner_idx]

                bias = loser_attrs[0] + self.alpha * (
                    target[0] - loser_attrs[0]
                )
                response = loser_attrs[1] + self.alpha * (
                    target[1] - loser_attrs[1]
                )

                # Do not interpolate categorical attributes.
                return jnp.array([
                    bias,
                    response,
                    loser_attrs[2],
                    loser_attrs[3],
                ])

            return jax.lax.cond(
                winner_idx == I_INF,
                keep_loser,
                blend,
            )

        node_keys = jax.random.split(randkey, genome.max_nodes)

        nudged_node_attrs = vmap(
            nudge_node, in_axes=(0, 0, 0, None)
        )(
            node_keys, loser_node_attrs, node_hom,
            winner_node_attrs
        )

        new_nodes = vmap(
            set_gene_attrs, in_axes=(None, 0, 0)
        )(
            genome.node_gene, loser_nodes, nudged_node_attrs
        )

        # Match connection genes by their fixed/identity attributes.
        winner_conn_keys = winner_conns[
            :, :len(genome.conn_gene.fixed_attrs)
        ]
        loser_conn_keys = loser_conns[
            :, :len(genome.conn_gene.fixed_attrs)
        ]

        winner_conn_attrs = vmap(
            extract_gene_attrs, in_axes=(None, 0)
        )(genome.conn_gene, winner_conns)

        loser_conn_attrs = vmap(
            extract_gene_attrs, in_axes=(None, 0)
        )(genome.conn_gene, loser_conns)

        conn_hom = find_homologous_indices(
            loser_conn_keys, winner_conn_keys
        )

        # Connection custom attributes are [weight].
        def nudge_conn(randkey, loser_attrs, winner_idx, winner_attrs):
            del randkey

            def keep_loser():
                return loser_attrs

            def blend():
                target = winner_attrs[winner_idx]
                weight = loser_attrs[0] + self.alpha * (
                    target[0] - loser_attrs[0]
                )
                return jnp.array([weight])

            return jax.lax.cond(
                winner_idx == I_INF,
                keep_loser,
                blend,
            )

        conn_keys = jax.random.split(
            node_keys[0], genome.max_conns
        )

        nudged_conn_attrs = vmap(
            nudge_conn, in_axes=(0, 0, 0, None)
        )(
            conn_keys, loser_conn_attrs, conn_hom,
            winner_conn_attrs
        )

        new_conns = vmap(
            set_gene_attrs, in_axes=(None, 0, 0)
        )(
            genome.conn_gene, loser_conns, nudged_conn_attrs
        )

        return new_nodes, new_conns

def find_homologous_indices(keys1, keys2):
    """For each key in keys1, return its matching row index in keys2.
    Return I_INF when no match exists.
    """
    matches = jnp.all(
        keys1[:, None, :] == keys2[None, :, :],
        axis=-1,
    )

    found = jnp.any(matches, axis=1)
    indices = jnp.argmax(matches, axis=1)

    return jnp.where(found, indices, I_INF)


def nudge_crossover_with_index(
        state, randkey, gene: BaseGene, attrs1, homologous_idx, all_attrs2,
        alpha, continuous,
):
    """Default crossover for one gene, with loser-picked continuous
    attributes replaced by a nudge toward the winner.
    """

    def use_winner():
        return attrs1

    def do_crossover():
        attrs2 = all_attrs2[homologous_idx]

        # Default per-attribute pick: each value comes from winner or loser.
        picked = gene.crossover(state, randkey, attrs1, attrs2)

        # What a loser-picked continuous attribute becomes: the loser's
        # value moved `alpha` of the way toward the winner's.
        nudged = attrs2 + alpha * (attrs1 - attrs2)

        # If picked == winner (or the parents agree), nudged == picked anyway,
        # so only genuinely loser-picked values change.
        from_loser = picked != attrs1
        return jnp.where(continuous & from_loser, nudged, picked)

    return jax.lax.cond(homologous_idx == I_INF, use_winner, do_crossover)