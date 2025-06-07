use std::collections::VecDeque;
use std::default::Default;
use std::fmt::{Debug, Formatter};
use itertools::Itertools;
use petgraph::Direction;
use petgraph::prelude::EdgeRef;
use rustc_hash::FxHashMap;
use tracing::{debug, trace, warn};
use mchprs_blocks::BlockPos;
use mchprs_redstone::bool_to_ss;
use super::Pass;
use crate::compile_graph::{CompileGraph, CompileLink, CompileNode, LinkType, NodeIdx, NodeState, NodeType};
use crate::{backend, CompilerInput, CompilerOptions};
use mchprs_world::{TickEntry, TickPriority, World};

pub struct ChainCoalesce;

#[derive(Default, Debug)]
struct ChainBuilder {
    chains: Vec<Chain>,
    chain_map: FxHashMap<NodeIdx, usize>,
}

struct Chain {
    nodes: VecDeque<NodeIdx>,
    base_block: (BlockPos, u32),
    pos_map: FxHashMap<BlockPos, NodeIdx>,
}

impl Debug for Chain {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        let mut f = f.debug_struct("Chain");
        f.field("base_block", &self.base_block);
        f.field("nodes", &self.nodes.iter().map(|idx| {
            format!(
                "{}",
                self.pos_map.iter()
                    .find(|(_, idx2)| *idx == **idx2)
                    .unwrap_or((&BlockPos::new(0, 0, 0), idx)).0
            )
        }).collect_vec());
        f.finish()
    }
}

impl<W: World> Pass<W> for ChainCoalesce {
    fn run_pass(
        &self,
        graph: &mut CompileGraph,
        _options: &CompilerOptions,
        ticks: &mut Vec<TickEntry>,
        link_breaks: &mut FxHashMap<NodeIdx, usize>,
        _input: &CompilerInput<'_, W>
    ) {
        let mut builder = ChainBuilder::default();
        let num_coalesced = builder.coalesce_all(graph, ticks, link_breaks);
        trace!("Iteration coalesced {} nodes", num_coalesced);
    }

    fn status_message(&self) -> &'static str {
        "Coalescing chains"
    }
}

impl ChainBuilder {
    fn new_chain(&mut self, chain: Chain) -> usize {
        let index = self.chains.len();
        for node in &chain.nodes {
            self.chain_map.insert(*node, index);
        }
        self.chains.push(chain);
        index
    }

    fn merge_chains(&mut self, first: Chain, second_idx: usize) {
        let second = &mut self.chains[second_idx];
        for node in &first.nodes {
            self.chain_map.insert(*node, second_idx);
        }
        second.push_front(first);
    }

    fn get_chain_index(&mut self, node: NodeIdx) -> Option<usize> {
        self.chain_map.get(&node).cloned()
    }

    fn find_chain(&mut self, graph: &CompileGraph, node: NodeIdx) {
        const MAX_LEN: usize = 255;

        let base_node = &graph[node];
        let base_block = base_node.block
            .expect("No base pos for chain");
        let mut current_node = node;
        let mut chain = Chain::new(current_node, base_block);

        'chain: loop {
            let Ok(edge_out) = graph.edges_directed(current_node, Direction::Outgoing).exactly_one() else {
                break 'chain;
            };
            if edge_out.weight().ty != LinkType::Default {
                break 'chain;
            }

            let target = edge_out.target();

            let existing_chain = self.get_chain_index(target);
            if let Some(existing_chain) = existing_chain {
                let other = &mut self.chains[existing_chain];
                if other.len() + chain.len() <= MAX_LEN {
                    debug!("Adding chain to chain: {:?} at {}", base_node.ty, base_block.0);
                    self.merge_chains(chain, existing_chain);
                    return;
                }
            }

            let node = &graph[target];
            if !matches!(node.ty, NodeType::Repeater { .. } | NodeType::Torch { .. } | NodeType::Chain { .. })
                || !node.is_removable() {
                break 'chain;
            }
            if graph.edges_directed(target, Direction::Incoming).exactly_one().is_err() {
                break 'chain;
            }

            let pos = node.block
                .expect("No pos for chain")
                .0;

            current_node = target;
            debug!("Adding to chain: {:?} at {}", node.ty, pos);
            chain.push_back(current_node, pos);
            if chain.len() >= MAX_LEN {
                break 'chain;
            }
        }

        self.new_chain(chain);
    }

    fn identify_chains(&mut self, graph: &mut CompileGraph) {
        for i in 0..graph.node_count() {
            let idx = NodeIdx::new(i);
            if !graph.contains_node(idx) {
                continue;
            }

            let node = &graph[idx];
            if !matches!(node.ty, NodeType::Repeater { .. } | NodeType::Torch { .. } | NodeType::Chain { .. })
                || !node.is_removable() {
                continue;
            }

            let Ok(edge_in) = graph.edges_directed(idx, Direction::Incoming).exactly_one() else {
                continue;
            };
            if edge_in.weight().ty != LinkType::Default {
                continue;
            }

            if self.get_chain_index(idx).is_none() {
                self.find_chain(graph, idx);
            }
        }
    }

    fn coalesce_all(
        &mut self,
        graph: &mut CompileGraph,
        ticks: &mut Vec<TickEntry>,
        link_breaks: &mut FxHashMap<NodeIdx, usize>,
    ) -> usize {
        self.identify_chains(graph);

        debug!("Chains: {:#?}", self.chains);

        let mut num_coalesced = 0;
        for chain in &self.chains {
            num_coalesced += chain.coalesce(graph, ticks, link_breaks);
        }
        num_coalesced
    }
}

impl Chain {
    fn new(node: NodeIdx, block: (BlockPos, u32)) -> Self {
        let mut chain = Chain {
            nodes: VecDeque::new(),
            base_block: block,
            pos_map: FxHashMap::default()
        };
        chain.nodes.push_back(node);
        chain.pos_map.insert(block.0, node);
        chain
    }

    fn len(&self) -> usize {
        self.nodes.len()
    }

    fn push_back(&mut self, node: NodeIdx, pos: BlockPos) {
        self.nodes.push_back(node);
        self.pos_map.insert(pos, node);
    }

    fn push_front(&mut self, other: Chain) {
        for i in (0..other.nodes.len()).rev() {
            let node = other.nodes[i];
            self.nodes.push_front(node);
        }
        for (pos, idx) in &other.pos_map {
            self.pos_map.insert(*pos, *idx);
        }
        self.base_block = other.base_block;
    }

    fn coalesce(
        &self,
        graph: &mut CompileGraph,
        ticks: &mut Vec<TickEntry>,
        link_breaks: &mut FxHashMap<NodeIdx, usize>,
    ) -> usize {
        let Some(&last) = self.nodes.back() else {
            return 0;
        };
        let last = &graph[last];

        let facing_diode = match last.ty {
            NodeType::Repeater { facing_diode, .. } => facing_diode,
            _ => false
        };

        let mut has_torch = false;
        let mut total_invert = false;
        let mut total_delay = 0;
        let mut max_comp_delay = 1;

        for i in (0..self.nodes.len()).rev() {
            let idx = self.nodes[i];
            let node = &graph[idx];
            match node.ty {
                NodeType::Repeater { delay, .. } => {
                    total_delay += delay;
                    max_comp_delay = max_comp_delay.max(delay);
                }
                NodeType::Torch { invert } => {
                    has_torch = true;
                    if invert {
                        total_invert = !total_invert;
                    };
                    total_delay += 1;
                }
                _ => warn!("Unexpected type found in chain: {:?}", node.ty)
            }
        }

        if has_torch && total_delay < 2 {
            // This would never be an optimization
            return 0;
        }

        if total_delay as usize > backend::direct::TickScheduler::NUM_QUEUES {
            warn!("Chain delay bigger than scheduler queue!");
        }
        let replacement = make_replacement(
            total_delay,
            facing_diode,
            max_comp_delay,
            has_torch,
            total_invert,
            self.base_block,
        );

        if replacement.len() >= self.nodes.len() {
            // This would not be an optimization and may result in endless loop
            return 0;
        }

        // Remove the block from the first decoupled node: it will be added to the new nodes
        // (Duplicate block will cause issues with ticking)
        let front = &mut graph[self.nodes[0]];
        front.block = None;

        let mut prev_powered = false;
        for edge in graph.edges_directed(self.nodes[0], Direction::Incoming) {
            let source = &graph[edge.source()];
            // Chains may have been added in the process, ignore them
            if !matches!(source.ty, NodeType::Chain { ..}) {
                if source.state.powered {
                    prev_powered = true;
                }
            }
        }

        let replacement_idx = add_replacement(graph, replacement);
        replace(graph, &self.nodes, replacement_idx);

        let mut pending_tick = false;
        for entry in ticks.iter() {
            if entry.pos == self.base_block.0 {
                pending_tick = true;
            }
        }

        // Do not add a tick if the previous block is not powered:
        // The first repeater in the chain will power itself
        if !pending_tick && prev_powered {
            ticks.push(TickEntry {
                ticks_left: 0, // Tick immediately, before starting execution
                tick_priority: TickPriority::Normal,
                pos: self.base_block.0,
            });
        }

        link_breaks.insert(*self.nodes.back().unwrap(), total_delay as usize);

        self.nodes.len()
    }
}

fn make_replacement(
    total_delay: u8,
    facing_diode: bool,
    max_comp_delay: u8,
    has_torch: bool,
    invert: bool,
    base_block: (BlockPos, u32),
) -> Vec<CompileNode> {
    let mut replacement = Vec::new();
    if !has_torch {
        replacement.push(
            repeater(
                max_comp_delay,
                facing_diode,
                Some(base_block),
            )
        );
        if total_delay > max_comp_delay {
            replacement.push(
                chain(
                    total_delay - max_comp_delay,
                    facing_diode,
                )
            );
        }
    } else if total_delay == 1 && facing_diode {
        // Not possible:
        // `facing_diode` means we have at least one repeater.
        // `total_delay == 1` means we have at most one component.
        // There is only 1 repeater => `has_torch` is false
        // Previous branch should have been triggered.
        panic!("Illegal state");
    } else {
        // Total delay is always more than 1, if there is a torch
        replacement.push(
            repeater(
                max_comp_delay,
                facing_diode,
                Some(base_block),
            )
        );
        replacement.push(
            torch(
                invert,
            )
        );
        if total_delay > max_comp_delay + 1 {
            replacement.push(
                chain(
                    total_delay - max_comp_delay - 1,
                    facing_diode,
                )
            );
        }
    }
    replacement
}

fn repeater(delay: u8, facing_diode: bool, block: Option<(BlockPos, u32)>) -> CompileNode {
    CompileNode {
        ty: NodeType::Repeater {
            delay,
            facing_diode,
        },
        block,
        state: NodeState {
            powered: false,
            repeater_locked: false,
            output_strength: 0,
        },
        is_input: false,
        is_output: false,
        annotations: Default::default(),
    }
}

fn torch(invert: bool) -> CompileNode {
    CompileNode {
        ty: NodeType::Torch {
            invert,
        },
        block: None,
        state: NodeState {
            powered: invert,
            repeater_locked: false,
            output_strength: bool_to_ss(invert),
        },
        is_input: false,
        is_output: false,
        annotations: Default::default()
    }
}

fn chain(
    delay: u8,
    facing_diode: bool,
) -> CompileNode {
    CompileNode {
        ty: NodeType::Chain {
            delay,
            facing_diode,
        },
        block: None,
        state: NodeState {
            powered: false,
            repeater_locked: false,
            output_strength: 0,
        },
        is_input: false,
        is_output: false,
        annotations: Default::default(),
    }
}

fn add_replacement(graph: &mut CompileGraph, replacement: Vec<CompileNode>) -> (NodeIdx, NodeIdx) {
    let mut replacement_idx = Vec::new();
    for node in replacement {
        replacement_idx.push(graph.add_node(node));
    }

    // Connect replacement to each other
    for i in 0..(replacement_idx.len() - 1) {
        let idx = replacement_idx[i];
        let next_idx = replacement_idx[i + 1];
        graph.add_edge(
            idx,
            next_idx,
            CompileLink {
                ty: LinkType::Default,
                ss: 0
            }
        );
    };

    (*replacement_idx.first().unwrap(), *replacement_idx.last().unwrap())
}

fn replace(graph: &mut CompileGraph, chain: &VecDeque<NodeIdx>, replacement: (NodeIdx, NodeIdx)) {
    let inputs = graph
        .edges_directed(*chain.front().unwrap(), Direction::Incoming)
        .map(|e| (e.id(), e.source(), e.weight().clone()))
        .collect_vec();

    let outputs = graph
        .edges_directed(*chain.back().unwrap(), Direction::Outgoing)
        .map(|e| (e.target(), e.weight().clone()))
        .collect_vec();

    // Connect inputs to replacement
    for (_, node, weight) in &inputs {
        graph.add_edge(
            *node,
            replacement.0,
            (*weight).clone(),
        );
    }

    // Connect outputs to replacement
    for (node, weight) in outputs {
        graph.add_edge(
            replacement.1,
            node,
            weight
        );
    }

    // Remove the original chain inputs
    for (idx, ..) in inputs {
        graph.remove_edge(idx);
    }
}
