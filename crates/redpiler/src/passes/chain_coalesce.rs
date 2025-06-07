//! # [`ChainCoalesce`]
//!
//! This pass merges chains of repeaters and torches if they don't have any interaction with nodes outside the chain.

use std::collections::VecDeque;
use std::default::Default;
use itertools::Itertools;
use petgraph::Direction;
use petgraph::graph::NodeIndex;
use petgraph::prelude::EdgeRef;
use rustc_hash::FxHashMap;
use tracing::{debug, trace, warn};
use mchprs_blocks::BlockPos;
use super::Pass;
use crate::compile_graph::{CompileGraph, CompileLink, CompileNode, LinkType, NodeIdx, NodeState, NodeType};
use crate::{CompilerInput, CompilerOptions, RuntimeAction};
use mchprs_world::{TickPriority, World};
use crate::backend::direct::TickScheduler;

pub struct ChainCoalesce;

#[derive(Default, Debug)]
struct ChainBuilder {
    chains: Vec<Chain>,
    chain_map: FxHashMap<NodeIdx, usize>,
}

#[derive(Debug)]
struct Chain {
    nodes: VecDeque<NodeIdx>,
    blocks: VecDeque<(BlockPos, u32)>,
}

impl<W: World> Pass<W> for ChainCoalesce {
    fn run_pass(
        &self,
        graph: &mut CompileGraph,
        options: &CompilerOptions,
        actions: &mut Vec<RuntimeAction>,
        input: &CompilerInput<'_, W>
    ) {
        let mut builder = ChainBuilder::default();
        let num_coalesced = builder.coalesce_all(graph, actions);
        trace!("Iteration coalesced {} nodes", num_coalesced);
    }

    fn status_message(&self) -> &'static str {
        "Coalescing chains"
    }
}

impl ChainBuilder {
    const MAX_LEN: usize = 255;

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
                if other.len() + chain.len() <= Self::MAX_LEN {
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

            let block = node.block
                .expect("No pos for chain");

            current_node = target;
            chain.push_back(current_node, block);
            if chain.len() >= Self::MAX_LEN {
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
        actions: &mut Vec<RuntimeAction>,
    ) -> usize {
        self.identify_chains(graph);

        let mut num_coalesced = 0;
        for chain in &self.chains {
            num_coalesced += chain.coalesce(graph, actions);
        }
        num_coalesced
    }
}

impl Chain {
    const MAX_DELAY: usize = TickScheduler::NUM_QUEUES;

    fn new(node: NodeIdx, block: (BlockPos, u32)) -> Self {
        let mut chain = Chain {
            nodes: VecDeque::new(),
            blocks: VecDeque::new(),
        };
        chain.nodes.push_back(node);
        chain.blocks.push_back(block);
        chain
    }

    fn len(&self) -> usize {
        self.nodes.len()
    }

    fn push_back(&mut self, node: NodeIdx, block: (BlockPos, u32)) {
        self.nodes.push_back(node);
        self.blocks.push_back(block);
    }

    fn push_front(&mut self, other: Chain) {
        for i in (0..other.nodes.len()).rev() {
            self.nodes.push_front(other.nodes[i]);
            self.blocks.push_front(other.blocks[i]);
        }
    }

    fn coalesce(
        &self,
        graph: &mut CompileGraph,
        actions: &mut Vec<RuntimeAction>,
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

        if total_delay as usize > Self::MAX_DELAY {
            return 0;
        }

        let replacement = make_replacement(
            total_delay,
            facing_diode,
            max_comp_delay,
            has_torch,
            total_invert,
            &self.blocks,
        );

        if replacement.len() >= self.nodes.len() {
            // This would not be an optimization and may result in endless loop
            return 0;
        }

        let tick_second = if let Some(second) = replacement.get(1) {
            matches!(second.ty, NodeType::Torch { .. })
        } else {
            false
        };

        let replacement_idx = add_replacement(graph, replacement);
        replace(graph, &self.nodes, &replacement_idx);

        actions.push(RuntimeAction::Update(
            *replacement_idx.first().unwrap()
        ));
        if tick_second {
            actions.push(RuntimeAction::Tick(
                *replacement_idx.get(1).unwrap(),
                max_comp_delay as u32 + 1, // First repeater delay + 1
                TickPriority::Normal,
            ));
        }
        actions.push(RuntimeAction::BreakLink(
            *self.nodes.back().unwrap(),
            total_delay as u32
        ));

        self.nodes.len()
    }
}

fn make_replacement(
    total_delay: u8,
    facing_diode: bool,
    max_comp_delay: u8,
    has_torch: bool,
    invert: bool,
    blocks: &VecDeque<(BlockPos, u32)>,
) -> Vec<CompileNode> {
    let mut replacement = Vec::new();
    if !has_torch {
        replacement.push(
            repeater(
                max_comp_delay,
                facing_diode,
                blocks.front().cloned(),
            )
        );
        if total_delay > max_comp_delay {
            replacement.push(
                chain(
                    total_delay - max_comp_delay,
                    facing_diode,
                    blocks.get(1).cloned(),
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
        // Total delay is always more than 1 if there is a torch
        replacement.push(
            repeater(
                max_comp_delay,
                facing_diode,
                blocks.front().cloned(),
            )
        );
        replacement.push(
            torch(
                invert,
                blocks.get(1).cloned(),
            )
        );
        if total_delay > max_comp_delay + 1 {
            replacement.push(
                chain(
                    total_delay - max_comp_delay - 1,
                    facing_diode,
                    blocks.get(2).cloned(),
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

fn torch(invert: bool, block: Option<(BlockPos, u32)>) -> CompileNode {
    CompileNode {
        ty: NodeType::Torch {
            invert,
        },
        block,
        state: NodeState {
            powered: false,
            repeater_locked: false,
            output_strength: 0,
        },
        is_input: false,
        is_output: false,
        annotations: Default::default()
    }
}

fn chain(delay: u8, facing_diode: bool, block: Option<(BlockPos, u32)>) -> CompileNode {
    CompileNode {
        ty: NodeType::Chain {
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

fn add_replacement(graph: &mut CompileGraph, replacement: Vec<CompileNode>) -> Vec<NodeIndex> {
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

    replacement_idx
}

fn replace(graph: &mut CompileGraph, chain: &VecDeque<NodeIdx>, replacement: &Vec<NodeIdx>) {
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
            *replacement.first().unwrap(),
            (*weight).clone(),
        );
    }

    // Connect outputs to replacement
    for (node, weight) in outputs {
        graph.add_edge(
            *replacement.last().unwrap(),
            node,
            weight
        );
    }

    // Remove the original chain inputs
    for (idx, ..) in inputs {
        graph.remove_edge(idx);
    }

    // Remove blocks of decoupled nodes
    // The same blocks will be used for some of the replacing nodes
    for node in chain {
        graph[*node].block = None;
    }
}
