use crate::backend::gpu::pipelines::CompilationData;
use crate::backend::gpu::shader_types::{ForwardLink, NeighborInfo, NodeType, State, TickPriority, TypeInfo};
use crate::compile_graph::{CompileGraph, LinkType, NodeIdx};
use crate::{CompilerOptions, TaskMonitor};
use itertools::Itertools;
use mchprs_blocks::blocks::{Block, ComparatorMode};
use mchprs_blocks::BlockPos;
use mchprs_world::TickEntry;
use petgraph::prelude::EdgeRef;
use petgraph::Direction;
use rustc_hash::FxHashMap;
use std::sync::Arc;

fn compile_node(
    graph: &CompileGraph,
    node_idx: NodeIdx,
    nodes_len: usize,
    nodes_map: &FxHashMap<NodeIdx, usize>,
    types: &mut Vec<TypeInfo>,
    neighbor_info: &mut Vec<NeighborInfo>,
    neighbor_links: &mut Vec<ForwardLink>,
    states: &mut Vec<State>,
    default_inputs: &mut Vec<[u32; 16]>,
    side_inputs: &mut Vec<[u32; 16]>,
) {
    let node = &graph[node_idx];

    let mut curr_default_inputs = [0; 16];
    let mut curr_side_inputs = [0; 16];
    for edge in graph.edges_directed(node_idx, Direction::Incoming) {
        let weight = edge.weight();
        let distance = weight.ss;
        let source = edge.source();
        let ss = graph[source].state.output_strength.saturating_sub(distance);
        match weight.ty {
            LinkType::Default => {
                curr_default_inputs[ss as usize] += 1;
            }
            LinkType::Side => {
                curr_side_inputs[ss as usize] += 1;
            }
        }
    }

    use crate::compile_graph::NodeType as CNodeType;
    let mut updates = if node.ty != CNodeType::Constant {
        graph
            .edges_directed(node_idx, Direction::Outgoing)
            .sorted_by_key(|edge| nodes_map[&edge.target()])
            .into_group_map_by(|edge| std::mem::discriminant(&graph[edge.target()].ty))
            .into_values()
            .flatten()
            .map(|edge| {
                let idx = nodes_map[&edge.target()];
                let weight = edge.weight();
                assert!(idx < nodes_len);
                ForwardLink {
                    distance: weight.ss,
                    is_side: weight.ty == LinkType::Side,
                    node_index: idx,
                }
            })
            .collect()
    } else {
        Vec::new()
    };
    if updates.len() > u8::MAX as usize {
        panic!("Too many updates to compile");
    }

    let curr_neighbor_info = NeighborInfo {
        count: updates.len() as u8,
        index: neighbor_links.len(),
    };

    let output_strength = if node.state.output_strength > 0 {
        node.state.output_strength
    } else if node.state.powered {
        15
    } else {
        0
    };
    let state = State {
        output_strength,
        repeater_locked: node.state.repeater_locked,
        changed: false,
    };

    let type_info = match &node.ty {
        CNodeType::Repeater {
            delay,
            facing_diode,
        } => TypeInfo {
            ty: NodeType::Repeater,
            data: *delay,
            facing_diode: *facing_diode,
            far_input: 0,
        },
        CNodeType::Torch => TypeInfo {
            ty: NodeType::Torch,
            data: 0,
            facing_diode: false,
            far_input: 0,
        },
        CNodeType::Comparator {
            mode,
            far_input,
            facing_diode,
        } => TypeInfo {
            ty: NodeType::Comparator,
            data: (*mode == ComparatorMode::Subtract) as u8,
            facing_diode: *facing_diode,
            far_input: far_input.unwrap_or(255),
        },
        CNodeType::Lamp => TypeInfo {
            ty: NodeType::Lamp,
            data: 0,
            facing_diode: false,
            far_input: 0,
        },
        CNodeType::Button => TypeInfo {
            ty: NodeType::Button,
            data: 0,
            facing_diode: false,
            far_input: 0,
        },
        CNodeType::Lever => TypeInfo {
            ty: NodeType::Lever,
            data: 0,
            facing_diode: false,
            far_input: 0,
        },
        CNodeType::PressurePlate => TypeInfo {
            ty: NodeType::PressurePlate,
            data: 0,
            facing_diode: false,
            far_input: 0,
        },
        CNodeType::Trapdoor => TypeInfo {
            ty: NodeType::Trapdoor,
            data: 0,
            facing_diode: false,
            far_input: 0,
        },
        CNodeType::Wire => TypeInfo {
            ty: NodeType::Wire,
            data: 0,
            facing_diode: false,
            far_input: 0,
        },
        CNodeType::Constant => TypeInfo {
            ty: NodeType::Constant,
            data: 0,
            facing_diode: false,
            far_input: 0,
        },
        CNodeType::NoteBlock {
            instrument,
            note,
        } => TypeInfo {
            ty: NodeType::NoteBlock,
            data: 0, //TODO
            facing_diode: false,
            far_input: 0,
        }
    };

    types.push(type_info);
    neighbor_info.push(curr_neighbor_info);
    neighbor_links.append(&mut updates);
    states.push(state);
    default_inputs.push(curr_default_inputs);
    side_inputs.push(curr_side_inputs);
}

pub struct CompiledBlockInfo {
    pub blocks: Vec<Option<(BlockPos, Block)>>,
    pub pos_map: FxHashMap<BlockPos, usize>,
}

pub fn compile(
    graph: CompileGraph,
    ticks: Vec<TickEntry>,
    options: &CompilerOptions,
    _monitor: Arc<TaskMonitor>,
) -> (CompilationData, CompiledBlockInfo) {
    // Create a mapping from compile to backend node indices
    let mut nodes_map = FxHashMap::with_capacity_and_hasher(graph.node_count(), Default::default());
    let mut nodes_map_inv = FxHashMap::with_capacity_and_hasher(graph.node_count(), Default::default());
    for node in graph.node_indices() {
        let idx = nodes_map.len();
        nodes_map.insert(node, idx);
        nodes_map_inv.insert(idx, node);
    }
    let nodes_len = nodes_map.len();

    let mut types = Vec::with_capacity(nodes_len);
    let mut neighbor_info = Vec::with_capacity(nodes_len);
    let mut neighbor_links = Vec::with_capacity(nodes_len * 2);
    let mut states = Vec::with_capacity(nodes_len);
    let mut default_inputs = Vec::with_capacity(nodes_len);
    let mut side_inputs = Vec::with_capacity(nodes_len);

    for idx in graph.node_indices() {
        compile_node(
            &graph,
            idx,
            nodes_len,
            &nodes_map,
            &mut types,
            &mut neighbor_info,
            &mut neighbor_links,
            &mut states,
            &mut default_inputs,
            &mut side_inputs,
        );
    }

    let blocks = graph
        .node_weights()
        .map(|node| node.block.map(|(pos, id)| (pos, Block::from_id(id))))
        .collect_vec();
    let mut pos_map = FxHashMap::default();

    // Create a mapping from block pos to backend node id
    for i in 0..blocks.len() {
        if let Some((pos, _)) = blocks[i] {
            pos_map.insert(pos, i);
        }
    }

    let mut tick_delay = vec![0u32; nodes_len * 4];

    for entry in ticks {
        if let Some(node) = pos_map.get(&entry.pos) {
            use mchprs_world::TickPriority as WTickPriority;
            let priority = match entry.tick_priority {
                WTickPriority::Highest => TickPriority::Highest,
                WTickPriority::Higher => TickPriority::Higher,
                WTickPriority::High => TickPriority::High,
                WTickPriority::Normal => TickPriority::Normal,
            };
            let priority_index = (priority as u32) * nodes_len as u32;
            tick_delay[priority_index as usize + *node] = entry.ticks_left;
        }
    }

    let compilation_data = CompilationData {
        node_count: nodes_len,
        types,
        neighbor_info,
        neighbor_links,
        states,
        default_inputs,
        side_inputs,
        tick_delay: tick_delay.into_boxed_slice(),
    };
    let block_info = CompiledBlockInfo {
        blocks,
        pos_map,
    };

    (compilation_data, block_info)
}
