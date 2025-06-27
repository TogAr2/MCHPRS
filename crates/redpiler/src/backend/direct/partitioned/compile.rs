use crate::compile_graph::{CompileGraph, LinkType, NodeIdx};
use crate::{CompilerOptions, TaskMonitor};
use itertools::Itertools;
use mchprs_blocks::blocks::{Block, Instrument};
use mchprs_blocks::BlockPos;
use mchprs_world::TickEntry;
use petgraph::visit::EdgeRef;
use petgraph::Direction;
use rustc_hash::FxHashMap;
use smallvec::SmallVec;
use std::sync::Arc;
use tokio::sync::mpsc::UnboundedSender;
use tracing::trace;
use crate::backend::direct::node::{Node, NodeId, NodeInput, NodeType, Nodes, NonMaxU8};
use crate::backend::direct::partitioned::partition::{NodeLocation, PartitionForwardLink};
use crate::backend::direct::partitioned::{GlobalMessage, Partition, PartitionInterface, PartitionMessage, PartitionedBackend};

#[derive(Debug, Default)]
struct FinalGraphStats {
    update_link_count: usize,
    side_link_count: usize,
    default_link_count: usize,
    nodes_bytes: usize,
}

fn compile_node(
    graph: &CompileGraph,
    node_idx: NodeIdx,
    nodes_map: &FxHashMap<NodeIdx, NodeLocation>,
    noteblock_info: &mut Vec<(BlockPos, Instrument, u32)>,
    stats: &mut FinalGraphStats,
) -> Node {
    let node = &graph[node_idx];

    const MAX_INPUTS: usize = 255;

    let mut default_input_count = 0;
    let mut side_input_count = 0;

    let mut default_inputs = NodeInput { ss_counts: [0; 16] };
    let mut side_inputs = NodeInput { ss_counts: [0; 16] };
    for edge in graph.edges_directed(node_idx, Direction::Incoming) {
        let weight = edge.weight();
        let distance = weight.ss;
        let source = edge.source();
        let ss = graph[source].state.output_strength.saturating_sub(distance);
        match weight.ty {
            LinkType::Default => {
                if default_input_count >= MAX_INPUTS {
                    panic!(
                        "Exceeded the maximum number of default inputs {}",
                        MAX_INPUTS
                    );
                }
                default_input_count += 1;
                default_inputs.ss_counts[ss as usize] += 1;
            }
            LinkType::Side => {
                if side_input_count >= MAX_INPUTS {
                    panic!("Exceeded the maximum number of side inputs {}", MAX_INPUTS);
                }
                side_input_count += 1;
                side_inputs.ss_counts[ss as usize] += 1;
            }
        }
    }
    stats.default_link_count += default_input_count;
    stats.side_link_count += side_input_count;

    use crate::compile_graph::NodeType as CNodeType;
    let updates = if node.ty != CNodeType::Constant {
        graph
            .edges_directed(node_idx, Direction::Outgoing)
            // Sorting by node location allows for optimization when executing
            .sorted_by_key(|edge| nodes_map[&edge.target()])
            .into_group_map_by(|edge| std::mem::discriminant(&graph[edge.target()].ty))
            .into_values()
            .flatten()
            .map(|edge| {
                let idx = edge.target();
                let location = nodes_map[&idx];
                let weight = edge.weight();
                PartitionForwardLink::new(location, weight.ty == LinkType::Side, weight.ss).data()
            })
            .collect()
    } else {
        SmallVec::new()
    };
    stats.update_link_count += updates.len();

    let ty = match &node.ty {
        CNodeType::Repeater {
            delay,
            facing_diode,
        } => NodeType::Repeater {
            delay: *delay,
            facing_diode: *facing_diode,
        },
        CNodeType::Torch => NodeType::Torch,
        CNodeType::Comparator {
            mode,
            far_input,
            facing_diode,
        } => NodeType::Comparator {
            mode: *mode,
            far_input: far_input.map(|value| NonMaxU8::new(value).unwrap()),
            facing_diode: *facing_diode,
        },
        CNodeType::Lamp => NodeType::Lamp,
        CNodeType::Button => NodeType::Button,
        CNodeType::Lever => NodeType::Lever,
        CNodeType::PressurePlate => NodeType::PressurePlate,
        CNodeType::Trapdoor => NodeType::Trapdoor,
        CNodeType::Wire => NodeType::Wire,
        CNodeType::Constant => NodeType::Constant,
        CNodeType::NoteBlock { instrument, note } => {
            let noteblock_id = noteblock_info.len().try_into().unwrap();
            noteblock_info.push((node.block.unwrap().0, *instrument, *note));
            NodeType::NoteBlock { noteblock_id }
        }
    };

    Node {
        ty,
        default_inputs,
        side_inputs,
        updates,
        powered: node.state.powered,
        output_power: node.state.output_strength,
        locked: node.state.repeater_locked,
        pending_tick: false,
        changed: false,
        is_io: node.is_input || node.is_output,
    }
}

pub fn compile(
    backend: &mut PartitionedBackend,
    graph: CompileGraph,
    ticks: Vec<TickEntry>,
    options: &CompilerOptions,
    _monitor: Arc<TaskMonitor>,
) -> Vec<Partition> {
    let partition_size = graph.node_count()
        .div_ceil(PartitionedBackend::MAX_PARTITIONS)
        .max(PartitionedBackend::MIN_PARTITION_SIZE);
    let partition_count = graph.node_count().div_ceil(partition_size);

    backend.blocks.clear();

    // Create a mapping from compile to backend node locations
    let mut nodes_map = FxHashMap::with_capacity_and_hasher(graph.node_count(), Default::default());
    for (i, node) in graph.node_indices().enumerate() {
        let partition = i / partition_size;
        let node_id = i % partition_size;
        let location = unsafe {
            // Safety: we know it will be a valid location given the context
            NodeLocation::from(partition, node_id)
        };
        nodes_map.insert(node, location);

        let block = graph[node].block.map(|(pos, id)| (pos, Block::from_id(id)));
        if let Some(block) = block {
            backend.blocks.insert(location, block);
        }
    }
    let nodes_len = nodes_map.len();

    // Lower nodes
    let mut stats = FinalGraphStats::default();

    let (
        mut global_tx,
        mut global_rx,
        mut partition_tx,
        mut partition_rx
    ) = (Vec::new(), Vec::new(), Vec::new(), Vec::new());
    for _ in 0..partition_count {
        let global = tokio::sync::mpsc::unbounded_channel::<GlobalMessage>();
        let partition = tokio::sync::mpsc::unbounded_channel::<PartitionMessage>();
        global_tx.push(global.0);
        global_rx.push(global.1);
        partition_tx.push(partition.0);
        partition_rx.push(Some(partition.1));
    }
    let partition_tx: Arc<[UnboundedSender<PartitionMessage>]> = Arc::from(partition_tx);

    let mut partitions = Vec::with_capacity(partition_count);
    let mut curr_nodes: Vec<Node> = Vec::with_capacity(partition_size);
    for (i, node) in graph.node_indices().enumerate() {
        curr_nodes.push(compile_node(
            &graph,
            node,
            &nodes_map,
            &mut backend.noteblock_info,
            &mut stats,
        ));

        if curr_nodes.len() == partition_size || i == graph.node_count() - 1 {
            let partition_idx = i / partition_size;
            let mut temp = Vec::with_capacity(partition_size);
            std::mem::swap(&mut temp, &mut curr_nodes);
            let nodes = Nodes::new(temp.into_boxed_slice());

            partitions.push(Partition::new(
                partition_idx,
                nodes,
                global_tx[partition_idx].clone(),
                partition_tx.clone(),
                partition_rx[partition_idx].take().unwrap()
            ));
        }
    }

    backend.partitions = global_rx.into_iter().zip(partition_tx.iter()).zip(partitions.iter())
        .map(|((global_rx, partition_tx), partition)| {
            PartitionInterface::new(partition_tx.clone(), global_rx, partition.nodes.clone())
        })
        .collect();

    stats.nodes_bytes = nodes_len * std::mem::size_of::<Node>();
    trace!("{:#?}", stats);

    backend.pos_map.clear();
    // Create a mapping from block pos to backend NodeLocation
    for (location, block) in &backend.blocks {
        backend.pos_map.insert(block.0, *location);
    }

    // Schedule backend ticks
    for entry in ticks {
        if let Some(node) = backend.pos_map.get(&entry.pos) {
            backend.cross_schedule_tick(*node, entry.ticks_left as usize, entry.tick_priority);
        }
    }

    // Dot file output
    if options.export_dot_graph {
        std::fs::write("backend_graph.dot", format!("{}", backend)).unwrap();
    }

    partitions
}
