use std::default::Default;
use std::fmt;
use std::future::Future;
use std::sync::Arc;
use std::time::Instant;
use futures::future::JoinAll;
use rustc_hash::FxHashMap;
use tokio::runtime::{Handle, Runtime};
use tokio::sync::mpsc::{UnboundedReceiver, UnboundedSender};
use tracing::{debug, warn};
use mchprs_blocks::block_entities::BlockEntity;
use mchprs_blocks::BlockPos;
use mchprs_blocks::blocks::{Block, ComparatorMode, Instrument};
use mchprs_redstone::{bool_to_ss, noteblock};
use mchprs_world::{TickEntry, TickPriority, World};
use crate::backend::direct::{Event, Queues, TickScheduler};
use crate::backend::direct::node::{ForwardLink, NodeId, NodeType, Nodes};
use crate::backend::direct::partitioned::partition::{NodeLocation, PartitionForwardLink};
use crate::backend::direct::update::update_node;
use crate::backend::JITBackend;
use crate::{block_powered_mut, CompilerOptions, TaskMonitor};
use crate::compile_graph::CompileGraph;

mod partition;
mod compile;
mod tick;

enum PartitionMessage {
    Tick(TickPriority),
    Update {
        node_id: NodeId,
        side: bool,
        old_power: u8,
        new_power: u8,
    },
    ScheduleTick {
        node_id: NodeId,
        delay: usize,
        priority: TickPriority,
    },
    SetNode {
        node_id: NodeId,
        powered: bool,
        new_power: u8,
    },
    RequestTickEntries(Box<[Option<(BlockPos, Block)>]>),
}

enum GlobalMessage {
    TickEnd(Nodes, Vec<Event>),
    TickEntries(Vec<TickEntry>),
}

pub struct PartitionInterface {
    sender: UnboundedSender<PartitionMessage>,
    receiver: UnboundedReceiver<GlobalMessage>,
    nodes: Nodes,
    events: Vec<Event>,
}

impl PartitionInterface {
    fn new(sender: UnboundedSender<PartitionMessage>, receiver: UnboundedReceiver<GlobalMessage>, nodes: Nodes) -> Self {
        PartitionInterface {
            sender,
            receiver,
            nodes,
            events: Default::default(),
        }
    }

    fn send(&self, message: PartitionMessage) {
        self.sender.send(message).unwrap();
    }

    async fn wait_tick_end(&mut self) {
        match self.receiver.recv().await.unwrap() {
            GlobalMessage::TickEnd(nodes, events) => {
                self.nodes = nodes;
                self.events = events;
            }
            _ => warn!("Unexpected global message while waiting for tick end"),
        }
    }

    async fn wait_tick_entries(&mut self) -> Vec<TickEntry> {
        match self.receiver.recv().await.unwrap() {
            GlobalMessage::TickEntries(entries) => entries,
            _ => {
                warn!("Unexpected global message while waiting for tick entries");
                Vec::new()
            }
        }
    }

    fn local_nodes(&mut self) -> &mut Nodes {
        &mut self.nodes
    }

    fn local_events(&mut self) -> &mut Vec<Event> {
        &mut self.events
    }
}

pub struct Partition {
    index: usize,
    nodes: Nodes,
    scheduler: TickScheduler,
    events: Vec<Event>,
    sender: UnboundedSender<GlobalMessage>,
    cross_senders: Arc<[UnboundedSender<PartitionMessage>]>,
    receiver: UnboundedReceiver<PartitionMessage>,
    curr_queues: Queues,
}

impl Partition {
    fn new(
        index: usize,
        nodes: Nodes,
        sender: UnboundedSender<GlobalMessage>,
        cross_senders: Arc<[UnboundedSender<PartitionMessage>]>,
        receiver: UnboundedReceiver<PartitionMessage>,
    ) -> Self {
        Self {
            index,
            nodes,
            scheduler: Default::default(),
            events: Default::default(),
            sender,
            cross_senders,
            receiver,
            curr_queues: Default::default(),
        }
    }

    fn handle_message(&mut self, message: PartitionMessage) {
        match message {
            PartitionMessage::Tick(priority) => {
                println!("Part {}: Received tick start", self.index);
                let instant = Instant::now();
                self.tick_with_priority(priority);
                let elapsed = instant.elapsed();
                println!("Part {}: {elapsed:?} elapsed before send", self.index);
                self.sender.send(GlobalMessage::TickEnd(
                    self.nodes.clone(),
                    self.events.clone(),
                )).unwrap();
                let elapsed = instant.elapsed();
                println!("Part {}: {elapsed:?} elapsed after send", self.index);
            },
            PartitionMessage::Update { node_id, side, old_power, new_power } => {
                //println!("Part {partition_idx}: Received update: {node_id:?}");
                self.update(node_id, side, old_power, new_power);
            },
            PartitionMessage::ScheduleTick { node_id, delay, priority } => {
                self.scheduler.schedule_tick(node_id, delay, priority);
                self.nodes[node_id].pending_tick = true;
            }
            PartitionMessage::SetNode { node_id, powered, new_power } => {
                self.set_node(node_id, powered, new_power);
            }
            PartitionMessage::RequestTickEntries(blocks) => {
                self.sender.send(GlobalMessage::TickEntries(
                    self.scheduler.reset_get_tick_entries(&blocks)
                )).unwrap();
            }
        }
    }

    fn start_tick(&mut self) {
        self.curr_queues = self.scheduler.queues_this_tick();
    }

    fn tick_with_priority(&mut self, priority: TickPriority) {
        if priority == TickPriority::Highest {
            self.start_tick();
        }
        unsafe {
            // Safety: priority will always fit in queues count
            for node_id in self.curr_queues.0.get_unchecked(priority as usize).clone() {
                self.tick_node(node_id);
            }
        }
        if priority == TickPriority::Normal {
            self.end_tick();
        }
    }

    fn end_tick(&mut self) {
        self.scheduler.end_tick(std::mem::take(&mut self.curr_queues));
    }

    fn update(&mut self, node_id: NodeId, side: bool, old_power: u8, new_power: u8) {
        //println!("Part {}: Update {:?}, old_power: {old_power}, new_power: {new_power}", self.index, node_id);

        let update_ref = &mut self.nodes[node_id];
        let inputs = if side {
            &mut update_ref.side_inputs
        } else {
            &mut update_ref.default_inputs
        };

        // Safety: signal strength is never larger than 15
        unsafe {
            *inputs.ss_counts.get_unchecked_mut(old_power as usize) -= 1;
            *inputs.ss_counts.get_unchecked_mut(new_power as usize) += 1;
        }

        update_node(
            &mut self.scheduler,
            &mut self.events,
            &mut self.nodes,
            node_id,
        );
    }

    fn set_node(&mut self, node_id: NodeId, powered: bool, new_power: u8) {
        //println!("Part {}: Set node {:?}, powered: {powered}, new_power: {new_power}", self.index, node_id);
        let node = &mut self.nodes[node_id];
        let old_power = node.output_power;

        node.changed = true;
        node.powered = powered;
        node.output_power = new_power;
        for i in 0..node.updates.len() {
            let node = &self.nodes[node_id];
            let link_data = unsafe { *node.updates.get_unchecked(i) };
            let update_link = PartitionForwardLink::from(link_data);
            let side = update_link.side();
            let distance = update_link.ss();
            let update = update_link.node();

            let old_power = old_power.saturating_sub(distance);
            let new_power = new_power.saturating_sub(distance);

            if old_power == new_power {
                continue;
            }

            if update.partition() == self.index {
                // Same partition, it can be handled immediately
                self.update(update.index(), side, old_power, new_power);
            } else {
                // Request an update from a different partition
                unsafe {
                    // Safety: NodeLocation is always valid
                    self.cross_senders.get_unchecked(update.partition()).send(PartitionMessage::Update {
                        node_id: update.index(),
                        side,
                        old_power,
                        new_power,
                    }).unwrap();
                }
            }
        }
    }
}

pub struct PartitionedBackend {
    partitions: Box<[PartitionInterface]>,
    blocks: FxHashMap<NodeLocation, (BlockPos, Block)>,
    pos_map: FxHashMap<BlockPos, NodeLocation>,
    noteblock_info: Vec<(BlockPos, Instrument, u32)>,
    runtime: Runtime,
}

impl Default for PartitionedBackend {
    fn default() -> Self {
        Self {
            partitions: Default::default(),
            blocks: Default::default(),
            pos_map: Default::default(),
            noteblock_info: Default::default(),
            runtime: Runtime::new().unwrap(),
        }
    }
}

impl PartitionedBackend {
    const MAX_PARTITIONS: usize = 2usize.pow(7);
    const MIN_PARTITION_SIZE: usize = 2usize.pow(15);

    fn cross_schedule_tick(&mut self, location: NodeLocation, delay: usize, priority: TickPriority) {
        unsafe {
            // Safety: NodeLocation is always valid
            let partition = self.partitions.get_unchecked(location.partition());
            partition.send(PartitionMessage::ScheduleTick {
                node_id: location.index(),
                delay,
                priority,
            });
        }
    }

    fn tick_with_priority(&mut self, priority: TickPriority) {
        //println!("Tick: Begin tick {priority:?}");
        for partition in self.partitions.iter() {
            partition.send(PartitionMessage::Tick(priority));
        }
        //println!("Tick: Done sending tick messages");
        let mut futures = Vec::with_capacity(self.partitions.len());
        for partition in self.partitions.iter_mut() {
            //println!("Tick: Waiting for partition {idx}");
            futures.push(partition.wait_tick_end());
        }
        self.runtime.block_on(futures::future::join_all(futures));

        //println!("Tick: End tick {priority:?}");
    }
}

impl JITBackend for PartitionedBackend {
    fn inspect(&mut self, pos: BlockPos) {
        let Some(location) = self.pos_map.get(&pos) else {
            debug!("could not find node at pos {}", pos);
            return;
        };

        let partition = &self.partitions[location.partition()];
        debug!("Node {:?}: {:#?}", location, partition.nodes[location.index()]);
    }

    fn reset<W: World>(&mut self, world: &mut W, io_only: bool) {
        let partitions = std::mem::take(&mut self.partitions);

        for (partition_idx, partition) in partitions.iter().enumerate() {
            let nodes = &partition.nodes;

            // Create a vec of blocks from node id, needed for tick scheduler reset
            let mut block_vec = Vec::with_capacity(nodes.inner().len());

            for (node_idx, node) in nodes.inner().iter().enumerate() {
                let location = unsafe { NodeLocation::from(partition_idx, node_idx) };
                let block = self.blocks.get(&location).copied();
                block_vec.push(block);
                let Some((pos, block)) = block else {
                    continue;
                };
                if matches!(node.ty, NodeType::Comparator { .. }) {
                    let block_entity = BlockEntity::Comparator {
                        output_strength: node.output_power,
                    };
                    world.set_block_entity(pos, block_entity);
                }

                if io_only && !node.is_io {
                    world.set_block(pos, block);
                }
            }

            partition.send(PartitionMessage::RequestTickEntries(block_vec.into()));
        }

        for mut partition in partitions {
            let tick_entries = self.runtime.block_on(partition.wait_tick_entries());
            for entry in tick_entries {
                world.schedule_tick(entry.pos, entry.ticks_left, entry.tick_priority);
            }
        }

        self.pos_map.clear();
        self.noteblock_info.clear();
        self.blocks.clear();
    }

    fn on_use_block(&mut self, pos: BlockPos) {
        let location = self.pos_map[&pos];
        let node = &self.partitions[location.partition()].nodes[location.index()];

        match node.ty {
            NodeType::Button => {
                if node.powered {
                    return;
                }
                let partition = &self.partitions[location.partition()];
                partition.send(PartitionMessage::ScheduleTick {
                    node_id: location.index(),
                    delay: 10,
                    priority: TickPriority::Normal,
                });
                partition.send(PartitionMessage::SetNode {
                    node_id: location.index(),
                    powered: true,
                    new_power: 15,
                });
            }
            NodeType::Lever => {
                let partition = &self.partitions[location.partition()];
                partition.send(PartitionMessage::SetNode {
                    node_id: location.index(),
                    powered: !node.powered,
                    new_power: bool_to_ss(!node.powered),
                });
            }
            _ => warn!("Tried to use a {:?} redpiler node", node.ty),
        }
    }

    fn set_pressure_plate(&mut self, pos: BlockPos, powered: bool) {
        let location = self.pos_map[&pos];
        let partition = &self.partitions[location.partition()];
        let node = &partition.nodes[location.index()];

        match node.ty {
            NodeType::PressurePlate => {
                partition.send(PartitionMessage::SetNode {
                    node_id: location.index(),
                    powered,
                    new_power: bool_to_ss(powered),
                });
            }
            _ => warn!("Tried to set pressure plate state for a {:?}", node.ty),
        }
    }

    fn tick(&mut self) {
        self.tick_with_priority(TickPriority::Highest);
        self.tick_with_priority(TickPriority::Higher);
        self.tick_with_priority(TickPriority::High);
        self.tick_with_priority(TickPriority::Normal);
    }

    fn flush<W: World>(&mut self, world: &mut W, io_only: bool) {
        for (partition_id, partition) in self.partitions.iter_mut().enumerate() {
            for event in partition.local_events().drain(..) {
                match event {
                    Event::NoteBlockPlay { noteblock_id } => {
                        let (pos, instrument, note) = self.noteblock_info[noteblock_id as usize];
                        noteblock::play_note(world, pos, instrument, note);
                    }
                }
            }
            for (i, node) in partition.local_nodes().inner_mut().iter_mut().enumerate() {
                let location = unsafe { NodeLocation::from(partition_id, i) };
                let Some((pos, block)) = self.blocks.get_mut(&location) else {
                    continue;
                };
                if node.changed && (!io_only || node.is_io) {
                    if let Some(powered) = block_powered_mut(block) {
                        *powered = node.powered
                    }
                    if let Block::RedstoneWire { wire, .. } = block {
                        wire.power = node.output_power
                    };
                    if let Block::RedstoneRepeater { repeater } = block {
                        repeater.locked = node.locked;
                    }
                    world.set_block(*pos, *block);
                }
                node.changed = false; //TODO this should also be done on partition side
            }
        }
    }

    fn compile(
        &mut self,
        graph: CompileGraph,
        ticks: Vec<TickEntry>,
        options: &CompilerOptions,
        monitor: Arc<TaskMonitor>,
    ) {
        let partitions = compile::compile(self, graph, ticks, options, monitor);

        for mut partition in partitions {
            self.runtime.spawn_blocking(move || {
                loop {
                    match Handle::current().block_on(partition.receiver.recv()) {
                        Some(message) => partition.handle_message(message),
                        None => {
                            break; //TODO this will most likely not happen
                        }
                    }
                }
            });
        }
    }

    fn has_pending_ticks(&self) -> bool {
        todo!();
    }
}

impl fmt::Display for PartitionedBackend {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "digraph {{")?;
        for (partition_id, partition) in self.partitions.iter().enumerate() {
            for (node_id, node) in partition.nodes.inner().iter().enumerate() {
                if matches!(node.ty, NodeType::Wire) {
                    continue;
                }
                let label = match node.ty {
                    NodeType::Repeater { delay, .. } => format!("Repeater({})", delay),
                    NodeType::Torch => format!("Torch"),
                    NodeType::Comparator { mode, .. } => format!(
                        "Comparator({})",
                        match mode {
                            ComparatorMode::Compare => "Cmp",
                            ComparatorMode::Subtract => "Sub",
                        }
                    ),
                    NodeType::Lamp => format!("Lamp"),
                    NodeType::Button => format!("Button"),
                    NodeType::Lever => format!("Lever"),
                    NodeType::PressurePlate => format!("PressurePlate"),
                    NodeType::Trapdoor => format!("Trapdoor"),
                    NodeType::Wire => format!("Wire"),
                    NodeType::Constant => format!("Constant({})", node.output_power),
                    NodeType::NoteBlock { .. } => format!("NoteBlock"),
                };
                let pos = unsafe {
                    if let Some((pos, _)) = self.blocks.get(&NodeLocation::from(partition_id, node_id)) {
                        format!("{}, {}, {}", pos.x, pos.y, pos.z)
                    } else {
                        "No Pos".to_string()
                    }
                };
                writeln!(f, "    p{}n{} [ label = \"{}\\n({})\" ];", partition_id, node_id, label, pos)?;
                for link in node.updates.iter() {
                    let link = PartitionForwardLink::from(*link);
                    let out_partition = link.node().partition();
                    let out_index = link.node().index().index();
                    let distance = link.ss();
                    let color = if link.side() { ",color=\"blue\"" } else { "" };
                    writeln!(
                        f,
                        "    p{}n{} -> p{}n{} [ label = \"{}\"{} ];",
                        partition_id, node_id, out_partition, out_index, distance, color
                    )?;
                }
            }
        }
        writeln!(f, "}}")
    }
}
