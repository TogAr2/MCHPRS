mod pipelines;
mod shader_types;
mod compile;

use std::sync::{Arc, Mutex};
use rustc_hash::FxHashMap;
use tokio::runtime::Runtime;
use mchprs_blocks::BlockPos;
use mchprs_blocks::blocks::Block;
use mchprs_world::{TickEntry, World};
use crate::backend::JITBackend;
use crate::compile_graph::{CompileGraph};
use crate::{CompilerOptions, TaskMonitor};
use crate::backend::gpu::pipelines::{CommandCache, CompilationData, Pipelines};

pub struct CompilationResult {
    pipelines: Pipelines,
    data: CompilationData,
}

pub struct GpuBackend {
    runtime: Runtime,
    device: Arc<wgpu::Device>,
    queue: wgpu::Queue,
    pipelines: Option<Arc<Pipelines>>,
    command_cache: Arc<Mutex<CommandCache>>,
    raw_states: Box<[u32]>,
    blocks: Vec<Option<(BlockPos, Block)>>,
    pos_map: FxHashMap<BlockPos, usize>,
}

impl GpuBackend {
    pub fn create() -> Self {
        let runtime = Runtime::new().unwrap();
        let (device, queue) = runtime.block_on(pipelines::open_device());
        Self {
            runtime,
            device: Arc::new(device),
            queue,
            pipelines: None,
            command_cache: Arc::new(Mutex::new(CommandCache::new())),
            raw_states: Box::new([]),
            blocks: Vec::default(),
            pos_map: FxHashMap::default(),
        }
    }
}

impl JITBackend for GpuBackend {
    fn compile(&mut self, graph: CompileGraph, ticks: Vec<TickEntry>, options: &CompilerOptions, monitor: Arc<TaskMonitor>) {
        let (data, block_info) = compile::compile(graph, ticks, options, monitor);
        let pipelines = Arc::new(Pipelines::create(&self.device, data.node_count, data.neighbor_links.len()));
        self.pipelines = Some(pipelines.clone());
        self.raw_states = data.get_raw_states();
        self.blocks = block_info.blocks;
        self.pos_map = block_info.pos_map;

        pipelines.upload_all(data);
        CommandCache::start_renew_thread(self.command_cache.clone(), self.device.clone(), pipelines.clone());
    }

    fn tick(&mut self) {
        self.tickn(1);
    }

    fn tickn(&mut self, ticks: u64) {
        if let Some(pipelines) = self.pipelines.as_mut() {
            pipelines.clone().run_passes(self.device.clone(), &self.queue, ticks, self.command_cache.clone());
        }
    }

    fn on_use_block(&mut self, pos: BlockPos) {
        todo!()
    }

    fn set_pressure_plate(&mut self, pos: BlockPos, powered: bool) {
        todo!()
    }

    fn flush<W: World>(&mut self, world: &mut W, io_only: bool) {
        //todo!()
    }

    fn reset<W: World>(&mut self, world: &mut W, io_only: bool) {
        //todo!()
    }

    fn has_pending_ticks(&self) -> bool {
        todo!()
    }

    fn inspect(&mut self, pos: BlockPos) {
        todo!()
    }
}
