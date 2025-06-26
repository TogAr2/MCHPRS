use crate::backend::gpu::shader_types::{ForwardLink, NeighborInfo, State, TickPriority, TypeInfo};
use itertools::Itertools;
use std::sync::{Arc, Mutex};
use tokio::runtime::{Handle, Runtime};
use tokio::sync::mpsc;
use wgpu::util::DeviceExt;

pub async fn open_device() -> (wgpu::Device, wgpu::Queue) {
    let instance = wgpu::Instance::default();
    let adapter = instance.request_adapter(&wgpu::RequestAdapterOptions {
        power_preference: wgpu::PowerPreference::HighPerformance,
        force_fallback_adapter: false,
        compatible_surface: None,
    }).await.unwrap();
    adapter.request_device(&wgpu::DeviceDescriptor {
        label: None,
        required_features: wgpu::Features::empty(),
        required_limits: wgpu::Limits::default(),
        memory_hints: wgpu::MemoryHints::Performance,
        trace: wgpu::Trace::Off,
    }).await.unwrap()
}

#[derive(Debug)]
pub struct CompilationData {
    pub node_count: usize,
    pub types: Vec<TypeInfo>,
    pub neighbor_info: Vec<NeighborInfo>,
    pub neighbor_links: Vec<ForwardLink>,
    pub states: Vec<State>,
    pub default_inputs: Vec<[u32; 16]>,
    pub side_inputs: Vec<[u32; 16]>,
    pub tick_delay: Box<[u32]>,
}

pub struct RawCompilationData {
    pub types: Box<[u32]>,
    pub neighbor_info: Box<[u32]>,
    pub neighbor_links: Box<[u32]>,
    pub states: Box<[u32]>,
    pub default_inputs: Box<[u32]>,
    pub side_inputs: Box<[u32]>,
    pub tick_delay: Box<[u32]>,
}

impl CompilationData {
    pub fn to_raw(self) -> RawCompilationData {
        let types_vec = self.types
            .iter()
            .map(|x| x.as_packed())
            .collect_vec();
        let neighbor_info_vec = self.neighbor_info
            .iter()
            .map(|x| x.as_packed())
            .collect_vec();
        let neighbor_links_vec = self.neighbor_links
            .iter()
            .map(|x| x.as_packed())
            .collect_vec();
        let default_inputs_vec = self.default_inputs
            .iter()
            .flat_map(|x| x.iter())
            .copied()
            .collect_vec();
        let side_inputs_vec = self.side_inputs
            .iter()
            .flat_map(|x| x.iter())
            .copied()
            .collect_vec();

        RawCompilationData {
            types: types_vec.into_boxed_slice(),
            neighbor_info: neighbor_info_vec.into_boxed_slice(),
            neighbor_links: neighbor_links_vec.into_boxed_slice(),
            states: self.get_raw_states(),
            default_inputs: default_inputs_vec.into_boxed_slice(),
            side_inputs: side_inputs_vec.into_boxed_slice(),
            tick_delay: self.tick_delay,
        }
    }

    pub fn get_raw_states(&self) -> Box<[u32]> {
        self.states
            .iter()
            .map(|x| x.as_packed())
            .collect_vec()
            .into_boxed_slice()
    }
}

struct ConstantStorage {
    layout: wgpu::BindGroupLayout,
    types: wgpu::Buffer,
    neighbor_info: wgpu::Buffer,
    neighbor_links: wgpu::Buffer,
}

impl ConstantStorage {
    fn new(device: &wgpu::Device, size: wgpu::BufferAddress, link_buffer_size: wgpu::BufferAddress) -> Self {
        Self {
            layout: Self::create_layout(device),
            types: create_storage_buffer(device, "types", size, false, false),
            neighbor_info: create_storage_buffer(device, "neighbor_info", size, false, false),
            neighbor_links: create_storage_buffer(device, "neighbor_links", link_buffer_size, false, false),
        }
    }

    fn create_layout(device: &wgpu::Device) -> wgpu::BindGroupLayout {
        device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Constant layout"),
            entries: &[
                buffer_layout_entry(0, true),
                buffer_layout_entry(1, true),
                buffer_layout_entry(2, true),
            ],
        })
    }

    fn create_bind_group(&self, device: &wgpu::Device) -> wgpu::BindGroup {
        device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Constant bind group"),
            layout: &self.layout,
            entries: &[
                buffer_group_entry(0, &self.types),
                buffer_group_entry(1, &self.neighbor_info),
                buffer_group_entry(2, &self.neighbor_links),
            ],
        })
    }
}

struct CommonStorage {
    layout: wgpu::BindGroupLayout,
    states: wgpu::Buffer,
    default_inputs_in: wgpu::Buffer,
    side_inputs_in: wgpu::Buffer,
    tick_delay_in: wgpu::Buffer,
    tick_delay_out: wgpu::Buffer,
}

impl CommonStorage {
    fn new(device: &wgpu::Device, size: wgpu::BufferAddress) -> Self {
        Self {
            layout: Self::create_layout(device),
            states: create_storage_buffer(device, "states", size, true, true),
            default_inputs_in: create_storage_buffer(device, "default_inputs_in", size * 16, true, false),
            side_inputs_in: create_storage_buffer(device, "side_inputs_in", size * 16, true, false),
            tick_delay_in: create_storage_buffer(device, "tick_delay_in", size * 4, true, false),
            tick_delay_out: create_storage_buffer(device, "tick_delay_out", size * 4, false, true),
        }
    }

    fn create_layout(device: &wgpu::Device) -> wgpu::BindGroupLayout {
        device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Common layout"),
            entries: &[
                buffer_layout_entry(0, false),
                buffer_layout_entry(1, true),
                buffer_layout_entry(2, true),
                buffer_layout_entry(3, true),
                buffer_layout_entry(4, false),
            ],
        })
    }

    fn create_bind_group(&self, device: &wgpu::Device) -> wgpu::BindGroup {
        device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Common bind group"),
            layout: &self.layout,
            entries: &[
                buffer_group_entry(0, &self.states),
                buffer_group_entry(1, &self.default_inputs_in),
                buffer_group_entry(2, &self.side_inputs_in),
                buffer_group_entry(3, &self.tick_delay_in),
                buffer_group_entry(4, &self.tick_delay_out),
            ],
        })
    }
}

struct TickStorage {
    layout: wgpu::BindGroupLayout,
    default_inputs_out: wgpu::Buffer,
    side_inputs_out: wgpu::Buffer,
    tick_priority: wgpu::Buffer,
    tick_priorities_cache: wgpu::Buffer,
}

impl TickStorage {
    fn new(device: &wgpu::Device, size: wgpu::BufferAddress) -> Self {
        Self {
            layout: Self::create_layout(device),
            default_inputs_out: create_storage_buffer(device, "default_inputs_out", size * 16, false, true),
            side_inputs_out: create_storage_buffer(device, "side_inputs_out", size * 16, false, true),
            tick_priority: device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("tick_priority"),
                size: 16 as wgpu::BufferAddress,
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: true,
            }),
            tick_priorities_cache: device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("tick_priorities_cache"),
                contents: bytemuck::cast_slice(&[
                    TickPriority::Highest as u32,
                    TickPriority::Higher as u32,
                    TickPriority::High as u32,
                    TickPriority::Normal as u32,
                ]),
                usage: wgpu::BufferUsages::COPY_SRC,
            }),
        }
    }

    fn create_layout(device: &wgpu::Device) -> wgpu::BindGroupLayout {
        device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Tick layout"),
            entries: &[
                buffer_layout_entry(0, false),
                buffer_layout_entry(1, false),
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                }
            ],
        })
    }

    fn create_bind_group(&self, device: &wgpu::Device) -> wgpu::BindGroup {
        device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Tick bind group"),
            layout: &self.layout,
            entries: &[
                buffer_group_entry(0, &self.default_inputs_out),
                buffer_group_entry(1, &self.side_inputs_out),
                buffer_group_entry(2, &self.tick_priority),
            ],
        })
    }
}

fn buffer_layout_entry(binding: u32, read_only: bool) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Storage { read_only },
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}

fn buffer_group_entry(binding: u32, buffer: &wgpu::Buffer) -> wgpu::BindGroupEntry {
    wgpu::BindGroupEntry {
        binding,
        resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
            buffer,
            offset: 0,
            size: None,
        }),
    }
}

fn create_storage_buffer(
    device: &wgpu::Device,
    label: &str,
    size: wgpu::BufferAddress,
    copy_dst: bool,
    copy_src: bool
) -> wgpu::Buffer {
    let mut usage = wgpu::BufferUsages::STORAGE;
    if copy_dst {
        usage |= wgpu::BufferUsages::COPY_DST;
    }
    if copy_src {
        usage |= wgpu::BufferUsages::COPY_SRC;
    }

    device.create_buffer(&wgpu::BufferDescriptor {
        label: Some(label),
        size,
        usage,
        mapped_at_creation: true,
    })
}

pub struct Pipelines {
    node_count: usize,
    buffer_size: wgpu::BufferAddress,
    update_pipeline: wgpu::ComputePipeline,
    tick_pipeline: wgpu::ComputePipeline,
    constant_storage: ConstantStorage,
    constant_bind_group: wgpu::BindGroup,
    common_storage: CommonStorage,
    common_bind_group: wgpu::BindGroup,
    tick_storage: TickStorage,
    tick_bind_group: wgpu::BindGroup,
    upload_buffer: wgpu::Buffer,
    download_buffer: wgpu::Buffer,
}

pub struct CommandCache {
    cache: Vec<wgpu::CommandBuffer>,
}

impl CommandCache {
    pub fn new() -> Self {
        Self {
            cache: Vec::new()
        }
    }

    pub fn start_renew_thread(this: Arc<Mutex<CommandCache>>, device: Arc<wgpu::Device>, pipelines: Arc<Pipelines>) {
        std::thread::spawn(move || {
            let runtime = Runtime::new().unwrap();
            loop {
                runtime.block_on(Self::renew(this.clone(), device.clone(), pipelines.clone(), &runtime));
            }
        });
    }

    async fn renew(this: Arc<Mutex<CommandCache>>, device: Arc<wgpu::Device>, pipelines: Arc<Pipelines>, runtime: &Runtime) {
        const THREAD_COUNT: usize = 10;
        const BATCH_SIZE: usize = 1;

        //println!("Renewing command cache...");
        let instant = std::time::Instant::now();

        if let Ok(this) = this.lock() {
            if this.cache.len() > 10 {
                return;
            }
        }
        for i in 0..THREAD_COUNT {
            //println!("LOOP {}", i);
            let buffer = pipelines.clone().create_command_buffer(device.clone(), BATCH_SIZE as u64).await;
            if let Ok(mut this) = this.lock() {
                this.cache.push(buffer);
            }
        }

        //println!("Renewed command cache in {:?}", instant.elapsed());
    }

    pub fn pop(this: Arc<Mutex<CommandCache>>) -> wgpu::CommandBuffer {
        loop {
            match this.lock() {
                Ok(mut this) => {
                    if !this.cache.is_empty() {
                        return this.cache.pop().unwrap();
                    }
                }
                Err(error) => panic!("{:?}", error),
            }
        }
    }
}

impl Pipelines {
    pub fn create(device: &wgpu::Device, node_count: usize, neighbor_link_count: usize) -> Self {
        let unpadded_size = (node_count * 4) as wgpu::BufferAddress; // u32 is 4 bytes
        let unpadded_link_size = (neighbor_link_count * 4) as wgpu::BufferAddress;
        // Make buffer sizes a multiple of 8
        let buffer_size = ((unpadded_size + 7) & !7).max(8);
        let link_buffer_size = ((unpadded_link_size + 7) & !7).max(8);

        let shader_module = device.create_shader_module(wgpu::include_wgsl!("redstone.wgsl"));

        let constant_storage = ConstantStorage::new(device, buffer_size, link_buffer_size);
        let constant_bind_group = constant_storage.create_bind_group(device);

        let common_storage = CommonStorage::new(device, buffer_size);
        let common_bind_group = common_storage.create_bind_group(device);

        let tick_storage = TickStorage::new(device, buffer_size);
        let tick_bind_group = tick_storage.create_bind_group(device);

        let update_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Update pipeline layout"),
            bind_group_layouts: &[
                &constant_storage.layout,
                &common_storage.layout,
            ],
            push_constant_ranges: &[],
        });

        let tick_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Tick pipeline layout"),
            bind_group_layouts: &[
                &constant_storage.layout,
                &common_storage.layout,
                &tick_storage.layout,
            ],
            push_constant_ranges: &[],
        });

        let update_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Update pipeline"),
            layout: Some(&update_pipeline_layout),
            module: &shader_module,
            entry_point: Some("update"),
            compilation_options: Default::default(),
            cache: None,
        });

        let tick_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Tick pipeline"),
            layout: Some(&tick_pipeline_layout),
            module: &shader_module,
            entry_point: Some("tick"),
            compilation_options: Default::default(),
            cache: None,
        });

        let upload_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Upload buffer"),
            size: buffer_size,
            usage: wgpu::BufferUsages::MAP_WRITE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let download_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Download buffer"),
            size: buffer_size,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        Self {
            node_count,
            buffer_size,
            update_pipeline,
            tick_pipeline,
            constant_storage,
            constant_bind_group,
            common_storage,
            common_bind_group,
            tick_storage,
            tick_bind_group,
            upload_buffer,
            download_buffer,
        }
    }

    async fn map_buffer(&self, device: &wgpu::Device, buffer: &wgpu::Buffer, mode: wgpu::MapMode) {
        let (sender, mut receiver) = mpsc::channel(1);
        buffer.slice(..).map_async(mode, move |result| {
            Handle::current().block_on(sender.send(result)).unwrap();
        });
        device.poll(wgpu::PollType::Poll).unwrap();
        receiver.recv().await.unwrap().expect("Failed to map buffer");
    }

    async fn write_buffer(&self, device: &wgpu::Device, buffer: &wgpu::Buffer, data: &[u8]) {
        self.map_buffer(device, buffer, wgpu::MapMode::Write).await;
        buffer.get_mapped_range_mut(..)
            .copy_from_slice(&data[..]);
        buffer.unmap();
    }

    async fn read_buffer(&self, device: &wgpu::Device, buffer: &wgpu::Buffer, output: &mut [u8]) {
        self.map_buffer(device, buffer, wgpu::MapMode::Read).await;
        output.copy_from_slice(&buffer.get_mapped_range(..));
        buffer.unmap();
    }

    /// Uses the upload buffer to upload data to the GPU.
    /// Returns when the operation is fully performed.
    async fn upload(&self, device: &wgpu::Device, queue: &wgpu::Queue, buffer: &wgpu::Buffer, data: &[u8]) {
        self.write_buffer(device, &self.upload_buffer, data).await;
        let mut command_encoder = device.create_command_encoder(&Default::default());
        command_encoder.copy_buffer_to_buffer(
            &self.upload_buffer,
            0,
            buffer,
            0,
            data.len() as wgpu::BufferAddress,
        );
        queue.submit(Some(command_encoder.finish()));
        device.poll(wgpu::PollType::Wait).unwrap();
    }

    /// Uses the download buffer to download data from the GPU.
    /// Returns when the operation is fully performed.
    async fn download(&self, device: &wgpu::Device, queue: &wgpu::Queue, buffer: &wgpu::Buffer, output: &mut [u8]) {
        let mut command_encoder = device.create_command_encoder(&Default::default());
        command_encoder.copy_buffer_to_buffer(
            buffer,
            0,
            &self.download_buffer,
            0,
            buffer.size() as wgpu::BufferAddress,
        );
        queue.submit(Some(command_encoder.finish()));
        device.poll(wgpu::PollType::Wait).unwrap();
        self.read_buffer(device, buffer, output).await;
    }

    fn store_initial(&self, buffer: &wgpu::Buffer, data: &[u8]) {
        let data_len = data.len() as wgpu::BufferAddress;

        let mut vec = Vec::from(data);
        if buffer.size() < data_len {
            panic!("Data bigger than buffer!");
        } else if buffer.size() > data_len {
            // Fill with a value indicating to the shader that this shouldn't be operated on
            vec.extend(std::iter::repeat(0xFF).take((buffer.size() - data_len) as usize));
        }
        let data: &[u8] = &vec;

        buffer.get_mapped_range_mut(..).copy_from_slice(data);
        buffer.unmap();
    }

    /// Populates all the buffers with initial data
    pub fn upload_all(&self, compilation_data: CompilationData) {
        let raw_data = compilation_data.to_raw();

        self.store_initial(&self.constant_storage.types, bytemuck::cast_slice(&raw_data.types));
        self.store_initial(&self.constant_storage.neighbor_info, bytemuck::cast_slice(&raw_data.neighbor_info));
        self.store_initial(&self.constant_storage.neighbor_links, bytemuck::cast_slice(&raw_data.neighbor_links));

        self.store_initial(&self.common_storage.states, bytemuck::cast_slice(&raw_data.states));
        self.store_initial(&self.common_storage.default_inputs_in, bytemuck::cast_slice(&raw_data.default_inputs));
        self.store_initial(&self.common_storage.side_inputs_in, bytemuck::cast_slice(&raw_data.side_inputs));

        self.store_initial(&self.common_storage.tick_delay_in, bytemuck::cast_slice(&raw_data.tick_delay));
        self.store_initial(&self.common_storage.tick_delay_out, bytemuck::cast_slice(&raw_data.tick_delay));

        self.store_initial(&self.tick_storage.default_inputs_out, bytemuck::cast_slice(&raw_data.default_inputs));
        self.store_initial(&self.tick_storage.side_inputs_out, bytemuck::cast_slice(&raw_data.side_inputs));
        self.store_initial(&self.tick_storage.tick_priority, bytemuck::cast_slice(&[0u32]));
    }

    fn copy_inputs_out_to_in(&self, command_encoder: &mut wgpu::CommandEncoder) {
        command_encoder.copy_buffer_to_buffer(
            &self.tick_storage.default_inputs_out,
            0,
            &self.common_storage.default_inputs_in,
            0,
            self.buffer_size * 16,
        );
        command_encoder.copy_buffer_to_buffer(
            &self.tick_storage.side_inputs_out,
            0,
            &self.common_storage.side_inputs_in,
            0,
            self.buffer_size * 16,
        );
    }

    fn copy_ticks_out_to_in(&self, command_encoder: &mut wgpu::CommandEncoder) {
        command_encoder.copy_buffer_to_buffer(
            &self.common_storage.tick_delay_out,
            0,
            &self.common_storage.tick_delay_in,
            0,
            self.buffer_size * 4,
        );
    }

    fn encode_tick_pass(
        &self,
        command_encoder: &mut wgpu::CommandEncoder,
        tick_priority: TickPriority
    ) {
        // The right tick priority value will be at its own index in the cached buffer
        command_encoder.copy_buffer_to_buffer(
            &self.tick_storage.tick_priorities_cache,
            (tick_priority as u32 * 4) as wgpu::BufferAddress, // Index into the cache
            &self.tick_storage.tick_priority,
            0,
            4, // u32
        );
        {
            let mut tick_pass = command_encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Tick pass"),
                timestamp_writes: None,
            });
            tick_pass.set_pipeline(&self.tick_pipeline);
            tick_pass.set_bind_group(0, &self.constant_bind_group, &[]);
            tick_pass.set_bind_group(1, &self.common_bind_group, &[]);
            tick_pass.set_bind_group(2, &self.tick_bind_group, &[]);
            tick_pass.dispatch_workgroups(64, 1, 1);
        }
        self.copy_inputs_out_to_in(command_encoder);
        self.copy_ticks_out_to_in(command_encoder);
    }
    
    fn encode_update_pass(&self, command_encoder: &mut wgpu::CommandEncoder) {
        {
            let mut update_pass = command_encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Update pass"),
                timestamp_writes: None,
            });
            update_pass.set_pipeline(&self.update_pipeline);
            update_pass.set_bind_group(0, &self.constant_bind_group, &[]);
            update_pass.set_bind_group(1, &self.common_bind_group, &[]);
        }
        self.copy_ticks_out_to_in(command_encoder);
    }

    async fn create_command_buffer(self: Arc<Pipelines>, device: Arc<wgpu::Device>, count: u64) -> wgpu::CommandBuffer {
        let mut command_encoder = device.create_command_encoder(&Default::default());
        for i in 0..count {
            self.encode_tick_pass(&mut command_encoder, TickPriority::Highest);
            self.encode_update_pass(&mut command_encoder);
            self.encode_tick_pass(&mut command_encoder, TickPriority::Higher);
            self.encode_update_pass(&mut command_encoder);
            self.encode_tick_pass(&mut command_encoder, TickPriority::High);
            self.encode_update_pass(&mut command_encoder);
            self.encode_tick_pass(&mut command_encoder, TickPriority::Normal);
            self.encode_update_pass(&mut command_encoder);
        }
        command_encoder.finish()
    }

    pub fn run_passes(
        self: Arc<Pipelines>,
        device: Arc<wgpu::Device>,
        queue: &wgpu::Queue,
        count: u64,
        cache: Arc<Mutex<CommandCache>>
    ) {
        let instant = std::time::Instant::now();
        for _ in 0..count {
            //println!("Trying to get buffer...");
            let buffer = CommandCache::pop(cache.clone());
            //println!("Got buffer!");
            queue.submit(Some(buffer));
            //println!("Submitted.");
        }

        //println!("Total time after submit: {}", instant.elapsed().as_millis());
        device.poll(wgpu::PollType::Wait).unwrap();
        //println!("Total time after run: {}", instant.elapsed().as_millis());
    }
}
