use bevy::{
    core_pipeline::draw_3d_graph,
    diagnostic::{FrameTimeDiagnosticsPlugin, LogDiagnosticsPlugin},
    ecs::system::{
        lifetimeless::{Read, SQuery, SRes},
        SystemParamItem,
    },
    input::mouse::MouseMotion,
    pbr::SetShadowViewBindGroup,
    prelude::*,
    reflect::TypeUuid,
    render::{
        camera::{ActiveCamera, Camera3d},
        mesh::PrimitiveTopology,
        render_graph::{self, NodeRunError, RenderGraph, RenderGraphContext, SlotInfo, SlotType},
        render_phase::{
            AddRenderCommand, DrawFunctionId, DrawFunctions, EntityPhaseItem, EntityRenderCommand,
            PhaseItem, RenderCommand, RenderCommandResult, RenderPhase, TrackedRenderPass,
        },
        render_resource::{
            std140::AsStd140, BindGroup, BindGroupDescriptor, BindGroupEntry, BindGroupLayout,
            BindGroupLayoutDescriptor, BindGroupLayoutEntry, BindingType, BlendState, Buffer,
            BufferBindingType, BufferInitDescriptor, BufferSize, BufferUsages, BufferVec,
            CachedRenderPipelineId, ColorTargetState, ColorWrites, CompareFunction, DepthBiasState,
            DepthStencilState, FragmentState, FrontFace, IndexFormat, LoadOp, MultisampleState,
            Operations, PipelineCache, PolygonMode, PrimitiveState,
            RenderPassDepthStencilAttachment, RenderPassDescriptor, RenderPipelineDescriptor,
            ShaderStages, StencilFaceState, StencilState, TextureFormat, VertexState,
        },
        renderer::{RenderContext, RenderDevice, RenderQueue},
        texture::BevyDefault,
        view::{ExtractedView, ViewDepthTexture, ViewTarget, ViewUniform},
        RenderApp, RenderStage,
    },
};
use bytemuck::{cast_slice, Pod, Zeroable};
use examples_utils::camera::{CameraController, CameraControllerPlugin};
use rand::Rng;

fn main() {
    App::new()
        .insert_resource(WindowDescriptor {
            title: format!(
                "{} {} - cubes",
                env!("CARGO_PKG_NAME"),
                env!("CARGO_PKG_VERSION")
            ),
            width: 1280.0,
            height: 720.0,
            ..Default::default()
        })
        .add_plugins(DefaultPlugins)
        .add_plugin(CameraControllerPlugin)
        .add_plugin(FrameTimeDiagnosticsPlugin)
        .add_plugin(LogDiagnosticsPlugin::default())
        .add_plugin(CubesPlugin)
        .add_startup_system(setup)
        .run();
}

#[derive(Clone, Debug, Default)]
pub struct Cube {
    color: Color,
    center: Vec3,
    half_extents: Vec3,
}

impl Cube {
    pub fn random<R: Rng + ?Sized>(rng: &mut R, min: Vec3, max: Vec3) -> Self {
        Self {
            color: Color::WHITE,
            center: random_point_vec3(rng, min, max),
            half_extents: 0.01 * Vec3::ONE,
        }
    }

    pub fn random_y<R: Rng + ?Sized>(x: f32, z: f32, rng: &mut R, min: f32, max: f32) -> Self {
        Self {
            color: Color::WHITE,
            center: Vec3::new(x, rng.gen_range(min..max), z),
            half_extents: 0.01 * Vec3::ONE,
        }
    }
}

fn random_point_vec3<R: Rng + ?Sized>(rng: &mut R, min: Vec3, max: Vec3) -> Vec3 {
    Vec3::new(
        rng.gen_range(min.x..max.x),
        rng.gen_range(min.y..max.y),
        rng.gen_range(min.z..max.z),
    )
}

#[derive(Clone, Component, Debug, Default)]
struct Cubes {
    data: Vec<Cube>,
    extracted: bool,
}

fn setup(mut commands: Commands) {
    let mut cubes = Cubes::default();
    let mut n_cubes = std::env::args()
        .nth(1)
        .and_then(|arg| arg.parse::<usize>().ok())
        .unwrap_or(1_000_000);
    let dim = (n_cubes as f32).sqrt().ceil() as usize;
    n_cubes = (dim * dim) as usize;
    info!("Generating {} cubes", n_cubes);
    let sin_scale = std::f32::consts::TAU / 50.0;
    let y_scale = 10.0;
    // dbg!(half_dim);
    for z in 0..dim {
        for x in 0..dim {
            let (x, z) = (x as f32, z as f32);
            let y = (x * sin_scale).sin() * (z * sin_scale).cos();
            cubes.data.push(Cube {
                color: Color::rgb(x / dim as f32, y, z / dim as f32),
                center: Vec3::new(x, y_scale * y, z),
                half_extents: 0.5 * Vec3::ONE,
            });
        }
    }
    commands.spawn_bundle((cubes,));

    commands
        .spawn_bundle(PerspectiveCameraBundle {
            transform: Transform::from_translation(100.0 * Vec3::new(-1.0, 1.0, -1.0))
                .looking_at(0.5 * Vec3::new(dim as f32, 0.0, dim as f32), Vec3::Y),
            ..default()
        })
        .insert(CameraController::default());
}

fn extract_cubes_phase(mut commands: Commands, active_3d: Res<ActiveCamera<Camera3d>>) {
    if let Some(entity) = active_3d.get() {
        commands
            .get_or_spawn(entity)
            .insert(RenderPhase::<CubesPhaseItem>::default());
    }
}

fn extract_cubes(mut commands: Commands, mut cubes: Query<(Entity, &mut Cubes)>) {
    for (entity, mut cubes) in cubes.iter_mut() {
        if cubes.extracted {
            commands.get_or_spawn(entity).insert(Cubes {
                data: Vec::new(),
                extracted: true,
            });
        } else {
            commands.get_or_spawn(entity).insert(cubes.clone());
            // NOTE: Set this after cloning so we don't extract next time
            cubes.extracted = true;
        }
    }
}

#[derive(Clone, Copy, Debug, Default, Pod, Zeroable)]
#[repr(C)]
struct GpuCube {
    center: Vec4,
    half_extents: Vec4,
    color: [f32; 4],
}

impl From<&Cube> for GpuCube {
    fn from(cube: &Cube) -> Self {
        Self {
            center: cube.center.extend(1.0),
            half_extents: cube.half_extents.extend(0.0),
            color: cube.color.as_rgba_f32(),
        }
    }
}

#[derive(Component)]
struct GpuCubes {
    index_buffer: Option<Buffer>,
    index_count: u32,
    instances: BufferVec<GpuCube>,
}

impl Default for GpuCubes {
    fn default() -> Self {
        Self {
            index_buffer: None,
            index_count: 0,
            instances: BufferVec::<GpuCube>::new(BufferUsages::STORAGE),
        }
    }
}

#[derive(Component)]
struct GpuCubesBindGroup {
    bind_group: BindGroup,
}

const CUBE_BACKFACE_OPTIMIZATION: bool = true;
const NUM_CUBE_INDICES: usize = if CUBE_BACKFACE_OPTIMIZATION {
    3 * 3 * 2
} else {
    3 * 6 * 2
};
const NUM_CUBE_VERTICES: usize = 8;

fn generate_index_buffer_data(num_cubes: usize) -> Vec<u32> {
    #[rustfmt::skip]
    let cube_indices = [
        0u32, 2, 1, 2, 3, 1,
        5, 4, 1, 1, 4, 0,
        0, 4, 6, 0, 6, 2,
        6, 5, 7, 6, 4, 5,
        2, 6, 3, 6, 7, 3,
        7, 1, 3, 7, 5, 1,
    ];

    let num_indices = num_cubes * NUM_CUBE_INDICES;

    (0..num_indices)
        .map(|i| {
            let cube = i / NUM_CUBE_INDICES;
            let cube_local = i % NUM_CUBE_INDICES;
            cube as u32 * NUM_CUBE_VERTICES as u32 + cube_indices[cube_local]
        })
        .collect()
}

fn prepare_cubes(
    cubes: Query<&Cubes>,
    render_device: Res<RenderDevice>,
    render_queue: Res<RenderQueue>,
    mut gpu_cubes: ResMut<GpuCubes>,
) {
    for cubes in cubes.iter() {
        if cubes.extracted {
            continue;
        }
        for cube in cubes.data.iter() {
            gpu_cubes.instances.push(GpuCube::from(cube));
        }
        gpu_cubes.index_count = gpu_cubes.instances.len() as u32 * NUM_CUBE_INDICES as u32;
        let indices = generate_index_buffer_data(gpu_cubes.instances.len());
        gpu_cubes.index_buffer = Some(render_device.create_buffer_with_data(
            &BufferInitDescriptor {
                label: Some("gpu_cubes_index_buffer"),
                contents: cast_slice(&indices),
                usage: BufferUsages::INDEX,
            },
        ));

        gpu_cubes
            .instances
            .write_buffer(&*render_device, &*render_queue);
    }
}

pub struct CubesPhaseItem {
    pub entity: Entity,
    pub draw_function: DrawFunctionId,
}

impl PhaseItem for CubesPhaseItem {
    type SortKey = u32;

    #[inline]
    fn sort_key(&self) -> Self::SortKey {
        0
    }

    #[inline]
    fn draw_function(&self) -> DrawFunctionId {
        self.draw_function
    }
}

impl EntityPhaseItem for CubesPhaseItem {
    #[inline]
    fn entity(&self) -> Entity {
        self.entity
    }
}

fn queue_cubes(
    mut commands: Commands,
    opaque_3d_draw_functions: Res<DrawFunctions<CubesPhaseItem>>,
    cubes_pipeline: Res<CubesPipeline>,
    render_device: Res<RenderDevice>,
    cubes_query: Query<Entity, With<Cubes>>,
    gpu_cubes: Res<GpuCubes>,
    mut views: Query<&mut RenderPhase<CubesPhaseItem>>,
) {
    let draw_cubes = opaque_3d_draw_functions
        .read()
        .get_id::<DrawCubes>()
        .unwrap();

    for mut opaque_phase in views.iter_mut() {
        for entity in cubes_query.iter() {
            commands
                .get_or_spawn(entity)
                .insert_bundle((GpuCubesBindGroup {
                    bind_group: render_device.create_bind_group(&BindGroupDescriptor {
                        label: Some("gpu_cubes_bind_group"),
                        layout: &cubes_pipeline.cubes_layout,
                        entries: &[BindGroupEntry {
                            binding: 0,
                            resource: gpu_cubes.instances.buffer().unwrap().as_entire_binding(),
                        }],
                    }),
                },));
            opaque_phase.add(CubesPhaseItem {
                entity,
                draw_function: draw_cubes,
            });
        }
    }
}

mod node {
    pub const CUBES_PASS: &str = "cubes_pass";
}

pub struct CubesPassNode {
    query: QueryState<
        (
            &'static RenderPhase<CubesPhaseItem>,
            &'static ViewTarget,
            &'static ViewDepthTexture,
        ),
        With<ExtractedView>,
    >,
}

impl CubesPassNode {
    pub const IN_VIEW: &'static str = "view";

    pub fn new(world: &mut World) -> Self {
        Self {
            query: QueryState::new(world),
        }
    }
}

impl render_graph::Node for CubesPassNode {
    fn input(&self) -> Vec<SlotInfo> {
        vec![SlotInfo::new(CubesPassNode::IN_VIEW, SlotType::Entity)]
    }

    fn update(&mut self, world: &mut World) {
        self.query.update_archetypes(world);
    }

    fn run(
        &self,
        graph: &mut RenderGraphContext,
        render_context: &mut RenderContext,
        world: &World,
    ) -> Result<(), NodeRunError> {
        let view_entity = graph.get_input_entity(Self::IN_VIEW)?;
        let (cubes_phase, target, depth) = match self.query.get_manual(world, view_entity) {
            Ok(query) => query,
            Err(_) => return Ok(()), // No window
        };

        #[cfg(feature = "trace")]
        let _main_cubes_pass_span = info_span!("main_cubes_pass").entered();
        let pass_descriptor = RenderPassDescriptor {
            label: Some("main_cubes_pass"),
            // NOTE: The cubes pass loads the color
            // buffer as well as writing to it.
            color_attachments: &[target.get_color_attachment(Operations {
                load: LoadOp::Load,
                store: true,
            })],
            depth_stencil_attachment: Some(RenderPassDepthStencilAttachment {
                view: &depth.view,
                // NOTE: The cubes main pass loads the depth buffer and possibly overwrites it
                depth_ops: Some(Operations {
                    load: LoadOp::Load,
                    store: true,
                }),
                stencil_ops: None,
            }),
        };

        let draw_functions = world.resource::<DrawFunctions<CubesPhaseItem>>();

        let render_pass = render_context
            .command_encoder
            .begin_render_pass(&pass_descriptor);
        let mut draw_functions = draw_functions.write();
        let mut tracked_pass = TrackedRenderPass::new(render_pass);
        for item in &cubes_phase.items {
            let draw_function = draw_functions.get_mut(item.draw_function).unwrap();
            draw_function.draw(world, &mut tracked_pass, view_entity, item);
        }

        Ok(())
    }
}

struct CubesPlugin;

impl Plugin for CubesPlugin {
    fn build(&self, app: &mut App) {
        app.world.resource_mut::<Assets<Shader>>().set_untracked(
            CUBES_SHADER_HANDLE,
            Shader::from_wgsl(include_str!("cubes.wgsl")),
        );

        let render_app = app.sub_app_mut(RenderApp);

        render_app
            .init_resource::<DrawFunctions<CubesPhaseItem>>()
            .add_render_command::<CubesPhaseItem, DrawCubes>()
            .init_resource::<CubesPipeline>()
            .init_resource::<GpuCubes>()
            .add_system_to_stage(RenderStage::Extract, extract_cubes_phase)
            .add_system_to_stage(RenderStage::Extract, extract_cubes)
            .add_system_to_stage(RenderStage::Prepare, prepare_cubes)
            .add_system_to_stage(RenderStage::Queue, queue_cubes);

        let cubes_pass_node = CubesPassNode::new(&mut render_app.world);
        let mut graph = render_app.world.resource_mut::<RenderGraph>();
        let draw_3d_graph = graph.get_sub_graph_mut(draw_3d_graph::NAME).unwrap();
        draw_3d_graph.add_node(node::CUBES_PASS, cubes_pass_node);
        draw_3d_graph
            .add_node_edge(node::CUBES_PASS, draw_3d_graph::node::MAIN_PASS)
            .unwrap();
        draw_3d_graph
            .add_slot_edge(
                draw_3d_graph.input_node().unwrap().id,
                draw_3d_graph::input::VIEW_ENTITY,
                node::CUBES_PASS,
                CubesPassNode::IN_VIEW,
            )
            .unwrap();
    }
}

struct CubesPipeline {
    pipeline_id: CachedRenderPipelineId,
    cubes_layout: BindGroupLayout,
}

const CUBES_SHADER_HANDLE: HandleUntyped =
    HandleUntyped::weak_from_u64(Shader::TYPE_UUID, 17343092250772987267);

impl FromWorld for CubesPipeline {
    fn from_world(world: &mut World) -> Self {
        let view_layout =
            world
                .resource::<RenderDevice>()
                .create_bind_group_layout(&BindGroupLayoutDescriptor {
                    entries: &[
                        // View
                        BindGroupLayoutEntry {
                            binding: 0,
                            visibility: ShaderStages::VERTEX | ShaderStages::FRAGMENT,
                            ty: BindingType::Buffer {
                                ty: BufferBindingType::Uniform,
                                has_dynamic_offset: true,
                                min_binding_size: BufferSize::new(
                                    ViewUniform::std140_size_static() as u64,
                                ),
                            },
                            count: None,
                        },
                    ],
                    label: Some("shadow_view_layout"),
                });

        let cubes_layout =
            world
                .resource::<RenderDevice>()
                .create_bind_group_layout(&BindGroupLayoutDescriptor {
                    label: None,
                    entries: &[BindGroupLayoutEntry {
                        binding: 0,
                        visibility: ShaderStages::VERTEX,
                        ty: BindingType::Buffer {
                            ty: BufferBindingType::Storage { read_only: true },
                            has_dynamic_offset: false,
                            min_binding_size: BufferSize::new(0),
                        },
                        count: None,
                    }],
                });

        let mut pipeline_cache = world.resource_mut::<PipelineCache>();
        let pipeline_id = pipeline_cache.queue_render_pipeline(RenderPipelineDescriptor {
            label: Some("cubes_pipeline".into()),
            layout: Some(vec![view_layout, cubes_layout.clone()]),
            vertex: VertexState {
                shader: CUBES_SHADER_HANDLE.typed(),
                shader_defs: vec![],
                entry_point: "vertex".into(),
                buffers: vec![],
            },
            fragment: Some(FragmentState {
                shader: CUBES_SHADER_HANDLE.typed(),
                shader_defs: vec![],
                entry_point: "fragment".into(),
                targets: vec![ColorTargetState {
                    format: TextureFormat::bevy_default(),
                    blend: Some(BlendState::REPLACE),
                    write_mask: ColorWrites::ALL,
                }],
            }),
            primitive: PrimitiveState {
                front_face: FrontFace::Ccw,
                cull_mode: None,
                unclipped_depth: false,
                polygon_mode: PolygonMode::Fill,
                conservative: false,
                topology: PrimitiveTopology::TriangleList,
                strip_index_format: None,
            },
            depth_stencil: Some(DepthStencilState {
                format: TextureFormat::Depth32Float,
                depth_write_enabled: true,
                depth_compare: CompareFunction::Greater,
                stencil: StencilState {
                    front: StencilFaceState::IGNORE,
                    back: StencilFaceState::IGNORE,
                    read_mask: 0,
                    write_mask: 0,
                },
                bias: DepthBiasState {
                    constant: 0,
                    slope_scale: 0.0,
                    clamp: 0.0,
                },
            }),
            multisample: MultisampleState {
                count: Msaa::default().samples,
                mask: !0,
                alpha_to_coverage_enabled: false,
            },
        });

        Self {
            pipeline_id,
            cubes_layout,
        }
    }
}

type DrawCubes = (
    SetCubesPipeline,
    SetShadowViewBindGroup<0>,
    SetGpuCubesBindGroup<1>,
    DrawVertexPulledCubes,
);

struct SetCubesPipeline;
impl<P: PhaseItem> RenderCommand<P> for SetCubesPipeline {
    type Param = (SRes<PipelineCache>, SRes<CubesPipeline>);
    #[inline]
    fn render<'w>(
        _view: Entity,
        _item: &P,
        params: SystemParamItem<'w, '_, Self::Param>,
        pass: &mut TrackedRenderPass<'w>,
    ) -> RenderCommandResult {
        let (pipeline_cache, cubes_pipeline) = params;
        if let Some(pipeline) = pipeline_cache
            .into_inner()
            .get_render_pipeline(cubes_pipeline.pipeline_id)
        {
            pass.set_render_pipeline(pipeline);
            RenderCommandResult::Success
        } else {
            RenderCommandResult::Failure
        }
    }
}

struct SetGpuCubesBindGroup<const I: usize>;
impl<const I: usize> EntityRenderCommand for SetGpuCubesBindGroup<I> {
    type Param = SQuery<Read<GpuCubesBindGroup>>;

    #[inline]
    fn render<'w>(
        _view: Entity,
        item: Entity,
        gpu_cubes_bind_groups: SystemParamItem<'w, '_, Self::Param>,
        pass: &mut TrackedRenderPass<'w>,
    ) -> RenderCommandResult {
        let gpu_cubes_bind_group = gpu_cubes_bind_groups.get_inner(item).unwrap();
        pass.set_bind_group(I, &gpu_cubes_bind_group.bind_group, &[]);

        RenderCommandResult::Success
    }
}

struct DrawVertexPulledCubes;
impl EntityRenderCommand for DrawVertexPulledCubes {
    type Param = SRes<GpuCubes>;

    #[inline]
    fn render<'w>(
        _view: Entity,
        _item: Entity,
        gpu_cubes: SystemParamItem<'w, '_, Self::Param>,
        pass: &mut TrackedRenderPass<'w>,
    ) -> RenderCommandResult {
        let gpu_cubes = gpu_cubes.into_inner();
        pass.set_index_buffer(
            gpu_cubes.index_buffer.as_ref().unwrap().slice(..),
            0,
            IndexFormat::Uint32,
        );
        pass.draw_indexed(0..gpu_cubes.index_count, 0, 0..1);
        RenderCommandResult::Success
    }
}
