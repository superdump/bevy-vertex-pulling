struct View {
    view_proj: mat4x4<f32>;
    view: mat4x4<f32>;
    inverse_view: mat4x4<f32>;
    projection: mat4x4<f32>;
    world_position: vec3<f32>;
    near: f32;
    far: f32;
    width: f32;
    height: f32;
};

struct Cube {
    center: vec4<f32>;
    half_extents: vec4<f32>;
    color: vec4<f32>;
};

struct Cubes {
    data: array<Cube>;
};

[[group(0), binding(0)]]
var<uniform> view: View;

[[group(1), binding(0)]]
var<storage> cubes: Cubes;

struct VertexOutput {
    [[builtin(position)]] clip_position: vec4<f32>;
    [[location(0)]] world_position: vec4<f32>;
    [[location(1)]] world_normal: vec3<f32>;
    [[location(2)]] uvw: vec3<f32>;
    [[location(3)]] color: vec4<f32>;
};

[[stage(vertex)]]
fn vertex([[builtin(vertex_index)]] vertex_index: u32) -> VertexOutput {
    var out: VertexOutput;

    let instance_index = vertex_index >> 3u;
    let cube = cubes.data[instance_index];

    // branchless mirroring
    let local_camera_pos = view.world_position - cube.center.xyz;
    let mirror_mask =
        u32(local_camera_pos.y < 0.0) << 2u |
        u32(local_camera_pos.z < 0.0) << 1u |
        u32(local_camera_pos.x < 0.0);
    let vx = vertex_index ^ mirror_mask;

    var xyz: vec3<i32> = vec3<i32>(
        i32(vx & 0x1u),
        i32((vx & 0x4u) >> 2u),
        i32((vx & 0x2u) >> 1u)
    );

    out.uvw = vec3<f32>(xyz);
    let relative_pos_unit = out.uvw * 2.0 - 1.0;
    let relative_pos = relative_pos_unit * cube.half_extents.xyz;

    out.world_position = vec4<f32>(cube.center.xyz + relative_pos, 1.0);
    out.world_normal = vec3<f32>(0.0, 0.0, 1.0);

    out.clip_position = view.view_proj * out.world_position;
    out.color = cube.color;
    return out;
}

struct FragmentInput {
    [[builtin(front_facing)]] is_front: bool;
    [[location(0)]] world_position: vec4<f32>;
    [[location(1)]] world_normal: vec3<f32>;
    [[location(2)]] uvw: vec3<f32>;
    [[location(3)]] color: vec4<f32>;
};

[[stage(fragment)]]
fn fragment(in: FragmentInput) -> [[location(0)]] vec4<f32> {
    return in.color;
}

