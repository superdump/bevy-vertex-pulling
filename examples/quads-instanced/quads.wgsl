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

[[group(0), binding(0)]]
var<uniform> view: View;

struct VertexInput {
    [[builtin(vertex_index)]] vertex_index: u32;
    [[location(0)]] i_center: vec2<f32>;
    [[location(1)]] i_half_extents: vec2<f32>;
    [[location(2)]] i_color: u32;
};

struct VertexOutput {
    [[builtin(position)]] clip_position: vec4<f32>;
    [[location(0)]] world_position: vec4<f32>;
    [[location(1)]] world_normal: vec3<f32>;
    [[location(2)]] uv: vec2<f32>;
    [[location(3)]] color: vec4<f32>;
};

[[stage(vertex)]]
fn vertex(vertex: VertexInput) -> VertexOutput {
    var out: VertexOutput;

    let xy = vec2<i32>(i32(vertex.vertex_index & 0x1u), i32((vertex.vertex_index & 0x2u) >> 1u));
    out.uv = vec2<f32>(xy.xy);
    let relative_pos_unit = out.uv * 2.0 - 1.0;
    let relative_pos = vec2<f32>(relative_pos_unit * vertex.i_half_extents);

    out.world_position = vec4<f32>(vertex.i_center.xy + relative_pos, 0.0, 1.0);
    out.world_normal = vec3<f32>(0.0, 0.0, 1.0);

    out.clip_position = view.view_proj * out.world_position;
    out.color = unpack4x8unorm(vertex.i_color);
    return out;
}

struct FragmentInput {
    [[builtin(front_facing)]] is_front: bool;
    [[location(0)]] world_position: vec4<f32>;
    [[location(1)]] world_normal: vec3<f32>;
    [[location(2)]] uv: vec2<f32>;
    [[location(3)]] color: vec4<f32>;
};

[[stage(fragment)]]
fn fragment(in: FragmentInput) -> [[location(0)]] vec4<f32> {
    return in.color;
}

