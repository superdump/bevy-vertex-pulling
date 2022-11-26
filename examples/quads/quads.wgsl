#import bevy_pbr::mesh_view_types

struct Quad {
    center: vec3<f32>,
    flags: u32,
    half_extents: vec4<f32>,
    color: vec4<f32>,
}

let QUAD_FLAG_BILLBOARD_BIT: u32 = 1u;
let QUAD_FLAG_BILLBOARD_WORLD_Y_BIT: u32 = 2u;
let QUAD_FLAG_BILLBOARD_FIXED_SCREEN_SIZE_BIT: u32 = 4u;

struct Quads {
    data: array<Quad>,
}

@group(0) @binding(0)
var<uniform> view: View;

@group(1) @binding(0)
var<storage> quads: Quads;

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) world_position: vec4<f32>,
    @location(1) world_normal: vec3<f32>,
    @location(2) uv: vec2<f32>,
    @location(3) color: vec4<f32>,
};

@vertex
fn vertex(@builtin(vertex_index) vertex_index: u32) -> VertexOutput {
    var out: VertexOutput;

    let instance_index = vertex_index >> 2u;
    let quad = quads.data[instance_index];

    let xyz = vec3<f32>(f32(vertex_index & 0x1u), f32((vertex_index & 0x2u) >> 1u), 0.5);
    out.uv = vec2<f32>(xyz.xy);
    let relative_pos_unit = xyz * 2.0 - vec3<f32>(1.0);
    var relative_pos: vec3<f32>;

    if ((quad.flags & QUAD_FLAG_BILLBOARD_BIT) != 0u) {
        // View-right in world space is the 0th column of the view matrix
        let right = normalize(view.view[0].xyz);
        var up: vec3<f32>;
        if ((quad.flags & QUAD_FLAG_BILLBOARD_WORLD_Y_BIT) != 0u) {
            // The world-space normal has only x and z components
            out.world_normal = normalize((view.world_position - quad.center) * vec3<f32>(1.0, 0.0, 1.0));
            // Use world-space up
            up = vec3<f32>(0.0, 1.0, 0.0);
        } else {
            // The world-space normal points from the quad center to the camera
            out.world_normal = normalize(view.world_position - quad.center);
            // View-up in world space is the 1st column of the view matrix
            up = normalize(view.view[1].xyz);
        }
        // Calculate the world-space offset in the right and up directions by the respective half
        // extents
        relative_pos = right * relative_pos_unit.x * quad.half_extents.x
            + up * relative_pos_unit.y * quad.half_extents.y;
        // Apply the world-space offset
        out.world_position = vec4<f32>(quad.center.xyz + relative_pos, 1.0);
        // Transform to clip space
        out.clip_position = view.view_proj * out.world_position;
    } else if ((quad.flags & QUAD_FLAG_BILLBOARD_FIXED_SCREEN_SIZE_BIT) != 0u) {
        // Transform the quad center position to clip space
        out.clip_position = view.view_proj * vec4<f32>(quad.center, 1.0);
        // Clip to normalized device coordinate space
        out.clip_position = out.clip_position / out.clip_position.w;

        // Offset by the proportion of the screen in x and y. half_extents are in screen pixels in
        // this mode.
        out.clip_position.x = out.clip_position.x + (quad.half_extents.x / view.viewport.z) * relative_pos_unit.x;
        out.clip_position.y = out.clip_position.y + (quad.half_extents.y / view.viewport.w) * relative_pos_unit.y;

        // Transform back to world coordinates
        out.world_position = view.inverse_projection * out.clip_position;
        out.world_position = out.world_position / out.world_position.w;
        // The world-space normal points from the quad center to the camera
        out.world_normal = normalize(view.world_position - quad.center);
    } else {
        // No billboarding so the world-space normal points along +z
        out.world_normal = vec3<f32>(0.0, 0.0, 1.0);

        // Calculate the world-space offset
        relative_pos = relative_pos_unit * vec3<f32>(quad.half_extents.xy, 0.0);
        // Apply the world-space offset
        out.world_position = vec4<f32>(quad.center.xyz + relative_pos, 1.0);
        // Transform to clip space
        out.clip_position = view.view_proj * out.world_position;
    }

    out.color = quad.color;
    return out;
}

struct FragmentInput {
    @builtin(front_facing) is_front: bool,
    @location(0) world_position: vec4<f32>,
    @location(1) world_normal: vec3<f32>,
    @location(2) uv: vec2<f32>,
    @location(3) color: vec4<f32>,
};

@fragment
fn fragment(in: FragmentInput) -> @location(0) vec4<f32> {
    return in.color;
}

