# bevy-vertex-pulling

Vertex pulling is a useful and efficient technique for drawing many quads or cubes by storing per-instance data in a buffer, indexing into it using an index calculated based on the vertex index, and calculating offsets from a world position for the instance based on which vertex of the quad/cube is being processed.

## Approaches Implemented

- Instance data stored in:
  - Storage buffer
- Shapes
  - Quads
  - Cuboids/voxels

## Things to do/try

- [ ] Instance data storage
  - [ ] Instance buffer
  - [ ] Texture buffer
  - [ ] Reusable abstraction
- [ ] Reduce overdraw
  - [ ] Depth prepass
  - [ ] Sorting from front to back
  - [ ] Tighter containing geometry
    - [ ] Triangle mesh to draw quads and `discard` like alpha mask
    - [ ] Bevy circular texture with a triangle mesh and `discard` like alpha mask
    - [ ] Bevy circular texture with a circular mesh
- [ ] Support more basic shapes
- [ ] Support complex meshes
- [ ] Billboarding (make the planar shape face the camera)
- [ ] Culling
  - [ ] Compute shader-based frustum culling
  - [ ] Compute shader-based occlusion culling
- [ ] Compute shader software rasterisation when the shape is small on-screen as raster shades fragments using 2x2 'pixel quads'
  - https://research.nvidia.com/publication/2011-08_high-performance-software-rasterization-gpus
  - https://raphlinus.github.io/
  - https://www.cg.tuwien.ac.at/research/publications/2021/SCHUETZ-2021-PCC/

## License

bevy-vertex-pulling is free and open source! All code in this repository is dual-licensed under either:

* MIT License ([LICENSE-MIT](docs/LICENSE-MIT) or [http://opensource.org/licenses/MIT](http://opensource.org/licenses/MIT))
* Apache License, Version 2.0 ([LICENSE-APACHE](docs/LICENSE-APACHE) or [http://www.apache.org/licenses/LICENSE-2.0](http://www.apache.org/licenses/LICENSE-2.0))

at your option. This means you can select the license you prefer! This dual-licensing approach is the de-facto standard in the Rust ecosystem and there are [very good reasons](https://github.com/bevyengine/bevy/issues/2373) to include both.

Unless you explicitly state otherwise, any contribution intentionally submitted for inclusion in the work by you, as defined in the Apache-2.0 license, shall be dual licensed as above, without any additional terms or conditions.
