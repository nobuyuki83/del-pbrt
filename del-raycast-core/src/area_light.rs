pub struct AreaLight {
    pub spectrum_rgb: Option<[f32; 3]>,
    pub two_sided: bool,
}

pub fn sampling_light(
    tri2cumsumarea: &[f32],
    r2: [f32; 2],
    tri2vtx: &[usize],
    vtx2xyz: &[f32],
    vtx2nrm: &[f32],
) -> ([f32; 3], [f32; 3]) {
    let (i_tri_light, r1, r2) =
        del_msh_core::trimesh::sample_uniformly(tri2cumsumarea, r2[0], r2[1]);
    let (p0, p1, p2) = del_msh_core::trimesh3::to_corner_points(tri2vtx, vtx2xyz, i_tri_light);
    let light_pos = del_geo_core::tri3::position_from_barycentric_coords(
        &p0,
        &p1,
        &p2,
        &[1. - r1 - r2, r1, r2],
    );
    let light_nrm =
        crate::shape::triangle_mesh_normal_at(tri2vtx, vtx2xyz, vtx2nrm, &light_pos, i_tri_light);
    let light_pos = del_geo_core::vec3::axpy(1.0e-3, &light_nrm, &light_pos);
    (light_pos, light_nrm)
}
