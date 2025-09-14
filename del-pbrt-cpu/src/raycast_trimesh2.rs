use num_traits::AsPrimitive;

pub struct TriMeshWithBvh<'a, Index, Real> {
    pub tri2vtx: &'a [Index],
    pub vtx2xy: &'a [Real],
    pub bvhnodes: &'a [Index],
    pub aabbs: &'a [Real],
}

///
/// * `transform_xy2pix` - from `xy` to `pixel coordinate`
#[allow(clippy::identity_op)]
pub fn draw_vtxcolor<Index, Real>(
    &(img_width, img_height): &(usize, usize),
    pix2color: &mut [Real],
    trimesh: TriMeshWithBvh<Index, Real>,
    vtx2color: &[Real],
    transform_xy2pix: &[Real; 9],
) where
    Real: num_traits::Float + 'static + Copy,
    Index: AsPrimitive<usize> + num_traits::PrimInt,
    usize: AsPrimitive<Real> + AsPrimitive<Index>,
{
    let half = Real::one() / (Real::one() + Real::one());
    let num_dim = pix2color.len() / (img_width * img_height);
    let num_vtx = trimesh.vtx2xy.len() / 2;
    assert_eq!(num_vtx, vtx2color.len());
    let transform_pix2xy = del_geo_core::mat3_col_major::try_inverse(transform_xy2pix).unwrap();
    assert_eq!(vtx2color.len(), num_vtx * num_dim);
    for i_h in 0..img_height {
        for i_w in 0..img_width {
            let p_xy = del_geo_core::mat3_col_major::transform_homogeneous::<Real>(
                &transform_pix2xy,
                &[
                    <usize as AsPrimitive<Real>>::as_(i_w) + half,
                    <usize as AsPrimitive<Real>>::as_(i_h) + half,
                ],
            )
            .unwrap();
            let mut res: Vec<(Index, Real, Real)> = vec![];
            del_msh_core::search_bvh2::including_point::<Real, Index>(
                &mut res,
                trimesh.tri2vtx,
                trimesh.vtx2xy,
                &p_xy,
                0,
                trimesh.bvhnodes,
                trimesh.aabbs,
            );
            let Some(&(i_tri, r0, r1)) = res.first() else {
                continue;
            };
            let i_tri: usize = i_tri.as_();
            let r2 = Real::one() - r0 - r1;
            let iv0: usize = trimesh.tri2vtx[i_tri * 3 + 0].as_();
            let iv1: usize = trimesh.tri2vtx[i_tri * 3 + 1].as_();
            let iv2: usize = trimesh.tri2vtx[i_tri * 3 + 2].as_();
            for i_dim in 0..num_dim {
                pix2color[(i_h * img_width + i_w) * num_dim + i_dim] = r0
                    * vtx2color[iv0 * num_dim + i_dim]
                    + r1 * vtx2color[iv1 * num_dim + i_dim]
                    + r2 * vtx2color[iv2 * num_dim + i_dim];
            }
        }
    }
}

#[test]
fn test0() -> anyhow::Result<()> {
    let (tri2vtx, vtx2xy) = del_msh_core::trimesh2_dynamic::meshing_from_polyloop2::<usize, f32>(
        &[0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0],
        0.03,
        0.03,
    );
    let bvhnodes = del_msh_core::bvhnodes_morton::from_triangle_mesh(&tri2vtx, &vtx2xy, 2);
    let aabbs = del_msh_core::bvhnode2aabb2::from_uniform_mesh_with_bvh(
        0,
        &bvhnodes,
        Some((&tri2vtx, 3)),
        &vtx2xy,
        None,
    );
    let vtx2color = {
        use rand::Rng;
        let mut reng = rand::thread_rng();
        let mut vtx2color = vec![0.5_f32; vtx2xy.len() / 2];
        for i_vtx in 0..vtx2color.len() {
            vtx2color[i_vtx] = reng.gen::<f32>();
        }
        vtx2color
    };
    dbg!(tri2vtx.len(), vtx2xy.len());
    let img_shape = (400, 300);
    let mut pix2color = vec![0_f32; img_shape.0 * img_shape.1];
    let transform_xy2pix =
        crate::cam2::transform_world2pix_ortho_preserve_asp(&img_shape, &[-0.1, -0.1, 1.1, 1.1]);
    let now = std::time::Instant::now();
    draw_vtxcolor(
        &img_shape,
        &mut pix2color,
        TriMeshWithBvh {
            tri2vtx: &tri2vtx,
            vtx2xy: &vtx2xy,
            bvhnodes: &bvhnodes,
            aabbs: &aabbs,
        },
        &vtx2color,
        &transform_xy2pix,
    );
    println!("{:?}", now.elapsed());
    del_canvas_image::write_png_from_float_image_grayscale(
        "../target/rasterize_trimesh2-test0.png",
        &img_shape,
        &pix2color,
    )?;
    Ok(())
}
