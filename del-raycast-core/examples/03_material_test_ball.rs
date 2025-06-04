use del_msh_cpu::search_bvh3::TriMeshWithBvh;
use del_raycast_core::textures::Texture;

struct Shape {
    vtx2xyz: Vec<f32>,
    tri2vtx: Vec<usize>,
    uvs: Vec<f32>,
    transform: [f32; 16],
    bvhnodes: Vec<usize>,
    bvhnode2aabb: Vec<f32>,
    material_index: Option<usize>,
}

// return scale factor t, shape index and triangle index of intersection
fn intersect_bvh(
    ray_org: &[f32; 3],
    ray_dir: &[f32; 3],
    shapes: &[Shape],
) -> Option<(f32, usize, usize)> {
    // compute intersection
    let mut t_min = f32::INFINITY;
    let mut shape_i: usize = usize::MAX;
    let mut tri_i: usize = usize::MAX;
    for (s_i, trimesh) in shapes.iter().enumerate() {
        let ti = del_geo_core::mat4_col_major::try_inverse(&trimesh.transform).unwrap();
        let ray_org = del_geo_core::mat4_col_major::transform_homogeneous(&ti, ray_org).unwrap();
        let ray_dir = del_geo_core::mat4_col_major::transform_direction(&ti, ray_dir);

        let Some((t, i_tri)) = del_msh_cpu::search_bvh3::first_intersection_ray(
            &ray_org,
            &ray_dir,
            &TriMeshWithBvh {
                tri2vtx: &trimesh.tri2vtx,
                vtx2xyz: &trimesh.vtx2xyz,
                bvhnodes: &trimesh.bvhnodes,
                bvhnode2aabb: &trimesh.bvhnode2aabb,
            },
            0,
            f32::INFINITY,
        ) else {
            continue;
        };
        if t < t_min {
            shape_i = s_i;
            tri_i = i_tri;
            t_min = t;
        }
    }

    if t_min == f32::INFINITY {
        None
    } else {
        Some((t_min, shape_i, tri_i))
    }
}

// get uv coordinates of position of a triangle tri_i
fn get_tri_uv(tri_i: usize, pos: &[f32; 3], shape: &Shape) -> [f32; 2] {
    let vertices = &shape.tri2vtx[tri_i..tri_i + 3];

    let f: Vec<[f32; 3]> = vertices
        .iter()
        .map(|&vert| {
            assert!(
                vert * 3 + 3 <= shape.vtx2xyz.len(),
                "vertex index out of bounds"
            );
            let v = &shape.vtx2xyz[vert * 3..(vert * 3 + 3)];
            [v[0], v[1], v[2]]
        })
        .collect();

    let bary = del_geo_core::tri3::to_barycentric_coords(&f[0], &f[1], &f[2], pos);

    let vert_uvs: Vec<[f32; 2]> = vertices
        .iter()
        .map(|&vert| {
            assert!(vert * 2 + 2 <= shape.uvs.len(), "uv index out of bounds");
            let uv = &shape.uvs[vert * 2..(vert * 2 + 2)];
            [uv[0], uv[1]]
        })
        .collect();

    let mut acc = [0.; 2];
    vert_uvs.iter().enumerate().for_each(|(i, vert_uv)| {
        acc[0] += vert_uv[0] * bary[i];
        acc[1] += vert_uv[1] * bary[i];
    });

    acc
}

fn parse() -> anyhow::Result<(
    Vec<Shape>,
    del_raycast_core::parse_pbrt::Camera,
    pbrt4::Scene,
)> {
    let path_file = "asset/material-testball/scene-v4.pbrt";
    let scene = pbrt4::Scene::from_file(path_file)?;
    let mut shapes: Vec<Shape> = vec![];
    let camera = del_raycast_core::parse_pbrt::camera(&scene);
    for shape_entity in scene.shapes.iter() {
        let transform = shape_entity.transform.to_cols_array();
        let (_, _, tri2vtx, vtx2xyz, _, uvs) =
            del_raycast_core::parse_pbrt::trimesh3_from_shape_entity(shape_entity, path_file)
                .unwrap();
        let bvhnodes = del_msh_cpu::bvhnodes_morton::from_triangle_mesh(&tri2vtx, &vtx2xyz, 3);
        let bvhnode2aabb = del_msh_cpu::bvhnode2aabb3::from_uniform_mesh_with_bvh(
            0,
            &bvhnodes,
            Some((&tri2vtx, 3)),
            &vtx2xyz,
            None,
        );
        let shape = Shape {
            tri2vtx,
            vtx2xyz,
            uvs,
            transform,
            bvhnodes,
            bvhnode2aabb,
            material_index: shape_entity.material_index,
        };
        shapes.push(shape);
    }
    Ok((shapes, camera, scene))
}

fn main() -> anyhow::Result<()> {
    let (shapes, camera, scene) = parse()?;
    {
        let mut tri2vtx: Vec<usize> = vec![];
        let mut vtx2xyz: Vec<f32> = vec![];
        for trimesh in shapes.iter() {
            let t = del_geo_core::mat4_col_major::mult_mat_col_major(
                &camera.transform_world2camlcl,
                &trimesh.transform,
            );
            let trimesh_vtx2xyz = del_msh_cpu::vtx2xyz::transform_homogeneous(&trimesh.vtx2xyz, &t);
            del_msh_cpu::uniform_mesh::merge(
                &mut tri2vtx,
                &mut vtx2xyz,
                &trimesh.tri2vtx,
                &trimesh_vtx2xyz,
                3,
            );
        }
        del_msh_cpu::io_obj::save_tri2vtx_vtx2xyz(
            "target/03_material_test_ball.obj",
            &tri2vtx,
            &vtx2xyz,
            3,
        )?;
    }

    let transform_cam_lcl2glbl =
        del_geo_core::mat4_col_major::try_inverse(&camera.transform_world2camlcl).unwrap();
    let img_shape = (540, 360);

    {
        // computing depth image
        let shoot_ray = |i_pix: usize, pix: &mut [f32]| {
            let pix = arrayref::array_mut_ref![pix, 0, 3];
            let iw = i_pix % img_shape.0;
            let ih = i_pix / img_shape.0;

            let (ray_org, ray_dir) = del_raycast_core::cam_pbrt::cast_ray_plus_z(
                (iw, ih),
                (0., 0.),
                img_shape,
                camera.camera_fov,
                transform_cam_lcl2glbl,
            );

            let Some((t, _i_shape_entity, _tri_i)) = intersect_bvh(&ray_org, &ray_dir, &shapes)
            else {
                return;
            };
            let v = (t - 1.5) * 0.8;
            *pix = [v; 3];
        };

        let mut img_out = vec![0f32; img_shape.0 * img_shape.1 * 3];
        use rayon::prelude::*;
        img_out
            .par_chunks_mut(3)
            .enumerate()
            .for_each(|(i_pix, pix)| shoot_ray(i_pix, pix));
        del_canvas::write_hdr_file("target/03_material_test_ball.hdr", img_shape, &img_out)?;
    }

    {
        // computing reflectance image
        let materials = del_raycast_core::parse_pbrt::parse_material(&scene);
        let textures = del_raycast_core::parse_pbrt::parse_texture(&scene);
        let shoot_ray = |i_pix: usize, pix: &mut [f32]| {
            let pix = arrayref::array_mut_ref![pix, 0, 3];
            let iw = i_pix % img_shape.0;
            let ih = i_pix / img_shape.0;

            let (ray_org, ray_dir) = del_raycast_core::cam_pbrt::cast_ray_plus_z(
                (iw, ih),
                (0., 0.),
                img_shape,
                camera.camera_fov,
                transform_cam_lcl2glbl,
            );

            let Some((t, i_shape_entity, tri_i)) = intersect_bvh(&ray_org, &ray_dir, &shapes)
            else {
                return;
            };

            let shape = &shapes[i_shape_entity];

            let i_material = shape.material_index.unwrap();

            assert!(
                i_material < materials.len(),
                "{} {}",
                i_material,
                materials.len()
            );

            let reflectance = match &materials[i_material] {
                del_raycast_core::material::Material::Diff(mat) => {
                    if mat.reflectance_texture != usize::MAX {
                        let tex = &textures[mat.reflectance_texture];
                        match tex {
                            Texture::Checkerboard(tex) => {
                                let pos = del_geo_core::vec3::axpy(t, &ray_dir, &ray_org);
                                let uv = get_tri_uv(tri_i, &pos, shape);
                                del_raycast_core::textures::sample_checkerboard(
                                    &uv, tex.uscale, tex.vscale, &tex.tex1, &tex.tex2,
                                )
                            }
                        }
                    } else {
                        mat.reflectance
                    }
                }
                del_raycast_core::material::Material::Cond(mat) => mat.reflectance,
                del_raycast_core::material::Material::CoaDiff(mat) => mat.reflectance,
                _ => {
                    panic!("No reflectance of Material");
                }
            };
            *pix = reflectance;
        };

        let mut img_out = vec![0f32; img_shape.0 * img_shape.1 * 3];
        use rayon::prelude::*;
        img_out
            .par_chunks_mut(3)
            .enumerate()
            .for_each(|(i_pix, pix)| shoot_ray(i_pix, pix));
        del_canvas::write_hdr_file(
            "target/03_material_test_ball_color.hdr",
            img_shape,
            &img_out,
        )?;
    }

    Ok(())
}
