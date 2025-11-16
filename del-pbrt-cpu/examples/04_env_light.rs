
struct MyScene {
    sphere_rad: f32,
    sphere_cntr: [f32; 3],
    tex_shape: (usize, usize),
    tex_data: Vec<f32>,
    transform_lcl2world_env: [f32; 16],
    transform_world2lcl_env: [f32; 16],
    materials: Vec<del_pbrt_cpu::material::Material>,
}

pub fn calc_grayscale(input_img: &[f32], w: usize, h: usize) -> Vec<f32> {
    let mut gray = vec![0f32; w * h];
    use rayon::iter::ParallelIterator;
    use rayon::iter::IndexedParallelIterator;
    use rayon::iter::IntoParallelRefMutIterator;
    gray.par_iter_mut().enumerate().for_each(|(i_pix, pix)| {
        let r = input_img[i_pix*3]; // use input_img instead img
        let g = input_img[i_pix*3+1]; // use input_img instead img
        let b = input_img[i_pix*3+2]; // use input_img instead img
        let gray = 0.2126 * r + 0.7152 * g + 0.0722 * b;
        *pix = gray;
    });
    gray
}

pub fn material_sampling(scene: &MyScene, camera: &del_pbrt_cpu::parse_pbrt::Camera) -> anyhow::Result<(), anyhow::Error>
{
    // material sampling
    let samples = 1024;
    let shoot_ray = |i_pix: usize, pix: &mut image::Rgb<f32>| {
        let ih = i_pix / camera.img_shape.0;
        let iw = i_pix % camera.img_shape.0;
        let (ray_org, ray_dir) = del_pbrt_cpu::cam_pbrt::cast_ray_plus_z(
            (iw, ih),
            (0., 0.),
            camera.img_shape,
            camera.camera_fov,
            camera.transform_camlcl2world,
        );
        let t = del_geo_core::sphere::intersection_ray(
            scene.sphere_rad,
            &scene.sphere_cntr,
            &ray_org,
            &ray_dir,
        );
        if let Some(t) = t {
            use del_geo_core::vec3;
            let hit_pos = vec3::axpy::<f32>(t, &ray_dir, &ray_org);
            let hit_nrm = vec3::sub(&hit_pos, &scene.sphere_cntr);
            let hit_nrm = vec3::normalize(&hit_nrm);
            let mut radiance = [0.; 3];
            use rand::Rng;
            use rand::SeedableRng;
            let mut rng = rand_chacha::ChaChaRng::seed_from_u64(i_pix as u64);
            for _isample in 0..samples {
                let refl_dir: [f32; 3] = del_pbrt_cpu::sampling::hemisphere_cos_weighted(
                    &[hit_nrm[0], hit_nrm[1], hit_nrm[2]],
                    &[rng.random::<f32>(), rng.random::<f32>()],
                )
                    .into();
                let refl_dir = vec3::normalize(&refl_dir);
                let env = del_geo_core::mat4_col_major::transform_homogeneous(
                    &scene.transform_world2lcl_env,
                    &refl_dir,
                )
                    .unwrap();
                let tex_coord = del_geo_core::uvec3::map_to_unit2_equal_area(&env);
                let c = del_canvas::image_interpolation::nearest::<3>(
                    &[
                        tex_coord[0] * scene.tex_shape.0 as f32,
                        (1.0 - tex_coord[1]) * scene.tex_shape.1 as f32,
                    ],
                    &scene.tex_shape,
                    &scene.tex_data,
                    false,
                );
                radiance = vec3::add(&radiance, &c);
            }
            vec3::scale_in_place(&mut radiance, 1. / (samples as f32));
            pix.0 = radiance;
        } else {
            let nrm = del_geo_core::vec3::normalize(&ray_dir);
            let env = del_geo_core::mat4_col_major::transform_homogeneous(
                &scene.transform_world2lcl_env,
                &nrm,
            )
                .unwrap();
            let tex_coord = del_geo_core::uvec3::map_to_unit2_equal_area(&env);
            pix.0 = del_canvas::image_interpolation::nearest::<3>(
                &[
                    tex_coord[0] * scene.tex_shape.0 as f32,
                    (1.0 - tex_coord[1]) * scene.tex_shape.1 as f32,
                ],
                &scene.tex_shape,
                &scene.tex_data,
                false,
            );
        }
    };

    let mut img = Vec::<image::Rgb<f32>>::new();
    img.resize(camera.img_shape.0 * camera.img_shape.1, image::Rgb([0_f32; 3]));

    use rayon::iter::IndexedParallelIterator;
    use rayon::iter::IntoParallelRefMutIterator;
    use rayon::iter::ParallelIterator;

    img.par_iter_mut()
        .enumerate()
        .for_each(|(i_pix, pix)| shoot_ray(i_pix, pix));

    let file_ms = std::fs::File::create("target/04_env_light_material_sampling.exr").unwrap();
    use image::codecs::hdr::HdrEncoder;
    let enc = HdrEncoder::new(file_ms);
    let _ = enc.encode(&img, camera.img_shape.0, camera.img_shape.1);
    Ok(())
}

pub fn light_sampling(scene: &MyScene, camera: &del_pbrt_cpu::parse_pbrt::Camera) -> anyhow::Result<(), anyhow::Error>
{
    // light sampling
    let samples = 256;

    let (texw, texh) = scene.tex_shape;

    let grayscale = calc_grayscale(&scene.tex_data, texw, texh);
    let alias_table = del_msh_cpu::alias_table::AliasTable::new(&grayscale);

    let shoot_ray = |i_pix: usize, pix: &mut [f32]| {
        use rand::Rng;
        use rand::SeedableRng;
        let mut rng = rand_chacha::ChaChaRng::seed_from_u64(i_pix as u64);
        let pix = arrayref::array_mut_ref![pix, 0, 3];
        let (ray_org, ray_dir) = del_pbrt_cpu::cam_pbrt::cast_ray_plus_z(
            (i_pix % camera.img_shape.0, i_pix / camera.img_shape.0),
            (0., 0.),
            camera.img_shape,
            camera.camera_fov,
            camera.transform_camlcl2world,
        );
        let t = del_geo_core::sphere::intersection_ray(
            scene.sphere_rad,
            &scene.sphere_cntr,
            &ray_org,
            &ray_dir,
        );
        if let Some(t) = t {
            use del_geo_core::vec3;
            let hit_pos = vec3::axpy::<f32>(t, &ray_dir, &ray_org);
            let nrm_world = vec3::sub(&hit_pos, &scene.sphere_cntr);
            let nrm_world = vec3::normalize(&nrm_world);
            let nrm_lcl = del_geo_core::mat4_col_major::transform_homogeneous(
                &scene.transform_world2lcl_env,
                &nrm_world,
            )
                .unwrap();

            let mut result = [0.; 3];
            for _isample in 0..samples {
                let i_tex = alias_table.sample(&mut rng);
                let iu_tex = i_tex % scene.tex_shape.0;
                let iv_tex = i_tex / scene.tex_shape.0;
                let u = (iu_tex as f32) / (scene.tex_shape.0 as f32);
                let v = (iv_tex as f32) / (scene.tex_shape.1 as f32);
                let ray_lcl = del_geo_core::unit2::to_uvec3_equal_area(&[u,1.-v]);
                let costheta = del_geo_core::vec3::dot(&ray_lcl, &nrm_lcl);
                let radiance = arrayref::array_ref![scene.tex_data,i_tex* 3 , 3];
                let pdf = grayscale[i_tex] / alias_table.sum_w as f32;
                let pdf = pdf * (scene.tex_shape.0 * scene.tex_shape.1) as f32;
                // joint probability of point (samplex,sampley)
                if costheta <= 0. || pdf <= 0. {
                    continue;
                }
                let radiance = vec3::scale(&radiance, costheta / pdf);
                result = vec3::add(&result, &radiance);
            }
            vec3::scale_in_place(&mut result, 4.0 / samples as f32);
            *pix = result;
        } else {
            let nrm = del_geo_core::vec3::normalize(&ray_dir);
            let env = del_geo_core::mat4_col_major::transform_homogeneous(
                &scene.transform_world2lcl_env,
                &nrm,
            )
                .unwrap();
            let tex_coord = del_geo_core::uvec3::map_to_unit2_equal_area(&env);
            *pix = del_canvas::image_interpolation::nearest::<3>(
                &[
                    tex_coord[0] * scene.tex_shape.0 as f32,
                    (1.0 - tex_coord[1]) * scene.tex_shape.1 as f32,
                ],
                &scene.tex_shape,
                &scene.tex_data,
                false,
            );
        }
    };

    let mut img = vec![0f32; camera.img_shape.0 * camera.img_shape.1 * 3];
    use rayon::iter::IndexedParallelIterator;
    use rayon::iter::ParallelIterator;
    use rayon::prelude::*;
    img.par_chunks_mut(3)
        .enumerate()
        .for_each(|(i_pix, pix)| shoot_ray(i_pix, pix));
    del_canvas::write_hdr_file("target/04_env_light_sampling.exr", camera.img_shape, &img)?;
    Ok(())
}

fn parse_pbrt_file(file_path: &str) -> anyhow::Result<(MyScene, del_pbrt_cpu::parse_pbrt::Camera)> {
    let scene = del_pbrt_pbrt4_parser::Scene::from_file(file_path)?;
    /*
    {
        let film = scene.film.as_ref().unwrap();
        let filename = &film.filename;
        dbg!(filename);
    }
     */
    assert_eq!(scene.lights.len(), 1);
    let path_env = {
        use std::path::Path;
        let lcl_path_env = match &scene.lights[0].params {
            del_pbrt_pbrt4_parser::types::Light::Infinite { filename, spectrum } => {
                let filename = filename.clone().unwrap();
                let filename = filename
                    .strip_prefix('"')
                    .unwrap()
                    .strip_suffix('"')
                    .unwrap();
                filename.to_string()
            }
            _ => {
                todo!()
            }
        };
        let path = Path::new(file_path);
        path.with_file_name(lcl_path_env)
    };
    let transform_lcl2world_env = scene.lights[0].transform.to_cols_array();
    let transform_world2lcl_env =
        del_geo_core::mat4_col_major::try_inverse_with_pivot(&transform_lcl2world_env).unwrap();
    let (tex_shape, tex_data) = {
        let pfm = del_pbrt_cpu::io_pfm::PFM::read_from(path_env)?;
        ((pfm.w, pfm.h), pfm.data)
    };
    let camera = del_pbrt_cpu::parse_pbrt::camera(&scene);
    let materials = del_pbrt_cpu::parse_pbrt::parse_material(&scene);
    let shape_entities = del_pbrt_cpu::parse_pbrt::parse_shapes(&scene);
    assert_eq!(shape_entities.len(), 1);
    //
    // Get the area light source
    let rad = match shape_entities[0].shape {
        del_pbrt_cpu::shape::ShapeType::Sphere { radius: rad } => rad,
        _ => panic!(),
    };
    let sphere_cntr = del_geo_core::mat4_col_major::to_vec3_translation(
        &shape_entities[0].transform_objlcl2world,
    );
    let scene = MyScene {
        sphere_rad: rad,
        sphere_cntr,
        transform_lcl2world_env,
        transform_world2lcl_env,
        tex_shape,
        tex_data,
        materials,
    };
    Ok((scene, camera))
}

fn main() -> anyhow::Result<()> {
    let pbrt_file_path = "asset/env_light/scene-v4.pbrt";
    let (scene, camera) = parse_pbrt_file(pbrt_file_path)?;
    {
        use image::Pixel;
        let img: Vec<image::Rgb<f32>> = scene
            .tex_data
            .chunks(3)
            .map(|rgb| *image::Rgb::<f32>::from_slice(rgb))
            .collect();
        use image::codecs::hdr::HdrEncoder;
        let file = std::fs::File::create("target/04_env_light_pfm.exr").unwrap();
        let enc = HdrEncoder::new(file);
        let _ = enc.encode(&img, scene.tex_shape.0, scene.tex_shape.1);
    }
    // --------------------
    {
        // mirror reflection
        let shoot_ray = |i_pix: usize, pix: &mut [f32]| {
            let pix = arrayref::array_mut_ref![pix, 0, 3];
            let (ray_org, ray_dir) = del_pbrt_cpu::cam_pbrt::cast_ray_plus_z(
                (i_pix % camera.img_shape.0, i_pix / camera.img_shape.0),
                (0., 0.),
                camera.img_shape,
                camera.camera_fov,
                camera.transform_camlcl2world,
            );
            let t = del_geo_core::sphere::intersection_ray(
                scene.sphere_rad,
                &scene.sphere_cntr,
                &ray_org,
                &ray_dir,
            );
            if let Some(t) = t {
                use del_geo_core::vec3;
                let pos = vec3::axpy::<f32>(t, &ray_dir, &ray_org);
                let nrm = vec3::sub(&pos, &scene.sphere_cntr);
                let hit_nrm = vec3::normalize(&nrm);
                let refl = vec3::mirror_reflection(&ray_dir, &hit_nrm);
                let refl = vec3::normalize(&refl);
                let env = del_geo_core::mat4_col_major::transform_homogeneous(
                    &scene.transform_world2lcl_env,
                    &refl,
                )
                .unwrap();
                let tex_coord = del_geo_core::uvec3::map_to_unit2_equal_area(&env);
                *pix = del_canvas::image_interpolation::nearest::<3>(
                    &[
                        tex_coord[0] * scene.tex_shape.0 as f32,
                        (1.0 - tex_coord[1]) * scene.tex_shape.1 as f32,
                    ],
                    &scene.tex_shape,
                    &scene.tex_data,
                    false,
                );
            } else {
                let nrm = del_geo_core::vec3::normalize(&ray_dir);
                let env = del_geo_core::mat4_col_major::transform_homogeneous(
                    &scene.transform_world2lcl_env,
                    &nrm,
                )
                .unwrap();
                let env = del_geo_core::vec3::normalize(&env);
                let tex_coord = del_geo_core::uvec3::map_to_unit2_equal_area(&env);
                *pix = del_canvas::image_interpolation::nearest::<3>(
                    &[
                        tex_coord[0] * scene.tex_shape.0 as f32,
                        (1.0 - tex_coord[1]) * scene.tex_shape.1 as f32,
                    ],
                    &scene.tex_shape,
                    &scene.tex_data,
                    false,
                );
            }
        };
        use rayon::iter::IndexedParallelIterator;
        use rayon::iter::ParallelIterator;
        use rayon::prelude::ParallelSliceMut;
        let mut img = vec![0f32; camera.img_shape.0 * camera.img_shape.1 * 3];
        img.par_chunks_mut(3)
            .enumerate()
            .for_each(|(i_pix, pix)| shoot_ray(i_pix, pix));
        del_canvas::write_hdr_file("target/04_env_light.exr", camera.img_shape, &img)?;
    }

    material_sampling(&scene, &camera).unwrap();
    light_sampling(&scene, &camera).unwrap();

    Ok(())
}
