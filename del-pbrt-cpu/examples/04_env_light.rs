use arrayref::array_mut_ref;

fn main() -> anyhow::Result<()> {
    let (tex_shape, tex_data) = {
        let pfm = del_raycast_core::io_pfm::PFM::read_from(
            "asset/material-testball/textures/envmap.pfm",
        )?;
        ((pfm.w, pfm.h), pfm.data)
    };
    {
        use image::Pixel;
        let img: Vec<image::Rgb<f32>> = tex_data
            .chunks(3)
            .map(|rgb| *image::Rgb::<f32>::from_slice(rgb))
            .collect();
        use image::codecs::hdr::HdrEncoder;
        let file = std::fs::File::create("target/04_env_light_pfm.hdr").unwrap();
        let enc = HdrEncoder::new(file);
        let _ = enc.encode(&img, tex_shape.0, tex_shape.1);
    }
    // --------------------
    let camera_fov = 20.0;
    // let transform_cam_lcl2glbl = del_geo_core::mat4_col_major::from_translate(&[0., 0., -5.]);
    let transform_world2camlcl: [f32; 16] = [
        0.721367, -0.373123, -0.583445, -0., -0., 0.842456, -0.538765, -0., -0.692553, -0.388647,
        -0.60772, -0., 0.0258668, -0.29189, 5.43024, 1.,
    ];
    let transform_camlcl2world =
        del_geo_core::mat4_col_major::try_inverse(&transform_world2camlcl).unwrap();
    let transform_env = [
        -0.386527, 0., 0.922278, 0., -0.922278, 0., -0.386527, 0., 0., 1., 0., 0., 0., 0., 0., 1.,
    ];
    let transform_env: [f32; 16] = {
        let m = nalgebra::Matrix4::<f32>::from_column_slice(&transform_env);
        let m = m.try_inverse().unwrap();
        // let transform_env = del_geo_core::mat4_col_major::try_inverse(&transform_env).unwrap();
        m.as_slice().try_into().unwrap()
    };
    let sphere_cntr = [0.15, 0.50, 0.16];
    let img_shape = (640, 360);
    {
        let shoot_ray = |i_pix: usize, pix: &mut [f32]| {
            let pix = array_mut_ref![pix, 0, 3];
            let (ray_org, ray_dir) = del_raycast_core::cam_pbrt::cast_ray_plus_z(
                (i_pix % img_shape.0, i_pix / img_shape.0),
                (0., 0.),
                img_shape,
                camera_fov,
                transform_camlcl2world,
            );
            let t = del_geo_core::sphere::intersection_ray(0.7, &sphere_cntr, &ray_org, &ray_dir);
            if let Some(t) = t {
                use del_geo_core::vec3;
                let pos = vec3::axpy::<f32>(t, &ray_dir, &ray_org);
                let nrm = vec3::sub(&pos, &sphere_cntr);
                let hit_nrm = vec3::normalize(&nrm);
                let refl = vec3::mirror_reflection(&ray_dir, &hit_nrm);
                let refl = vec3::normalize(&refl);
                let env =
                    del_geo_core::mat4_col_major::transform_homogeneous(&transform_env, &refl)
                        .unwrap();
                let tex_coord = del_geo_core::uvec3::map_to_unit2_equal_area(&env);
                *pix = del_canvas::texture::nearest_integer_center::<3>(
                    &[
                        tex_coord[0] * tex_shape.0 as f32,
                        tex_coord[1] * tex_shape.1 as f32,
                    ],
                    &tex_shape,
                    &tex_data,
                );
            } else {
                let nrm = del_geo_core::vec3::normalize(&ray_dir);
                let env = del_geo_core::mat4_col_major::transform_homogeneous(&transform_env, &nrm)
                    .unwrap();
                let tex_coord = del_geo_core::uvec3::map_to_unit2_equal_area(&env);
                *pix = del_canvas::texture::nearest_integer_center::<3>(
                    &[
                        tex_coord[0] * tex_shape.0 as f32,
                        tex_coord[1] * tex_shape.1 as f32,
                    ],
                    &tex_shape,
                    &tex_data,
                );
            }
        };
        use rayon::iter::IndexedParallelIterator;
        use rayon::iter::ParallelIterator;
        use rayon::prelude::ParallelSliceMut;
        let mut img = vec![0f32; img_shape.0 * img_shape.1 * 3];
        img.par_chunks_mut(3)
            .enumerate()
            .for_each(|(i_pix, pix)| shoot_ray(i_pix, pix));
        del_canvas::write_hdr_file("target/04_env_light.hdr", img_shape, &img)?;
    }

    {
        // material sampling
        let samples = 64;
        let shoot_ray = |i_pix: usize, pix: &mut image::Rgb<f32>| {
            let ih = i_pix / img_shape.0;
            let iw = i_pix % img_shape.0;
            let (ray_org, ray_dir) = del_raycast_core::cam_pbrt::cast_ray_plus_z(
                (iw, ih),
                (0., 0.),
                img_shape,
                camera_fov,
                transform_camlcl2world,
            );
            let t = del_geo_core::sphere::intersection_ray(0.7, &sphere_cntr, &ray_org, &ray_dir);
            if let Some(t) = t {
                use del_geo_core::vec3;
                let hit_pos = vec3::axpy::<f32>(t, &ray_dir, &ray_org);
                let hit_nrm = vec3::sub(&hit_pos, &sphere_cntr);
                let hit_nrm = vec3::normalize(&hit_nrm);
                let mut radiance = [0.; 3];
                use rand::Rng;
                use rand::SeedableRng;
                let mut rng = rand_chacha::ChaChaRng::seed_from_u64(i_pix as u64);
                for _isample in 0..samples {
                    let refl_dir: [f32; 3] = del_raycast_core::sampling::hemisphere_cos_weighted(
                        &[hit_nrm[0], hit_nrm[1], hit_nrm[2]],
                        &[rng.random::<f32>(), rng.random::<f32>()],
                    )
                    .into();
                    let refl_dir = vec3::normalize(&refl_dir);
                    let env = del_geo_core::mat4_col_major::transform_homogeneous(
                        &transform_env,
                        &refl_dir,
                    )
                    .unwrap();
                    let tex_coord = del_geo_core::uvec3::map_to_unit2_equal_area(&env);
                    let c = del_canvas::texture::nearest_integer_center::<3>(
                        &[
                            tex_coord[0] * tex_shape.0 as f32,
                            tex_coord[1] * tex_shape.1 as f32,
                        ],
                        &tex_shape,
                        &tex_data,
                    );
                    radiance = vec3::add(&radiance, &c);
                }
                vec3::scale_in_place(&mut radiance, 1. / (samples as f32));
                pix.0 = radiance;
            } else {
                let nrm = del_geo_core::vec3::normalize(&ray_dir);
                let env = del_geo_core::mat4_col_major::transform_homogeneous(&transform_env, &nrm)
                    .unwrap();
                let tex_coord = del_geo_core::uvec3::map_to_unit2_equal_area(&env);
                pix.0 = del_canvas::texture::nearest_integer_center::<3>(
                    &[
                        tex_coord[0] * tex_shape.0 as f32,
                        tex_coord[1] * tex_shape.1 as f32,
                    ],
                    &tex_shape,
                    &tex_data,
                );
            }
        };

        let mut img = Vec::<image::Rgb<f32>>::new();
        img.resize(img_shape.0 * img_shape.1, image::Rgb([0_f32; 3]));

        use rayon::iter::IndexedParallelIterator;
        use rayon::iter::IntoParallelRefMutIterator;
        use rayon::iter::ParallelIterator;

        img.par_iter_mut()
            .enumerate()
            .for_each(|(i_pix, pix)| shoot_ray(i_pix, pix));

        let file_ms = std::fs::File::create("target/04_env_light_material_sampling.hdr").unwrap();
        use image::codecs::hdr::HdrEncoder;
        let enc = HdrEncoder::new(file_ms);
        let _ = enc.encode(&img, img_shape.0, img_shape.1);
    }

    {
        // light sampling
        use del_raycast_core::env_map::*;
        use image::Pixel;
        let samples = 64;
        let img: Vec<image::Rgb<f32>> = tex_data
            .chunks(3)
            .map(|rgb| *image::Rgb::<f32>::from_slice(rgb))
            .collect();

        let texw = tex_shape.0;
        let texh = tex_shape.1;

        let grayscale = calc_grayscale(&img, texw, texh);
        let itgr = calc_integral_over_grayscale(&grayscale, texw, texh);
        let (marginal_map, conditional_map) = calc_inverse_cdf_map(&grayscale, itgr, texw, texh);

        // uncomment to debug inverse cdf sampling
        /*
        let mut towrite: Vec<image::Rgb<f32>> = tex_data
            .chunks(3)
            .map(|rgb| *image::Rgb::<f32>::from_slice(rgb))
            .collect();

        for _isample in 0..1024 * 1024 {
            let r_x: f32 = del_raycast_core::sampling::radical_inverse(_isample, 2);
            let r_y: f32 = del_raycast_core::sampling::radical_inverse(_isample, 3);

            let sampley = marginal_map[tex2pixel(r_y, texh)][0];
            let samplex =
                conditional_map[tex2pixel(r_x, texw) + tex2pixel(sampley, texh) * texw][0];

            let pixelx = tex2pixel(samplex, texw);
            let pixely = tex2pixel(sampley, texh);

            towrite[pixely * texw + pixelx] = image::Rgb([1., 0., 0.]);
        }

        let file_ms = std::fs::File::create("target/04_env_light_debug.hdr").unwrap();
        let enc = HdrEncoder::new(file_ms);
        let _ = enc.encode(&towrite, texw, texh);
        return Ok(());
        */

        let shoot_ray = |i_pix: usize, pix: &mut [f32]| {
            let pix = arrayref::array_mut_ref![pix, 0, 3];
            let (ray_org, ray_dir) = del_raycast_core::cam_pbrt::cast_ray_plus_z(
                (i_pix % img_shape.0, i_pix / img_shape.0),
                (0., 0.),
                img_shape,
                camera_fov,
                transform_camlcl2world,
            );
            let t = del_geo_core::sphere::intersection_ray(0.7, &sphere_cntr, &ray_org, &ray_dir);
            if let Some(t) = t {
                use del_geo_core::vec3;
                let hit_pos = vec3::axpy::<f32>(t, &ray_dir, &ray_org);
                let hit_nrm = vec3::sub(&hit_pos, &sphere_cntr);
                let hit_nrm = vec3::normalize(&hit_nrm);
                let nrm =
                    del_geo_core::mat4_col_major::transform_homogeneous(&transform_env, &hit_nrm)
                        .unwrap();

                let mut result = [0.; 3];
                for _isample in 0..samples {
                    let r_x: f32 = del_raycast_core::sampling::radical_inverse(_isample, 2);
                    let r_y: f32 = del_raycast_core::sampling::radical_inverse(_isample, 3);

                    let sampley = marginal_map[tex2pixel(r_y, texh)][0];
                    let samplex =
                        conditional_map[tex2pixel(r_x, texw) + tex2pixel(sampley, texh) * texw][0];

                    let pixelx = tex2pixel(samplex, texw);
                    let pixely = tex2pixel(sampley, texh);

                    let sample_ray = envmap2unitsphere(&[samplex, 1. - sampley]);

                    let costheta = del_geo_core::vec3::dot(&nrm, &sample_ray);

                    // joint probability of point (samplex,sampley)
                    let pdf = grayscale[pixely * texw + pixelx][0] / itgr;
                    if costheta <= 0. || pdf <= 0. {
                        continue;
                    }
                    let mut radiance = img[pixely * texw + pixelx].0;
                    del_geo_core::vec3::scale_in_place(&mut radiance, costheta / pdf);
                    result = vec3::add(&result, &radiance);
                }
                vec3::scale_in_place(&mut result, 4. / samples as f32);
                *pix = result;
            } else {
                let nrm = del_geo_core::vec3::normalize(&ray_dir);
                let env = del_geo_core::mat4_col_major::transform_homogeneous(&transform_env, &nrm)
                    .unwrap();
                let tex_coord = del_geo_core::uvec3::map_to_unit2_equal_area(&env);
                *pix = del_canvas::texture::nearest_integer_center::<3>(
                    &[
                        tex_coord[0] * tex_shape.0 as f32,
                        tex_coord[1] * tex_shape.1 as f32,
                    ],
                    &tex_shape,
                    &tex_data,
                );
            }
        };

        let mut img = vec![0f32; img_shape.0 * img_shape.1 * 3];
        use rayon::iter::IndexedParallelIterator;
        use rayon::iter::ParallelIterator;
        use rayon::prelude::*;
        img.par_chunks_mut(3)
            .enumerate()
            .for_each(|(i_pix, pix)| shoot_ray(i_pix, pix));
        del_canvas::write_hdr_file("target/04_env_light_sampling.hdr", img_shape, &img)?;
    }

    Ok(())
}
