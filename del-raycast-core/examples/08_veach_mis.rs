#[derive(Default, Clone)]
struct AreaLightGeometry {
    pub i_shape_entity: usize,
    pub area: f32,
    pub cog: [f32; 3],
}

struct MyScene {
    shape_entities: Vec<del_raycast_core::shape::ShapeEntity>,
    area_lights: Vec<del_raycast_core::area_light::AreaLight>,
    materials: Vec<del_raycast_core::material::Material>,
    area_light_geometries: Vec<AreaLightGeometry>,
    is_light_sample_uniform: bool,
}

fn parse_pbrt_file(
    file_path: &str,
) -> anyhow::Result<(MyScene, del_raycast_core::parse_pbrt::Camera)> {
    let scene = pbrt4::Scene::from_file(file_path)?;
    let camera = del_raycast_core::parse_pbrt::camera(&scene);
    let materials = del_raycast_core::parse_pbrt::parse_material(&scene);
    let area_lights = del_raycast_core::parse_pbrt::parse_area_light(&scene);
    let shape_entities = del_raycast_core::parse_pbrt::parse_shapes(&scene);
    let area_light_geometries = {
        let mut area_light_geometries = vec![AreaLightGeometry::default(); area_lights.len()];
        for (i_shape_entity, shape_entity) in shape_entities.iter().enumerate() {
            let Some(i_area_light) = shape_entity.area_light_index else {
                continue;
            };
            assert!(i_area_light < area_light_geometries.len());
            area_light_geometries[i_area_light].i_shape_entity = i_shape_entity;
            let (cog, area) = shape_entity.cog_and_area();
            area_light_geometries[i_area_light].area = area;
            area_light_geometries[i_area_light].cog = cog;
        }
        area_light_geometries
    };
    let my_scene = MyScene {
        shape_entities,
        area_lights,
        materials,
        area_light_geometries,
        is_light_sample_uniform: false,
    };
    Ok((my_scene, camera))
}

impl MyScene {
    fn build_area_light_importance_heuristic(&self, hit_pos: &[f32; 3]) -> Vec<f32> {
        let mut al2mag = vec![0f32; self.area_light_geometries.len() + 1];
        for ial in 0..self.area_light_geometries.len() {
            let area = self.area_light_geometries[ial].area;
            let cog = self.area_light_geometries[ial].cog;
            let dist_sq = del_geo_core::edge3::squared_length(&cog, hit_pos);
            let emission = self.area_lights[ial].spectrum_rgb.unwrap();
            let emission = emission.iter().fold(f32::NAN, |a, b| a.max(*b));
            let mag = area * emission / dist_sq;
            al2mag[ial + 1] = al2mag[ial] + mag;
        }
        al2mag
    }

    /// # Return
    /// - `Some(radiance: [f32;3], pdf_usphere: f32, uvec_hit2light:[f32;3])`
    ///    - `pdf_usphere: f32` the pdf is computed on the unit hemisphere (pdf of light / geometric term)
    /// - `None`
    fn sample_light_uniform<Rng: rand::Rng>(
        &self,
        i_shape_entity_light: usize,
        pos_observe: &[f32; 3],
        rng: &mut Rng,
    ) -> Option<([f32; 3], f32, [f32; 3])> {
        use del_geo_core::vec3;
        let (pos_light, nrm_light, pdf_shape) =
            self.shape_entities[i_shape_entity_light].sample_uniform(&[rng.random(), rng.random()]);
        let uvec_hit2light = vec3::normalize(&vec3::sub(&pos_light, pos_observe));
        let cos_theta_light = -vec3::dot(&nrm_light, &uvec_hit2light);
        if cos_theta_light < 0. {
            return None;
        } // backside of light
        if !del_raycast_core::shape::is_visible(
            &self.shape_entities,
            pos_observe,
            &pos_light,
            i_shape_entity_light,
        ) {
            return None;
        }
        let r2 = del_geo_core::edge3::squared_length(&pos_light, pos_observe);
        let geo_term = cos_theta_light / r2;
        let i_area_light = self.shape_entities[i_shape_entity_light]
            .area_light_index
            .unwrap();
        let l_i = self.area_lights[i_area_light].spectrum_rgb.unwrap();
        Some((l_i, pdf_shape / geo_term, uvec_hit2light))
    }

    fn pdf_light_uniform(
        &self,
        i_shape_entity_light: usize,
        pos_light: &[f32; 3],
        nrm_light: &[f32; 3],
        pos_observe: &[f32; 3],
    ) -> f32 {
        use del_geo_core::vec3;
        use del_geo_core::vec3::Vec3;
        let ial = self.shape_entities[i_shape_entity_light]
            .area_light_index
            .unwrap();
        let pdf1_obj = 1.0 / self.area_light_geometries[ial].area;
        let r2 = del_geo_core::edge3::squared_length(pos_light, pos_observe);
        let uvec_hit2light = pos_light.sub(pos_observe).normalize();
        let cos_theta_light = -vec3::dot(nrm_light, &uvec_hit2light);
        if cos_theta_light <= 0. {
            return f32::EPSILON;
        } // backside of light
        let geo_term = cos_theta_light / r2;
        pdf1_obj / geo_term
    }

    /// # Returns
    /// (pdf_usphere: f32)
    /// pdf_usphere
    fn sample_light_visible<Rng: rand::Rng>(
        &self,
        i_shape_entity_light: usize,
        pos_observe: &[f32; 3],
        rng: &mut Rng,
    ) -> Option<([f32; 3], f32, [f32; 3])> {
        // sampling light on the unit sphere around `pos_observe`
        let Some((uvec_obs2light, pos_light, pdf_usphere)) = self.shape_entities
            [i_shape_entity_light]
            .sample_visible(pos_observe, &[rng.random(), rng.random()])
        else {
            return None;
        };
        // cast a shadow ray
        if !del_raycast_core::shape::is_visible(
            &self.shape_entities,
            pos_observe,
            &pos_light,
            i_shape_entity_light,
        ) {
            return None;
        }
        let i_area_light = self.shape_entities[i_shape_entity_light]
            .area_light_index
            .unwrap();
        let l_i = self.area_lights[i_area_light].spectrum_rgb.unwrap();
        Some((l_i, pdf_usphere, uvec_obs2light))
    }
    fn pdf_light_visible(&self, i_shape_entity_light: usize, pos_observe: &[f32; 3]) -> f32 {
        self.shape_entities[i_shape_entity_light].pdf_visible(pos_observe)
    }
}

impl del_raycast_core::monte_carlo_integrator::Scene for MyScene {
    fn eval_brdf(
        &self,
        i_shape_entity: usize,
        obj_nrm: &[f32; 3],
        ray_in_outward_normalized: &[f32; 3],
        ray_out_normalized: &[f32; 3],
        minimum_roughness: f32,
    ) -> [f32; 3] {
        use del_geo_core::vec3::Vec3;
        assert!((ray_in_outward_normalized.norm() - 1.0).abs() < 1.0e-5);
        let i_material = self.shape_entities[i_shape_entity].material_index.unwrap();
        del_raycast_core::material::eval_brdf(
            &self.materials[i_material],
            obj_nrm,
            ray_in_outward_normalized,
            ray_out_normalized,
            minimum_roughness,
        )
    }

    fn pdf_light(
        &self,
        pos_observe: &[f32; 3],
        pos_light: &[f32; 3],
        nrm_light: &[f32; 3],
        i_shape_element_light: usize,
    ) -> f32 {
        let ial = self.shape_entities[i_shape_element_light]
            .area_light_index
            .unwrap();
        let al2mag = self.build_area_light_importance_heuristic(pos_observe);
        let pdf0 = (al2mag[ial + 1] - al2mag[ial]) / al2mag.last().unwrap();
        let pdf1_usphere = if self.is_light_sample_uniform {
            self.pdf_light_uniform(i_shape_element_light, pos_light, nrm_light, pos_observe)
        } else {
            self.pdf_light_visible(i_shape_element_light, pos_observe)
        };
        pdf0 * pdf1_usphere
    }
    fn sample_brdf<RNG>(
        &self,
        nrm_obj: &[f32; 3],
        ray_in_uvec_outward: &[f32; 3],
        i_shape_entity: usize,
        rng: &mut RNG,
        min_roughness: f32,
    ) -> Option<([f32; 3], [f32; 3], f32)>
    where
        RNG: rand::Rng,
    {
        // dbg!(nrm_obj.dot(ray_in_uvec_outward));
        // dbg!(del_geo_core::vec3::norm(&ray_in_uvec_outward));
        let se = &self.shape_entities[i_shape_entity];
        let i_material = se.material_index.unwrap();
        assert!(i_material < self.materials.len());
        del_raycast_core::material::sample_brdf(
            &self.materials[i_material],
            nrm_obj,
            ray_in_uvec_outward,
            rng,
            min_roughness,
        )
    }
    fn hit_position_normal_emission_roughness_at_ray_intersection(
        &self,
        ray_org: &[f32; 3],
        ray_dir: &[f32; 3],
    ) -> Option<([f32; 3], [f32; 3], [f32; 3], f32, usize)> {
        let Some((t, i_shape_entity, i_elem)) =
            del_raycast_core::shape::intersection_ray_against_shape_entities(
                ray_org,
                ray_dir,
                &self.shape_entities,
            )
        else {
            return None;
        };
        let hit_pos_world = del_geo_core::vec3::axpy(t, ray_dir, ray_org);
        let se = &self.shape_entities[i_shape_entity];
        let hit_nrm_world = del_raycast_core::shape::normal_at(se, &hit_pos_world, i_elem);
        let hit_emission = if let Some(ial) = se.area_light_index {
            self.area_lights[ial].spectrum_rgb.unwrap_or([0f32; 3])
        } else {
            [0f32; 3]
        };
        let hit_roughness = {
            let i_material = self.shape_entities[i_shape_entity].material_index.unwrap();
            let material = &self.materials[i_material];
            match material {
                del_raycast_core::material::Material::None => 100f32,
                del_raycast_core::material::Material::Diff(_diff) => 100f32,
                del_raycast_core::material::Material::Cond(cond) => {
                    cond.uroughness.max(cond.vroughness)
                }
                _ => panic!("Not implement material for roughness"),
            }
        };
        Some((
            hit_pos_world,
            hit_nrm_world,
            hit_emission,
            hit_roughness,
            i_shape_entity,
        ))
    }

    /// # Return
    /// - `Some(radiance: [f32;3], pdf: f32, uvec_hit2light:[f32;3])`
    ///    - `pdf: f32` the pdf is computed on the unit hemisphere (pdf of light / geometric term)
    /// - `None`
    fn sample_light<RNG: rand::Rng>(
        &self,
        pos_observe: &[f32; 3],
        i_shape_entity_observe: usize,
        rng: &mut RNG,
    ) -> Option<([f32; 3], f32, [f32; 3])> {
        let al2mag = self.build_area_light_importance_heuristic(pos_observe);
        let (ial, _rand1, pdf0) = del_msh_cpu::cumsum::sample(&al2mag, rng.random::<f32>());
        let ise = self.area_light_geometries[ial].i_shape_entity;
        if i_shape_entity_observe == ise {
            return None;
        }
        let res = if self.is_light_sample_uniform {
            self.sample_light_uniform(ise, pos_observe, rng)
        } else {
            self.sample_light_visible(ise, pos_observe, rng)
        };
        let Some((radiance, pdf1, uvec_obsrv2light)) = res else {
            return None;
        };
        Some((radiance, pdf0 * pdf1, uvec_obsrv2light))
    }
}

enum IntegrationType {
    PathTracing,
    NextEventEstimation,
    Mis,
}

fn mc_integration(
    integration_type: IntegrationType,
    str_type: &str,
    scene: &MyScene,
    camera: &del_raycast_core::parse_pbrt::Camera,
    num_sample: usize,
    max_depth: usize,
    img_gt: &[f32],
    is_increasing_roughness: bool,
) -> anyhow::Result<()> {
    let shoot_ray = |i_pix: usize, pix: &mut [f32]| {
        let pix = arrayref::array_mut_ref!(pix, 0, 3);
        use rand::Rng;
        use rand::SeedableRng;
        let mut rng = rand_chacha::ChaChaRng::seed_from_u64(i_pix as u64);
        let mut l_o = [0., 0., 0.];
        for _i_sample in 0..num_sample {
            let (ray0_org, ray0_dir) = camera.ray(
                i_pix,
                [
                    del_raycast_core::sampling::tent(rng.random::<f32>()),
                    del_raycast_core::sampling::tent(rng.random::<f32>()),
                ],
            );
            let rad = match integration_type {
                IntegrationType::PathTracing => {
                    del_raycast_core::monte_carlo_integrator::radiance_pt(
                        &ray0_org, &ray0_dir, scene, max_depth, &mut rng,
                    )
                }
                IntegrationType::Mis => del_raycast_core::monte_carlo_integrator::radiance_mis(
                    &ray0_org,
                    &ray0_dir,
                    scene,
                    max_depth,
                    &mut rng,
                    is_increasing_roughness,
                ),
                IntegrationType::NextEventEstimation => {
                    del_raycast_core::monte_carlo_integrator::radiance_nee(
                        &ray0_org,
                        &ray0_dir,
                        scene,
                        max_depth,
                        &mut rng,
                        is_increasing_roughness,
                    )
                }
            };
            l_o = del_geo_core::vec3::add(&l_o, &rad);
        }
        *pix = del_geo_core::vec3::scale(&l_o, 1. / num_sample as f32);
    };
    let img_out = {
        let mut img_out = vec![0f32; camera.img_shape.0 * camera.img_shape.1 * 3];
        use rayon::prelude::*;
        img_out
            .par_chunks_mut(3)
            // .chunks_mut(3)
            .enumerate()
            .for_each(|(i_pix, pix)| shoot_ray(i_pix, pix));
        img_out
    };
    del_canvas::write_hdr_file(
        format!("target/08_veach_mis__{}_{}.hdr", str_type, num_sample),
        camera.img_shape,
        &img_out,
    )?;
    let path_error_map = format!(
        "target/08_veach_mis__{}_{}_error_map.hdr",
        str_type, num_sample
    );
    del_canvas::write_hdr_file_mse_rgb_error_map(
        path_error_map,
        camera.img_shape,
        img_gt,
        &img_out,
    )?;
    let err = del_canvas::rmse_error(img_gt, &img_out);
    println!("num_sample: {}, mse: {}", num_sample, err);
    Ok(())
}

fn main() -> anyhow::Result<()> {
    let pbrt_file_path = "asset/veach-mis/scene-v4.pbrt";
    let (scene, camera) = parse_pbrt_file(pbrt_file_path)?;
    del_raycast_core::shape::write_wavefront_obj_file_from_camera_view(
        "target/08_veach_mis.obj",
        &scene.shape_entities,
        &camera.transform_world2camlcl,
    )?;
    {
        // computing depth image
        let shoot_ray = |i_pix: usize, pix: &mut [f32]| {
            let pix = arrayref::array_mut_ref![pix, 0, 3];
            let (ray_org, ray_dir) = camera.ray(i_pix, [0.; 2]);
            use del_raycast_core::shape::intersection_ray_against_shape_entities;
            let t = match intersection_ray_against_shape_entities(
                &ray_org,
                &ray_dir,
                &scene.shape_entities,
            ) {
                Some((t, _ise, _ie)) => t,
                None => f32::INFINITY,
            };
            *pix = [t * 0.05; 3];
        };
        let mut img_out = vec![0f32; camera.img_shape.0 * camera.img_shape.1 * 3];
        use rayon::prelude::*;
        img_out
            .par_chunks_mut(3)
            .enumerate()
            .for_each(|(i_pix, pix)| shoot_ray(i_pix, pix));
        del_canvas::write_hdr_file("target/08_veach_mis__depth.hdr", camera.img_shape, &img_out)?;
    }
    let img_gt = image::open("asset/veach-mis/TungstenRender.exr")
        .unwrap()
        .to_rgb32f();
    assert_eq!(
        img_gt.dimensions(),
        (camera.img_shape.0 as u32, camera.img_shape.1 as u32)
    );
    //
    println!("---------------------path tracer---------------------");
    mc_integration(
        IntegrationType::PathTracing,
        "pt",
        &scene,
        &camera,
        24,
        4,
        &img_gt,
        false,
    )?;
    // ---------------------------------
    // increasing roughness
    // ---------------------------------
    let scene = MyScene {
        shape_entities: scene.shape_entities,
        materials: scene.materials,
        area_lights: scene.area_lights,
        area_light_geometries: scene.area_light_geometries,
        is_light_sample_uniform: false,
    };
    println!("---------------------NEE tracer VisibleLightSampling IncreasingRoughness---------------------");
    mc_integration(
        IntegrationType::NextEventEstimation,
        "nee_visl_incr",
        &scene,
        &camera,
        24,
        4,
        &img_gt,
        true,
    )?;
    println!("---------------------Light sampling VisibleLightSampling IncreasingRoughness---------------------");
    mc_integration(
        IntegrationType::NextEventEstimation,
        "ls_visl_incr",
        &scene,
        &camera,
        24,
        1,
        &img_gt,
        true,
    )?;
    println!("---------------------MIS sampling VisibleLightSampling IncreasingRoughness---------------------");
    mc_integration(
        IntegrationType::Mis,
        "mis_visl_incr",
        &scene,
        &camera,
        24,
        4,
        &img_gt,
        true,
    )?;
    // ----------------------------------------------------
    // Default unbiased
    // ----------------------------------------------------
    println!("---------------------NEE tracer---------------------");
    mc_integration(
        IntegrationType::NextEventEstimation,
        "nee_visl",
        &scene,
        &camera,
        24,
        4,
        &img_gt,
        false,
    )?;
    println!("---------------------Light sampling---------------------");
    mc_integration(
        IntegrationType::NextEventEstimation,
        "ls_visl",
        &scene,
        &camera,
        24,
        1,
        &img_gt,
        false,
    )?;
    println!("---------------------MIS sampling---------------------");
    mc_integration(
        IntegrationType::Mis,
        "mis_visl",
        &scene,
        &camera,
        24,
        4,
        &img_gt,
        false,
    )?;
    // --------------------------------------
    let scene = MyScene {
        shape_entities: scene.shape_entities,
        materials: scene.materials,
        area_lights: scene.area_lights,
        area_light_geometries: scene.area_light_geometries,
        is_light_sample_uniform: true,
    };
    println!("---------------------NEE tracer UniformLightSampling IncreasingRoughness---------------------");
    mc_integration(
        IntegrationType::NextEventEstimation,
        "nee_unil_incr",
        &scene,
        &camera,
        24,
        4,
        &img_gt,
        true,
    )?;
    println!("---------------------light sampling VisibleLightSampling IncreasingRoughness---------------------");
    mc_integration(
        IntegrationType::NextEventEstimation,
        "ls_unil_incr",
        &scene,
        &camera,
        24,
        1,
        &img_gt,
        true,
    )?;
    println!("---------------------MIS sampling VisibleLightSampling IncreasingRoughness---------------------");
    mc_integration(
        IntegrationType::Mis,
        "mis_unil_incr",
        &scene,
        &camera,
        24,
        4,
        &img_gt,
        true,
    )?;
    Ok(())
}
