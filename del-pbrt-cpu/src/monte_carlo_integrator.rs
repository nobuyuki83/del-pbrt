pub trait Scene {
    #[allow(clippy::type_complexity)]
    fn hit_position_normal_emission_roughness_at_ray_intersection(
        &self,
        ray_org: &[f32; 3],
        ray_dir: &[f32; 3],
    ) -> Option<([f32; 3], [f32; 3], [f32; 3], f32, usize)>;

    fn eval_brdf(
        &self,
        itrimsh: usize,
        obj_nrm: &[f32; 3],
        ray_in_outward_normlized: &[f32; 3],
        ray_out_normalized: &[f32; 3],
        minimum_roughness: f32,
    ) -> [f32; 3];

    /// `uvec_ray_in_outward` should be facing outward (same direction as `obj_nrm`)
    fn sample_brdf<Rng: rand::Rng>(
        &self,
        obj_nrm: &[f32; 3],
        uvec_ray_in_outward: &[f32; 3],
        i_shape_entity: usize,
        rng: &mut Rng,
        minimum_roughness: f32,
    ) -> Option<([f32; 3], [f32; 3], f32)>;

    /// # Return
    /// - `Some(radiance: [f32;3], pdf: f32, uvec_hit2light:[f32;3])`
    ///    - `pdf: f32` the pdf is computed on the unit hemisphere (pdf of light / geometric term)
    /// - `None`
    fn sample_light<Rng: rand::Rng>(
        &self,
        pos_observeffset: &[f32; 3],
        i_shape_entity_observe: usize,
        rng: &mut Rng,
    ) -> Option<([f32; 3], f32, [f32; 3])>;

    /// pdf should be the density on the unit sphere around the `pos_observe`
    fn pdf_light(
        &self,
        pos_observe: &[f32; 3],
        pos_light: &[f32; 3],
        nrm_light: &[f32; 3],
        i_shape_entity: usize,
    ) -> f32;
}

pub fn radiance_pt<RNG, SCENE>(
    ray_org_ini: &[f32; 3],
    ray_dir_ini: &[f32; 3],
    scene: &SCENE,
    max_depth: usize,
    rng: &mut RNG,
) -> [f32; 3]
where
    RNG: rand::Rng,
    SCENE: Scene,
{
    use del_geo_core::vec3::Vec3;
    let mut rad_out = [0f32; 3];
    let mut throughput = [1f32; 3];
    let mut ray_org: [f32; 3] = ray_org_ini.to_owned();
    let mut ray_dir: [f32; 3] = ray_dir_ini.to_owned();
    for _i_depth in 0..max_depth {
        let Some((hit_pos, hit_nrm, hit_emission, _hit_roughness, hit_itrimsh)) =
            scene.hit_position_normal_emission_roughness_at_ray_intersection(&ray_org, &ray_dir)
        else {
            break;
        };
        rad_out = rad_out.add(&hit_emission.element_wise_mult(&throughput));
        //
        let Some((ray_dir_next, brdf, pdf)) = scene.sample_brdf(
            &hit_nrm,
            &ray_dir.scale(-1f32).normalize(),
            hit_itrimsh,
            rng,
            0.0,
        ) else {
            break;
        };
        let cos_hit = ray_dir_next.dot(&hit_nrm);
        throughput = throughput.element_wise_mult(&brdf.scale(cos_hit / pdf));
        {
            let russian_roulette_prob = throughput.iter().fold(f32::NAN, |a, b| a.max(*b));
            if rng.random::<f32>() < russian_roulette_prob {
                throughput = del_geo_core::vec3::scale(&throughput, 1.0 / russian_roulette_prob);
            } else {
                break; // terminate ray
            }
        }
        let hit_pos_w_offset = del_geo_core::vec3::axpy(1.0e-3, &hit_nrm, &hit_pos);
        ray_org = hit_pos_w_offset;
        ray_dir = ray_dir_next;
    }
    rad_out
}

pub fn radiance_nee<RNG, SCENE>(
    ray_org_ini: &[f32; 3],
    ray_dir_ini: &[f32; 3],
    scene: &SCENE,
    max_depth: usize,
    rng: &mut RNG,
    is_increasing_roughness: bool,
) -> [f32; 3]
where
    RNG: rand::Rng,
    SCENE: Scene,
{
    use del_geo_core::vec3::Vec3;
    let mut rad_out = [0f32; 3];
    let mut throughput = [1f32; 3];
    let mut ray_org: [f32; 3] = ray_org_ini.to_owned();
    let mut ray_dir: [f32; 3] = ray_dir_ini.to_owned();
    let mut max_roughness = 0f32;
    for i_depth in 0..max_depth {
        use del_geo_core::vec3;
        let Some((hit_pos, hit_nrm, hit_emission, hit_roughness, hit_i_shape_entity)) =
            scene.hit_position_normal_emission_roughness_at_ray_intersection(&ray_org, &ray_dir)
        else {
            break;
        };
        if is_increasing_roughness {
            max_roughness = max_roughness.max(hit_roughness);
        }
        // println!("{} {} {}", i_depth, hit_roughness, max_roughness);
        // ------------
        if i_depth == 0 {
            rad_out = rad_out.add(&hit_emission.element_wise_mult(&throughput));
        };
        let hit_pos_w_offset = vec3::axpy(1.0e-3, &hit_nrm, &hit_pos);
        if hit_emission == [0f32; 3] {
            // sample light
            if let Some((li_light, pdf_light, uvec_hit2light)) =
                scene.sample_light(&hit_pos_w_offset, hit_i_shape_entity, rng)
            {
                let brdf_hit = scene.eval_brdf(
                    hit_i_shape_entity,
                    &hit_nrm,
                    &ray_dir.scale(-1.).normalize(),
                    &uvec_hit2light,
                    max_roughness,
                );
                let cos_hit = vec3::dot(&uvec_hit2light, &hit_nrm);
                let lo_light =
                    vec3::element_wise_mult(&brdf_hit, &li_light.scale(cos_hit / pdf_light));
                rad_out = rad_out.add(&lo_light.element_wise_mult(&throughput));
            }
        }
        if i_depth == max_depth - 1 {
            break;
        }
        let ray_dir_next = {
            let Some((ray_dir_next, brdf, pdf)) = scene.sample_brdf(
                &hit_nrm,
                &ray_dir.scale(-1f32).normalize(),
                hit_i_shape_entity,
                rng,
                max_roughness,
            ) else {
                break;
            };
            let cos_hit = ray_dir_next.dot(&hit_nrm); //.clamp(f32::EPSILON, 1f32);
            throughput = throughput.element_wise_mult(&brdf.scale(cos_hit / pdf));
            ray_dir_next
        };
        {
            // russian roulette
            let &russian_roulette_prob = throughput
                .iter()
                .max_by(|&a, &b| a.partial_cmp(b).unwrap())
                .unwrap();
            if rng.random::<f32>() < russian_roulette_prob {
                throughput = vec3::scale(&throughput, 1.0 / russian_roulette_prob);
            } else {
                break; // terminate ray
            }
        }
        ray_org = hit_pos_w_offset;
        ray_dir = ray_dir_next;
    }
    rad_out
}

pub fn radiance_mis<RNG, SCENE>(
    ray_org_ini: &[f32; 3],
    ray_dir_ini: &[f32; 3],
    scene: &SCENE,
    max_depth: usize,
    rng: &mut RNG,
    is_increasing_roughness: bool,
) -> [f32; 3]
where
    RNG: rand::Rng,
    SCENE: Scene,
{
    use del_geo_core::vec3::Vec3;
    let mut rad_out = [0f32; 3];
    let mut throughput = [1f32; 3];
    let mut ray_org: [f32; 3] = ray_org_ini.to_owned();
    let mut ray_dir: [f32; 3] = ray_dir_ini.to_owned();
    let mut max_roughness = 0f32;
    for i_depth in 0..max_depth {
        use del_geo_core::vec3;
        let Some((hit_pos, hit_nrm, hit_emission, hit_roughness, hit_i_shape_entity)) =
            scene.hit_position_normal_emission_roughness_at_ray_intersection(&ray_org, &ray_dir)
        else {
            break;
        };
        if is_increasing_roughness {
            max_roughness = max_roughness.max(hit_roughness);
        }
        // ------------
        if i_depth == 0 {
            rad_out = rad_out.add(&hit_emission.element_wise_mult(&throughput));
        };
        let hit_pos_w_offset = vec3::axpy(1.0e-3, &hit_nrm, &hit_pos);
        if hit_emission == [0f32; 3] {
            // sample light seeking for direct light
            if let Some((li_light, pdf_light, uvec_hit2light)) =
                scene.sample_light(&hit_pos_w_offset, hit_i_shape_entity, rng)
            {
                let brdf_hit = scene.eval_brdf(
                    hit_i_shape_entity,
                    &hit_nrm,
                    &ray_dir.scale(-1.).normalize(),
                    &uvec_hit2light,
                    max_roughness,
                );
                let cos_hit = vec3::dot(&uvec_hit2light, &hit_nrm);
                let pdf_brdf = cos_hit * std::f32::consts::FRAC_1_PI;
                let mis_weight_light = pdf_light / (pdf_light + pdf_brdf);
                let lo_light = vec3::element_wise_mult(
                    &brdf_hit,
                    &li_light.scale(cos_hit / pdf_light * mis_weight_light),
                );
                rad_out = rad_out.add(&lo_light.element_wise_mult(&throughput));
            }
        }
        if hit_emission == [0f32; 3] {
            // sample material seeking for direct light
            let Some((ray_dir_brdf, brdf, pdf_brdf)) = scene.sample_brdf(
                &hit_nrm,
                &ray_dir.scale(-1f32).normalize(),
                hit_i_shape_entity,
                rng,
                max_roughness,
            ) else {
                break;
            };
            if let Some((
                hit_pos_light,
                hit_nrm_light,
                hit_emission_light,
                _hit_roughness,
                hit_i_shape_entity_light,
            )) = scene.hit_position_normal_emission_roughness_at_ray_intersection(
                &hit_pos_w_offset,
                &ray_dir_brdf,
            ) {
                // the material-sampled ray hit light
                if hit_emission_light != [0f32; 3] {
                    let cos_hit = ray_dir_brdf.dot(&hit_nrm).clamp(f32::EPSILON, 1f32);
                    let pdf_light = scene.pdf_light(
                        &hit_pos,
                        &hit_pos_light,
                        &hit_nrm_light,
                        hit_i_shape_entity_light,
                    );
                    let mis_weight_brdf = pdf_brdf / (pdf_brdf + pdf_light);
                    let lo_brdf = hit_emission_light
                        .element_wise_mult(&brdf.scale(cos_hit / pdf_brdf * mis_weight_brdf));
                    rad_out = rad_out.add(&lo_brdf.element_wise_mult(&throughput));
                }
            }
        }
        let ray_dir_next = {
            // update throughput
            let Some((ray_dir_next, brdf, pdf_brdf)) = scene.sample_brdf(
                &hit_nrm,
                &ray_dir.scale(-1f32).normalize(),
                hit_i_shape_entity,
                rng,
                max_roughness,
            ) else {
                break;
            };
            let cosine = ray_dir_next.dot(&hit_nrm).clamp(f32::EPSILON, 1f32);
            throughput = throughput.element_wise_mult(&brdf.scale(cosine / pdf_brdf));
            ray_dir_next
        };
        {
            let &russian_roulette_prob = throughput
                .iter()
                .max_by(|&a, &b| a.partial_cmp(b).unwrap())
                .unwrap();
            if rng.random::<f32>() < russian_roulette_prob {
                throughput = vec3::scale(&throughput, 1.0 / russian_roulette_prob);
            } else {
                break; // terminate ray
            }
        }
        ray_org = hit_pos_w_offset;
        ray_dir = ray_dir_next;
    }
    rad_out
}
