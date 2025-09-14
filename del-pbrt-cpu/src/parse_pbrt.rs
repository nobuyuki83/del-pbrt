use crate::area_light::AreaLight;
use crate::shape::ShapeEntity;
use ply_rs::ply::PropertyAccess;

pub struct Camera {
    pub camera_fov: f32,
    /// this transformation actually flip the scene in x direction.
    /// the camera is looking from -z to +z direction
    pub transform_world2camlcl: [f32; 16],
    /// inverse of `transform_world2camlcl'
    pub transform_camlcl2world: [f32; 16],
    pub img_shape: (usize, usize),
}

pub fn camera(scene: &pbrt4::Scene) -> Camera {
    let camera = scene.camera.as_ref().unwrap();
    let transform = camera.transform.to_cols_array();
    // dbg!(&camera.params);
    let fov = match camera.params {
        pbrt4::types::Camera::Perspective { fov, .. } => fov,
        _ => {
            todo!()
        }
    };
    let film = scene.film.as_ref().unwrap();
    let img_shape = (film.xresolution as usize, film.yresolution as usize);
    Camera {
        camera_fov: fov,
        transform_world2camlcl: transform,
        transform_camlcl2world: del_geo_core::mat4_col_major::try_inverse(&transform).unwrap(),
        img_shape,
    }
}

impl Camera {
    /// # Return
    /// `(ray_org: [f32;3], ray_dir [f32;3])`
    /// ray_org: camera focus point
    /// ray_dir: ray direction (not normalized)
    pub fn ray(&self, i_pix: usize, offset: [f32; 2]) -> ([f32; 3], [f32; 3]) {
        crate::cam_pbrt::cast_ray_plus_z(
            (i_pix % self.img_shape.0, i_pix / self.img_shape.0),
            offset.into(),
            self.img_shape,
            self.camera_fov,
            self.transform_camlcl2world,
        )
    }
}

// material index, area light index, tri2vtx, vtx2xyz,normals,uvs
type Trimesh3Data = (
    usize,
    Option<usize>,
    Vec<usize>,
    Vec<f32>,
    Vec<f32>,
    Vec<f32>,
);

#[allow(clippy::type_complexity)]
pub fn trimesh3_from_shape_entity(
    shape_entity: &pbrt4::ShapeEntity,
    path_file: &str,
) -> Option<Trimesh3Data> {
    match &shape_entity.params {
        pbrt4::types::Shape::TriangleMesh {
            positions,
            indices,
            normals,
            uvs,
            ..
        } => {
            let tri2vtx = indices.iter().map(|&v| v as usize).collect::<Vec<usize>>();
            Some((
                shape_entity.material_index.unwrap(),
                shape_entity.area_light_index,
                tri2vtx,
                positions.to_vec(),
                normals.to_vec(),
                uvs.to_vec(),
            ))
        }
        pbrt4::types::Shape::PlyMesh { filename } => {
            let filename = filename.strip_suffix("\"").unwrap().to_string();
            let filename = filename.strip_prefix("\"").unwrap().to_string();
            let path = std::path::Path::new(path_file);
            let path = path.parent().unwrap();
            let path = path.to_str().unwrap().to_string() + "/" + &filename;
            let mut f = std::fs::File::open(path).unwrap();
            let p = ply_rs::parser::Parser::<ply_rs::ply::DefaultElement>::new();
            let ply = p.read_ply(&mut f).unwrap();
            let mut vtx2xyz: Vec<f32> = vec![];
            let mut tri2vtx: Vec<usize> = vec![];
            for (str, vals) in ply.payload {
                if str == "vertex" {
                    vtx2xyz.resize(vals.len() * 3, 0f32);
                    for (i_vtx, val) in vals.iter().enumerate() {
                        vtx2xyz[i_vtx * 3] = val.get_float(&"x".to_string()).unwrap();
                        vtx2xyz[i_vtx * 3 + 1] = val.get_float(&"y".to_string()).unwrap();
                        vtx2xyz[i_vtx * 3 + 2] = val.get_float(&"z".to_string()).unwrap();
                        /*
                        let nx = val.get_float(&"nx".to_string()).unwrap();
                        let ny = val.get_float(&"ny".to_string()).unwrap();
                        let nz = val.get_float(&"nz".to_string()).unwrap();
                        let u = val.get_float(&"u".to_string()).unwrap();
                        let v = val.get_float(&"v".to_string()).unwrap();
                         */
                    }
                } else if str == "face" {
                    tri2vtx.resize(vals.len() * 3, 0usize);
                    for (i_vtx, val) in vals.iter().enumerate() {
                        let a = val.get_list_int(&"vertex_indices".to_string()).unwrap();
                        tri2vtx[i_vtx * 3] = a[0] as usize;
                        tri2vtx[i_vtx * 3 + 1] = a[1] as usize;
                        tri2vtx[i_vtx * 3 + 2] = a[2] as usize;
                    }
                }
            }
            //  TODO: parse normals and uvs for .ply
            Some((
                shape_entity.material_index.unwrap(),
                shape_entity.area_light_index,
                tri2vtx,
                vtx2xyz,
                vec![],
                vec![],
            ))
        }
        _ => {
            panic!();
        }
    }
}

pub fn spectrum_from_light_entity(area_light_entity: &pbrt4::types::AreaLight) -> Option<[f32; 3]> {
    match area_light_entity {
        pbrt4::types::AreaLight::Diffuse {
            filename: _filename,
            two_sided: _two_sided,
            spectrum,
            scale: _scale,
        } => match spectrum.unwrap().to_owned() {
            pbrt4::param::Spectrum::Rgb(rgb) => Some(rgb),
            _ => {
                panic!();
            }
        },
    }
}

pub fn get_f32_array3_from_params(
    key: &str,
    dict_mp: &std::collections::HashMap<String, (pbrt4::param::ParamType, String, String)>,
) -> Option<[f32; 3]> {
    let mp = dict_mp.get(key)?;
    assert_eq!(mp.1, key.to_string());
    let res: [f32; 3] =
        mp.2.split_whitespace()
            .map(|v| v.parse::<f32>().unwrap())
            .collect::<Vec<_>>()
            .try_into()
            .unwrap();
    Some(res)
}

fn get_bool_from_params(
    key: &str,
    dict_mp: &std::collections::HashMap<String, (pbrt4::param::ParamType, String, String)>,
) -> Option<bool> {
    let mp = dict_mp.get(key)?;
    assert_eq!(mp.1, key.to_string());
    let res: bool = mp.2.parse().ok()?;
    Some(res)
}

pub fn get_f32_from_params(
    key: &str,
    dict_mp: &std::collections::HashMap<String, (pbrt4::param::ParamType, String, String)>,
) -> anyhow::Result<f32> {
    let mp = match dict_mp.get(key) {
        Some(mp) => mp,
        None => {
            return Err(anyhow::Error::msg("hoge"));
        }
    };
    Ok(mp.2.parse::<f32>()?)
}

fn get_texture_index_from_material_param(
    key: &str,
    dict_mp: &std::collections::HashMap<String, (pbrt4::param::ParamType, String, String)>,
    textures: &[pbrt4::types::Texture],
) -> Option<usize> {
    let mp = dict_mp.get(key)?;
    if mp.0 != pbrt4::param::ParamType::Texture {
        return None;
    }

    // can be optimized using hash
    for (i, tex) in textures.iter().enumerate() {
        let name = &mp.2[1..mp.2.len() - 1];
        if tex.name == name {
            return Some(i);
        }
    }

    None
}

pub fn parse_material(scene: &pbrt4::Scene) -> Vec<crate::material::Material> {
    let mut materials = Vec::<crate::material::Material>::with_capacity(scene.materials.len());
    for mat in scene.materials.iter() {
        match mat.attributes.as_str() {
            "diffuse" => {
                let diff = crate::material::DiffuseMaterial {
                    reflectance: mat.reflectance.get_rgb(),
                    reflectance_texture: get_texture_index_from_material_param(
                        "reflectance",
                        &mat.params,
                        &scene.textures,
                    )
                    .unwrap_or(usize::MAX),
                };
                materials.push(crate::material::Material::Diff(diff));
            }
            "conductor" => {
                let uroughness = get_f32_from_params("uroughness", &mat.params).unwrap();
                let vroughness = get_f32_from_params("vroughness", &mat.params).unwrap();
                let reflectance = get_f32_array3_from_params("reflectance", &mat.params)
                    .unwrap_or([1.0, 1.0, 1.0]);
                let k = get_f32_array3_from_params("k", &mat.params).unwrap();
                let eta = get_f32_array3_from_params("eta", &mat.params).unwrap();
                let mat = crate::material::ConductorMaterial {
                    uroughness,
                    vroughness,
                    reflectance,
                    k,
                    eta,
                };
                materials.push(crate::material::Material::Cond(mat))
            }
            "coateddiffuse" => {
                let uroughness = get_f32_from_params("uroughness", &mat.params).unwrap();
                let vroughness = get_f32_from_params("vroughness", &mat.params).unwrap();
                let reflectance = get_f32_array3_from_params("reflectance", &mat.params)
                    .unwrap_or([1.0, 1.0, 1.0]);
                let remaproughness = get_bool_from_params("remaproughness", &mat.params).unwrap();

                let coadiff = crate::material::CoatedDiffuse {
                    uroughness,
                    vroughness,
                    reflectance,
                    remaproughness,
                };
                materials.push(crate::material::Material::CoaDiff(coadiff))
            }
            _ => {
                dbg!(&mat.attributes);
                panic!("Material paser not support");
            }
        }
    }
    materials
}

pub fn parse_area_light(scene: &pbrt4::Scene) -> Vec<AreaLight> {
    let mut area_lights = Vec::<AreaLight>::new();
    for area_light in &scene.area_lights {
        match area_light {
            pbrt4::types::AreaLight::Diffuse {
                filename: _filename,
                two_sided,
                spectrum,
                scale: _scale,
            } => {
                let spectrum_rgb = if let Some(spectrum) = spectrum {
                    match spectrum {
                        pbrt4::param::Spectrum::Rgb(rgb) => Some(rgb),
                        pbrt4::param::Spectrum::Blackbody(_i) => {
                            todo!()
                        }
                    }
                } else {
                    None
                };
                let al = AreaLight {
                    spectrum_rgb: spectrum_rgb.copied(),
                    two_sided: *two_sided,
                };
                area_lights.push(al);
            }
        }
    }
    area_lights
}

pub fn parse_shapes(scene: &pbrt4::Scene) -> Vec<ShapeEntity> {
    let mut shape_entities = Vec::<ShapeEntity>::new();
    for shape_entity in scene.shapes.iter() {
        let shape = match &shape_entity.params {
            pbrt4::types::Shape::TriangleMesh {
                indices,
                positions,
                normals,
                ..
            } => {
                let tri2vtx: Vec<usize> = indices.iter().map(|&v| v as usize).collect();
                let vtx2xyz = positions.clone();
                let tri2cumsumarea = if shape_entity.area_light_index.is_some() {
                    let tri2cumsumarea =
                        del_msh_cpu::trimesh::tri2cumsumarea(&tri2vtx, &vtx2xyz, 3);
                    Some(tri2cumsumarea)
                } else {
                    None
                };
                crate::shape::ShapeType::TriangleMesh {
                    tri2vtx,
                    vtx2xyz,
                    vtx2nrm: normals.clone(),
                    tri2cumsumarea,
                }
            }
            pbrt4::types::Shape::Sphere { radius, .. } => {
                crate::shape::ShapeType::Sphere { radius: *radius }
            }
            _ => {
                dbg!(&shape_entity.params);
                panic!("Parse unsupported shape")
            }
        };

        let transform_objlcl2world = shape_entity.transform.as_ref().to_owned();
        let transform_world2objlcl =
            del_geo_core::mat4_col_major::try_inverse(&transform_objlcl2world).unwrap();
        shape_entities.push(ShapeEntity {
            shape,
            transform_objlcl2world,
            transform_world2objlcl,
            material_index: shape_entity.material_index,
            area_light_index: shape_entity.area_light_index,
        });
    }
    shape_entities
}

pub fn parse_texture(scene: &pbrt4::Scene) -> Vec<crate::textures::Texture> {
    let mut textures = Vec::<crate::textures::Texture>::with_capacity(scene.textures.len());
    for tex in scene.textures.iter() {
        match tex.class.as_str() {
            "checkerboard" => {
                let tex1 = get_f32_array3_from_params("tex1", &tex.params).unwrap();
                let tex2 = get_f32_array3_from_params("tex2", &tex.params).unwrap();
                let uscale = get_f32_from_params("uscale", &tex.params).unwrap();
                let vscale = get_f32_from_params("vscale", &tex.params).unwrap();

                let checker = crate::textures::CheckerBoardTexture {
                    uscale,
                    vscale,
                    tex1,
                    tex2,
                };
                textures.push(crate::textures::Texture::Checkerboard(checker));
            }
            _ => {
                dbg!(&tex.class);
                panic!("Texture paser not support");
            }
        }
    }
    textures
}
