type Real = f64;

struct Ray {
    o: [Real; 3],
    d: [Real; 3],
}

impl Ray {
    fn new(o: [Real; 3], d: [Real; 3]) -> Self {
        Self { o, d }
    }
}

enum ReflT {
    DIFF,
    SPEC,
} // material types, used in radiance()

struct Sphere {
    rad: Real,
    // radius
    p: [Real; 3],
    // position
    e: [Real; 3],
    // emission
    c: [Real; 3],
    // color
    refl: ReflT, // reflection type (DIFFuse, SPECular, REFRactive)
}

impl Sphere {
    fn new(rad_: Real, p_: [Real; 3], e_: [Real; 3], c_: [Real; 3], refl_: ReflT) -> Self {
        Self {
            rad: rad_,
            p: p_,
            e: e_,
            c: c_,
            refl: refl_,
        }
    }

    fn intersect(&self, r: &Ray) -> Option<Real> {
        let Some(t) = del_geo_core::sphere::intersection_ray(self.rad, &self.p, &r.o, &r.d) else {
            return None;
        };
        Some(t)
    }
}

fn intersect(r: &Ray, spheres: &[Sphere]) -> Option<(Real, usize)> {
    const INF: Real = 1e20;
    let mut t_min = INF;
    let mut id = usize::MAX;
    for (isphere, sphere) in spheres.iter().enumerate() {
        let Some(t) = sphere.intersect(r) else {
            continue;
        };
        if t < t_min {
            t_min = t;
            id = isphere;
        }
    }
    if t_min < INF {
        return Some((t_min, id));
    }
    None
}

fn radiance(r: &Ray, depth: i64, spheres: &[Sphere], rng: &mut rand::rngs::ThreadRng) -> [Real; 3] {
    use del_geo_core::vec3::Vec3;
    use rand::Rng;
    let Some((t, id)) = intersect(r, spheres) else {
        return [0., 0., 0.]; // if ray miss, return black
    };
    let obj = &spheres[id]; // the hit object
    let hit_pos = r.o.add(&r.d.scale(t)); // hit pos
    let hit_nrm = hit_pos.sub(&obj.p).normalize();
    let hit_nrm = if hit_nrm.dot(&r.d) < 0. {
        hit_nrm
    } else {
        hit_nrm.scale(-1.)
    };
    let mut f = obj.c;
    let depth = depth + 1;
    if depth > 5 {
        let p = if f[0] > f[1] && f[0] > f[2] {
            f[0]
        } else if f[1] > f[2] {
            f[1]
        } else {
            f[2]
        }; // max refl
        if rng.random::<Real>() < p {
            f = f.scale(1. / p);
        } else {
            return obj.e;
        } //R.R.
    }
    let hit_pos_offset = hit_pos.add(&hit_nrm.scale(1.0e-3));
    let next_dir = match obj.refl {
        ReflT::DIFF => del_raycast_core::sampling::hemisphere_cos_weighted(
            &hit_nrm,
            &[rng.random::<Real>(), rng.random::<Real>()],
        ),
        ReflT::SPEC => r.d.sub(&hit_nrm.scale(2. * hit_nrm.dot(&r.d))),
    };
    obj.e.add(&f.element_wise_mult(&radiance(
        &Ray::new(hit_pos_offset, next_dir),
        depth,
        spheres,
        rng,
    )))
}

fn main() {
    use crate::ReflT::{DIFF, SPEC};
    use del_geo_core::vec3::Vec3;
    use rand::Rng;
    let spheres = [
        Sphere::new(
            1e5,
            [1e5 + 1., 40.8, 81.6],
            [0., 0., 0.],
            [0.75, 0.25, 0.25],
            DIFF,
        ), //Left
        Sphere::new(
            1e5,
            [-1e5 + 99., 40.8, 81.6],
            [0., 0., 0.],
            [0.25, 0.25, 0.75],
            DIFF,
        ), //Rght
        Sphere::new(
            1e5,
            [50., 40.8, 1e5],
            [0., 0., 0.],
            [0.75, 0.75, 0.75],
            DIFF,
        ), //Back
        Sphere::new(
            1e5,
            [50., 40.8, -1e5 + 170.],
            [0., 0., 0.],
            [0., 0., 0.],
            DIFF,
        ), //Frnt
        Sphere::new(
            1e5,
            [50., 1e5, 81.6],
            [0., 0., 0.],
            [0.75, 0.75, 0.75],
            DIFF,
        ), //Botm
        Sphere::new(
            1e5,
            [50., -1e5 + 81.6, 81.6],
            [0., 0., 0.],
            [0.75, 0.75, 0.75],
            DIFF,
        ), //Top
        Sphere::new(
            16.5,
            [27., 16.5, 47.],
            [0., 0., 0.],
            [1., 1., 1.].scale(0.999),
            SPEC,
        ), //Mirr
        Sphere::new(
            600.,
            [50., 681.6 - 0.27, 81.6],
            [12., 12., 12.],
            [0., 0., 0.],
            DIFF,
        ), //Lite
    ];
    let mut rng = rand::rng();
    let samps = 16;
    let cam =
        del_raycast_core::cam3::Camera3::new(1024, 768, [50., 52., 295.6], [0., -0.042612, -1.]);
    let mut img = Vec::<image::Rgb<f32>>::new();
    img.resize(cam.w * cam.h, image::Rgb([0_f32; 3]));
    for y in 0..cam.h {
        // Loop over image rows
        for x in 0..cam.w {
            let mut c = [0., 0., 0.];
            for sy in 0..2 {
                // 2x2 subpixel rows
                for sx in 0..2 {
                    // 2x2 subpixel cols
                    for _s in 0..samps {
                        let dx = del_raycast_core::sampling::tent(rng.random::<Real>());
                        let dy = del_raycast_core::sampling::tent(rng.random::<Real>());
                        let sx = sx as Real;
                        let x0 = (sx as Real + 0.5 + dx) / 2. + x as Real;
                        let y0 = (sy as Real + 0.5 + dy) / 2. + y as Real;
                        let ray = cam.ray(x0, y0);
                        let ray = Ray::new(ray.0, ray.1);
                        c = c.add(&radiance(&ray, 0, &spheres, &mut rng));
                    } // Camera rays are pushed ^^^^^ forward to start in interior
                }
            }
            c = c.scale(0.25 * (1. / samps as Real));
            img[(cam.h - 1 - y) * cam.w + x] = image::Rgb([c[0] as f32, c[1] as f32, c[2] as f32]);
        }
    }
    {
        use image::codecs::hdr::HdrEncoder;
        let file = std::fs::File::create("target/00_smallpt.hdr").unwrap();
        let enc = HdrEncoder::new(file);
        let _ = enc.encode(&img, cam.w, cam.h);
    }
}
