use num_traits::AsPrimitive;

pub struct Camera3<T> {
    pub w: usize,
    pub h: usize,
    o: [T; 3],
    d: [T; 3],
    cx: [T; 3],
    cy: [T; 3],
}

impl<T> Camera3<T>
where
    T: num_traits::Float + 'static + Copy,
    f64: AsPrimitive<T>,
    usize: AsPrimitive<T>,
{
    pub fn new(w: usize, h: usize, o: [T; 3], d: [T; 3]) -> Self {
        use del_geo_core::vec3::Vec3;
        let d = d.normalize();
        let cx = [w.as_() * 0.5135.as_() / h.as_(), T::zero(), T::zero()];
        let cy = cx.cross(&d).normalize().scale(0.5135.as_());
        Camera3 { w, h, o, d, cx, cy }
    }

    pub fn ray(&self, x0: T, y0: T) -> ([T; 3], [T; 3]) {
        use del_geo_core::vec3::Vec3;
        let d = del_geo_core::vec3::add_three(
            &self.cx.scale(x0 / self.w.as_() - 0.5.as_()),
            &self.cy.scale(y0 / self.h.as_() - 0.5.as_()),
            &self.d,
        );
        (self.o.add(&d.scale(140.as_())), d.normalize())
    }
}

/// the ray start from the front plane and ends on the back plane
pub fn ray3_homogeneous(
    pix_coord: (usize, usize),
    image_size: (usize, usize),
    transform_ndc_to_world: &[f32; 16],
) -> ([f32; 3], [f32; 3]) {
    let x0 = 2. * (pix_coord.0 as f32 + 0.5f32) / (image_size.0 as f32) - 1.;
    let y0 = 1. - 2. * (pix_coord.1 as f32 + 0.5f32) / (image_size.1 as f32);
    let p0 =
        del_geo_core::mat4_col_major::transform_homogeneous(transform_ndc_to_world, &[x0, y0, 1.])
            .unwrap();
    let p1 =
        del_geo_core::mat4_col_major::transform_homogeneous(transform_ndc_to_world, &[x0, y0, -1.])
            .unwrap();
    let ray_org = p0;
    let ray_dir = del_geo_core::vec3::sub(&p1, &p0);
    (ray_org, ray_dir)
}
