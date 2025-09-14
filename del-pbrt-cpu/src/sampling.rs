use num_traits::AsPrimitive;

const INV_PI: f64 = 1. / std::f64::consts::PI;
// largest float number less than 1
#[cfg(target_pointer_width = "64")]
pub const ONE_MINUS_EPSILON: f64 = 1.0 - f64::EPSILON;
#[cfg(target_pointer_width = "32")]
pub const ONE_MINUS_EPSILON: f32 = 1.0 - f32::EPSILON;

pub fn hemisphere_zup_cos_weighted<T>(r2: &[T; 2]) -> [T; 3]
where
    T: num_traits::Float + 'static + Copy,
    f64: AsPrimitive<T>,
{
    let z = (r2[0]).sqrt();
    let r = (T::one() - z * z).sqrt();
    let phi: T = 2_f64.as_() * std::f64::consts::PI.as_() * r2[1];
    [r * phi.cos(), r * phi.sin(), z]
}

pub fn hemisphere_cos_weighted<T>(n: &[T; 3], r2: &[T; 2]) -> [T; 3]
where
    T: num_traits::Float + Copy + 'static,
    f64: AsPrimitive<T>,
{
    use del_geo_core::vec3::Vec3;
    let h0 = hemisphere_zup_cos_weighted(r2);
    let t = if num_traits::Float::abs(n[0]) > 0.1.as_() {
        [T::zero(), T::one(), T::zero()]
    } else {
        [T::one(), T::zero(), T::zero()]
    };
    let u = t.cross(n).normalize(); // orthogonal to w
    let v = n.cross(&u); // orthogonal to w and u
    del_geo_core::vec3::add_three(&u.scale(h0[0]), &v.scale(h0[1]), &n.scale(h0[2])).normalize()
}

pub fn tent<T>(r0: T) -> T
where
    T: num_traits::Float + 'static + Copy,
    f64: AsPrimitive<T>,
{
    let r1: T = 2_f64.as_() * r0;
    if r1 < T::one() {
        r1.sqrt() - T::one()
    } else {
        let tmp: T = 2_f64.as_() - r1;
        T::one() - tmp.sqrt()
    } // tent filter (-1 .. +1 )
}

pub fn pdf_hemisphere_cos<T>(cos_theta: T) -> T
where
    T: num_traits::Float + 'static + Copy,
    f64: AsPrimitive<T>,
{
    cos_theta * T::from(INV_PI).unwrap()
}

pub fn radical_inverse<Real>(mut a: usize, base: usize) -> Real
where
    Real: num_traits::Float,
{
    // base must be prime numbers
    let inv_base = (Real::one()) / (Real::from(base).unwrap());
    let mut inv_base_m = Real::one();
    //reversed digits:
    let mut rev_digits: usize = 0;
    while a != 0 {
        let next: usize = a / base;
        // least significant digit
        let digit: usize = a - next * base;
        rev_digits = rev_digits * base + digit;
        inv_base_m = inv_base_m * inv_base;
        a = next;
    }
    // can be expressed as (d_1*b^(m-1) + d_2*b^(m-2) ... + d_m*b^0 )/b^(m)
    let inv = Real::from(rev_digits).unwrap() * inv_base_m;
    Real::min(inv, Real::from(ONE_MINUS_EPSILON).unwrap())
}
