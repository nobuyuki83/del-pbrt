use crate::sampling;
use nalgebra::{Vector2, Vector3};
use std::ops::{BitAnd, BitOr, Mul};

type Real = f32;
type Vec3f = Vector3<Real>;
type Vec2f = Vector2<Real>;

const N_SPECTRUM_SAMPLES: usize = 4;
const INV_PI: Real = 1. / std::f32::consts::PI;

#[derive(Debug, Clone, Copy)]
pub struct BxDFFlags(u32);

impl BxDFFlags {
    pub const UNSET: Self = Self(0);
    pub const REFLECTION: Self = Self(1);
    pub const TRANSMISSION: Self = Self(1 << 2);
    pub const DIFFUSE: Self = Self(1 << 3);
    pub const GLOSSY: Self = Self(1 << 4);
    pub const SPECULAR: Self = Self(1 << 5);
}

#[derive(Debug, Clone, Copy)]
pub struct BxDFReflTransFlags {
    _flg: u32,
}

impl BxDFReflTransFlags {
    pub const UNSET: Self = Self { _flg: 0 };
    pub const REFLECTION: Self = Self { _flg: 1 };
    pub const TRANSMISSION: Self = Self { _flg: 1 << 2 };
}

impl BitAnd for BxDFFlags {
    type Output = Self;
    fn bitand(self, rhs: Self) -> Self::Output {
        BxDFFlags(self.0 & rhs.0)
    }
}

impl BitOr for BxDFFlags {
    type Output = Self;
    fn bitor(self, rhs: Self) -> Self::Output {
        BxDFFlags(self.0 | rhs.0)
    }
}

#[derive(Debug, Clone, Copy)]
pub struct SampledSpectrum {
    pub arr: [Real; N_SPECTRUM_SAMPLES],
}

impl Mul<Real> for SampledSpectrum {
    type Output = Self;
    fn mul(self, rhs: Real) -> Self::Output {
        let res = self.arr.map(|x| x * rhs);
        SampledSpectrum { arr: res }
    }
}

pub struct BSDFSample {
    pub flag: BxDFFlags,
    pub wi: Vec3f,
    pub pdf: Real,
}

impl BSDFSample {
    pub fn has_flag(&self, flag: BxDFFlags) -> bool {
        self.flag.0 & flag.0 != 0
    }
}

#[derive(Debug, Clone, Copy)]
pub enum TransportMode {
    Radiance,
    Importance,
}

pub struct DiffuseBxDF {
    r: SampledSpectrum,
}

impl DiffuseBxDF {
    pub fn new(r: SampledSpectrum) -> Self {
        DiffuseBxDF { r }
    }

    fn same_hemisphere(wo: Vec3f, wi: Vec3f) -> bool {
        wo.z * wi.z > 0.
    }

    pub fn f(&self, wo: Vec3f, wi: Vec3f) -> SampledSpectrum {
        if !Self::same_hemisphere(wo, wi) {
            return SampledSpectrum {
                arr: [0.; N_SPECTRUM_SAMPLES],
            };
        }
        self.r * INV_PI
    }

    pub fn pdf(&self, wo: Vec3f, wi: Vec3f) -> Real {
        if !Self::same_hemisphere(wo, wi) {
            return 0.;
        }
        sampling::pdf_hemisphere_cos(wi.z.abs())
    }

    /// return incident light of given outgoing light wo
    pub fn sample_f(&self, wo: Vec3f, u: Vec2f) -> Option<BSDFSample> {
        let mut wi = sampling::hemisphere_zup_cos_weighted(&[u.x, u.y]);
        if wo.z < 0. {
            wi[2] *= -1.;
        }
        let pdf = sampling::pdf_hemisphere_cos(wi[2].abs());

        Some(BSDFSample {
            flag: BxDFFlags::DIFFUSE | BxDFFlags::REFLECTION,
            wi: Vec3f::new(wi[0], wi[1], wi[2]),
            pdf,
        })
    }

    pub fn get_type(&self) -> BxDFFlags {
        BxDFFlags::DIFFUSE | BxDFFlags::REFLECTION
    }
}
