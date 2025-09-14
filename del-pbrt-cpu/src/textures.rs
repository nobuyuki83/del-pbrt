#[derive(Debug)]
pub enum Texture {
    Checkerboard(CheckerBoardTexture),
}

#[derive(Debug)]
pub struct CheckerBoardTexture {
    pub uscale: f32,
    pub vscale: f32,
    pub tex1: [f32; 3],
    pub tex2: [f32; 3],
}

pub fn sample_checkerboard(
    uv: &[f32; 2],
    uscale: f32,
    vscale: f32,
    tex1: &[f32; 3],
    tex2: &[f32; 3],
) -> [f32; 3] {
    let u = uv[0];
    let v = uv[1];

    let d = (u * uscale).floor() + (v * vscale).floor();

    if d as usize % 2 == 0 {
        [tex1[0], tex1[1], tex1[2]]
    } else {
        [tex2[0], tex2[1], tex2[2]]
    }
}
