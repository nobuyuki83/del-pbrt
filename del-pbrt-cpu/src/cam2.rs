pub fn transform_world2pix_ortho_preserve_asp(
    image_size: &(usize, usize),
    aabb_world: &[f32; 4],
) -> [f32; 9] {
    let width_img = image_size.0;
    let height_img = image_size.1;
    let asp_img = width_img as f32 / height_img as f32;
    let width_world = aabb_world[2] - aabb_world[0];
    let height_world = aabb_world[3] - aabb_world[1];
    let cntr_world = [
        (aabb_world[0] + aabb_world[2]) * 0.5,
        (aabb_world[1] + aabb_world[3]) * 0.5,
    ];
    let aabb_world1 = if (width_world / height_world) > asp_img {
        [
            aabb_world[0],
            cntr_world[1] - width_world / asp_img * 0.5,
            aabb_world[2],
            cntr_world[1] + width_world / asp_img * 0.5,
        ]
    } else {
        [
            cntr_world[0] - height_world * asp_img * 0.5,
            aabb_world[1],
            cntr_world[0] + height_world * asp_img * 0.5,
            aabb_world[3],
        ]
    };
    let p_tl = [aabb_world1[0], aabb_world1[3]];
    let p_br = [aabb_world1[2], aabb_world1[1]];
    let a = width_img as f32 / (p_br[0] - p_tl[0]);
    let c = -a * p_tl[0];
    let b = height_img as f32 / (p_br[1] - p_tl[1]);
    let d = -b * p_tl[1];
    [a, 0., 0., 0., b, 0., c, d, 1.]
}
