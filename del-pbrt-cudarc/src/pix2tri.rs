use cudarc::driver::{CudaDevice, CudaSlice, DeviceSlice};
use del_cudarc::cudarc;

#[allow(clippy::too_many_arguments)]
pub fn fwd(
    dev: &std::sync::Arc<CudaDevice>,
    img_shape: (usize, usize),
    pix2tri: &mut CudaSlice<u32>,
    tri2vtx: &CudaSlice<u32>,
    vtx2xyz: &CudaSlice<f32>,
    bvhnodes: &CudaSlice<u32>,
    bvhnode2aabb: &CudaSlice<f32>,
    transform_ndc2world: &CudaSlice<f32>,
) -> std::result::Result<(), cudarc::driver::DriverError> {
    // let cfg = cudarc::driver::LaunchConfig::for_num_elems((img_shape.0 * img_shape.1) as u32);
    let cfg = {
        let n = (img_shape.0 * img_shape.1) as u32;
        const NUM_THREADS: u32 = 32;
        let num_blocks = (n + NUM_THREADS - 1) / NUM_THREADS;
        cudarc::driver::LaunchConfig {
            grid_dim: (num_blocks, 1, 1),
            block_dim: (NUM_THREADS, 1, 1),
            shared_mem_bytes: 0,
        }
    };
    let num_tri = tri2vtx.len() / 3;
    let param = (
        pix2tri,
        num_tri as u32,
        tri2vtx,
        vtx2xyz,
        img_shape.0 as u32,
        img_shape.1 as u32,
        transform_ndc2world,
        bvhnodes,
        bvhnode2aabb,
    );
    //unsafe { self.pix_to_tri.clone().launch(cfg,param) }.unwrap();
    let func =
        del_cudarc::get_or_load_func(dev, "fwd_pix2tri", del_raycast_cudarc_kernel::PIX2TRI)?;
    use cudarc::driver::LaunchAsync;
    unsafe { func.launch(cfg, param) }?;
    Ok(())
}
