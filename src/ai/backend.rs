#[cfg(feature = "cuda")]
pub type Backend = burn::backend::Cuda<f32, i32>;

#[cfg(feature = "cuda")]
pub type Device = burn::backend::cuda::CudaDevice;

#[cfg(not(any(feature = "cuda")))]
pub type Backend = burn::backend::Flex;

#[cfg(not(any(feature = "cuda")))]
pub type Device = burn::backend::flex::FlexDevice;

pub fn default_device() -> Device {
    Default::default()
}
