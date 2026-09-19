use clap::ValueEnum;
#[derive(Debug, Clone, ValueEnum, Copy)]
pub enum DeviceKind {
    Cpu,
    Cuda,
}
