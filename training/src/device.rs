use clap::ValueEnum;
use std::fmt;
#[derive(Debug, Clone, ValueEnum, Copy)]
pub enum DeviceKind {
    Cpu,
    Cuda,
}

impl fmt::Display for DeviceKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let s = match self {
            DeviceKind::Cpu => "cpu",
            DeviceKind::Cuda => "cuda",
        };
        f.write_str(s)
    }
}
