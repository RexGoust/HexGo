use std::fmt;

use clap::ValueEnum;

#[derive(Debug, Clone, ValueEnum, Copy)]
pub enum ModelType {
    Mlp,
    Gnn,
}

impl ModelType {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Mlp => "mlp",
            Self::Gnn => "gnn",
        }
    }
}

impl fmt::Display for ModelType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}
