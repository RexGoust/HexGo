use crate::game::action::ACTION_SIZE;
use burn::prelude::*;
use serde::{Deserialize, Serialize};

pub mod mlp;
pub mod store;
pub use mlp::{MLPModel, MLPModelConfig};

const POLICY_SIZE: usize = ACTION_SIZE;

#[derive(Debug, Serialize, Deserialize, Clone, PartialEq)]
#[serde(tag = "type", rename_all = "lowercase")]
pub enum ModelConfig {
    Mlp(MLPModelConfig),
}

impl ModelConfig {
    pub fn parse_with_fallback(json_str: &str) -> Result<Self, serde_json::Error> {
        if let Ok(config) = serde_json::from_str::<ModelConfig>(json_str) {
            return Ok(config);
        }

        let old_mlp = serde_json::from_str::<MLPModelConfig>(json_str)?;
        Ok(ModelConfig::Mlp(old_mlp))
    }
}

pub struct ModelOutput<B: Backend> {
    pub policy: Tensor<B, 2>,
    pub value: Tensor<B, 2>,
}

#[derive(Module, Debug)]
pub enum HexGoModel<B: Backend> {
    Mlp(MLPModel<B>),
}

impl<B: Backend> HexGoModel<B> {
    pub fn new(config: ModelConfig, device: &B::Device) -> Self {
        match config {
            ModelConfig::Mlp(cfg) => Self::Mlp(MLPModel::new(cfg, device)),
        }
    }

    pub fn config(&self) -> ModelConfig {
        match self {
            Self::Mlp(m) => ModelConfig::Mlp(m.config()),
        }
    }

    pub fn forward(&self, input: Tensor<B, 2>) -> ModelOutput<B> {
        match self {
            Self::Mlp(m) => m.forward(input),
        }
    }
}
