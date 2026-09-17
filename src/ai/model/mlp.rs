use burn::module::Module;
use burn::{
    nn::{Linear, LinearConfig, Relu},
    prelude::*,
};
use serde::{Deserialize, Serialize};

use crate::ai::encoder::INPUT_SIZE;
use crate::ai::model::{ModelOutput, POLICY_SIZE};

const DEFAULT_HIDDEN_SIZE: usize = 128;

#[derive(Debug, Serialize, Deserialize, Clone, PartialEq)]
pub struct MLPModelConfig {
    pub hidden_size: usize,
}

impl Default for MLPModelConfig {
    fn default() -> Self {
        Self {
            hidden_size: DEFAULT_HIDDEN_SIZE,
        }
    }
}

#[derive(Module, Debug)]
pub struct MLPModel<B: Backend> {
    fc1: Linear<B>,
    fc2: Linear<B>,
    policy: Linear<B>,
    value: Linear<B>,
    config: MLPModelConfig,
}

impl<B: Backend> MLPModel<B> {
    pub fn new(config: MLPModelConfig, device: &B::Device) -> Self {
        Self {
            fc1: LinearConfig::new(INPUT_SIZE, config.hidden_size).init(device),
            fc2: LinearConfig::new(config.hidden_size, config.hidden_size).init(device),
            policy: LinearConfig::new(config.hidden_size, POLICY_SIZE).init(device),
            value: LinearConfig::new(config.hidden_size, 1).init(device),
            config,
        }
    }

    pub fn config(&self) -> MLPModelConfig {
        self.config.clone()
    }

    pub fn forward(&self, input: Tensor<B, 2>) -> ModelOutput<B> {
        let x = self.fc1.forward(input);
        let x = Relu::new().forward(x);

        let x = self.fc2.forward(x);
        let x = Relu::new().forward(x);

        let policy = self.policy.forward(x.clone());
        let value = self.value.forward(x).tanh();

        ModelOutput { policy, value }
    }
}
