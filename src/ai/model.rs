use crate::ai::encoder::INPUT_SIZE;
use crate::game::action::ACTION_SIZE;
use burn::{
    nn::{Linear, LinearConfig, Relu},
    prelude::*,
};

const HIDDEN_SIZE: usize = 128;
const POLICY_SIZE: usize = ACTION_SIZE;

#[derive(Module, Debug)]
pub struct HexGoModel<B: Backend> {
    fc1: Linear<B>,
    fc2: Linear<B>,
    policy: Linear<B>,
    value: Linear<B>,
}

pub struct ModelOutput<B: Backend> {
    pub policy: Tensor<B, 2>,
    pub value: Tensor<B, 2>,
}

impl<B: Backend> HexGoModel<B> {
    pub fn new(device: &B::Device) -> Self {
        Self {
            fc1: LinearConfig::new(INPUT_SIZE, HIDDEN_SIZE).init(device),
            fc2: LinearConfig::new(HIDDEN_SIZE, HIDDEN_SIZE).init(device),
            policy: LinearConfig::new(HIDDEN_SIZE, POLICY_SIZE).init(device),
            value: LinearConfig::new(HIDDEN_SIZE, 1).init(device),
        }
    }

    pub fn forward(&self, input: Tensor<B, 2>) -> ModelOutput<B> {
        let x = self.fc1.forward(input);
        let x = Relu::new().forward(x);

        let x = self.fc2.forward(x);
        let x = Relu::new().forward(x);

        let policy = self.policy.forward(x.clone());
        let value = self.value.forward(x);

        ModelOutput { policy, value }
    }
}
