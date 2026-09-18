use burn::{
    Tensor,
    module::Module,
    nn::{Dropout, DropoutConfig, LayerNorm, LayerNormConfig, Linear, LinearConfig, Relu},
    tensor::backend::Backend,
};
use ricci::GCNConv;
use serde::{Deserialize, Serialize};

use crate::ai::{encoder::VERTEX_COUNT, model::ModelOutput};

const DEFAULT_HIDDEN_DIM: usize = 128;
pub const FEATURE_DIM: usize = 8;

#[derive(Debug, Serialize, Deserialize, Clone, PartialEq)]
pub struct GnnModelConfig {
    pub feature_dim: usize,
    pub hidden_dim: usize,
    pub num_vertices: usize,
}

impl Default for GnnModelConfig {
    fn default() -> Self {
        Self {
            feature_dim: FEATURE_DIM,
            hidden_dim: DEFAULT_HIDDEN_DIM,
            num_vertices: VERTEX_COUNT,
        }
    }
}

#[derive(Module, Debug)]
pub struct GnnModel<B: Backend> {
    conv1: GCNConv<B>,
    conv2: GCNConv<B>,

    policy_node: Linear<B>,
    policy_pass: Linear<B>,
    value: Linear<B>,

    dropout: Dropout,
    norm1: LayerNorm<B>,
    norm2: LayerNorm<B>,

    config: GnnModelConfig,
}

impl<B: Backend> GnnModel<B> {
    pub fn new(config: GnnModelConfig, device: &B::Device) -> Self {
        Self {
            config: config.clone(),
            conv1: GCNConv::init(config.feature_dim, config.hidden_dim, device),
            conv2: GCNConv::init(config.hidden_dim, config.hidden_dim, device),
            policy_node: LinearConfig::new(config.hidden_dim, 1).init(device),
            policy_pass: LinearConfig::new(config.hidden_dim, 1).init(device),
            value: LinearConfig::new(config.hidden_dim, 1).init(device),
            dropout: DropoutConfig::new(0.2).init(),
            norm1: LayerNormConfig::new(config.hidden_dim).init(device),
            norm2: LayerNormConfig::new(config.hidden_dim).init(device),
        }
    }

    pub fn forward(&self, x: Tensor<B, 2>, adj: Tensor<B, 2>) -> ModelOutput<B> {
        // GCN + Norm + ReLU + Dropout
        let h = self.conv1.forward(x, adj.clone());
        let h = self.norm1.forward(h);
        let h = Relu::new().forward(h);
        let h = self.dropout.forward(h);

        // GCN + Residual + Norm + ReLU + Dropout
        let h_in = h.clone();
        let h = self.conv2.forward(h, adj);
        let h = h + h_in;
        let h = self.norm2.forward(h);
        let h = Relu::new().forward(h);
        let h = self.dropout.forward(h);

        let node_logits = self.policy_node.forward(h.clone()); // [N, 1]
        let node_logits = node_logits.squeeze::<1>(); // [N] = [88]
        let global = h.mean_dim(0); // [1, hidden_dim]

        let pass_logit = self.policy_pass.forward(global.clone()); // [1, 1]
        let pass_logit = pass_logit.squeeze::<1>();

        let policy = Tensor::cat(vec![node_logits, pass_logit], 0); // [89]
        let policy = policy.reshape([1, 89]);

        let value = self.value.forward(global).tanh(); // [1, 1]

        ModelOutput { policy, value }
    }

    pub fn config(&self) -> GnnModelConfig {
        self.config.clone()
    }
}
