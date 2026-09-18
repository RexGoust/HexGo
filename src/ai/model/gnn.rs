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
    conv3: GCNConv<B>,

    policy_node: Linear<B>,
    policy_pass: Linear<B>,
    value: Linear<B>,

    dropout: Dropout,
    norm1: LayerNorm<B>,
    norm2: LayerNorm<B>,
    norm3: LayerNorm<B>,
    config: GnnModelConfig,
}

impl<B: Backend> GnnModel<B> {
    pub fn new(config: GnnModelConfig, device: &B::Device) -> Self {
        Self {
            config: config.clone(),
            conv1: GCNConv::init(config.feature_dim, config.hidden_dim, device),
            conv2: GCNConv::init(config.hidden_dim, config.hidden_dim, device),
            conv3: GCNConv::init(config.hidden_dim, config.hidden_dim, device),
            policy_node: LinearConfig::new(config.hidden_dim, 1).init(device),
            policy_pass: LinearConfig::new(config.hidden_dim, 1).init(device),
            value: LinearConfig::new(config.hidden_dim, 1).init(device),
            dropout: DropoutConfig::new(0.2).init(),
            norm1: LayerNormConfig::new(config.hidden_dim).init(device),
            norm2: LayerNormConfig::new(config.hidden_dim).init(device),
            norm3: LayerNormConfig::new(config.hidden_dim).init(device),
        }
    }

    fn conv_forward(&self, conv: &GCNConv<B>, x: Tensor<B, 3>, adj: Tensor<B, 2>) -> Tensor<B, 3> {
        let [batch_size, num_nodes, _] = x.dims();

        let projected = conv.linear().forward(x);
        let [_, _, d_out] = projected.dims();

        let (projected_features, bias_val) = match &conv.linear().bias {
            Some(bias) => {
                let b = bias.val().reshape([1, 1, d_out]);
                (projected - b.clone(), Some(b))
            }
            None => (projected, None),
        };

        let p = projected_features
            .swap_dims(0, 1)
            .reshape([num_nodes, batch_size * d_out]);
        let aggregated = adj
            .matmul(p)
            .reshape([num_nodes, batch_size, d_out])
            .swap_dims(0, 1);

        match bias_val {
            Some(b) => aggregated + b,
            None => aggregated,
        }
    }

    pub fn forward(&self, x: Tensor<B, 3>, adj: Tensor<B, 2>) -> ModelOutput<B> {
        let [batch_size, num_nodes, _] = x.dims();

        // GCN + Norm + ReLU + Dropout
        let h = self.conv_forward(&self.conv1, x, adj.clone());
        let h = self.norm1.forward(h);
        let h = Relu::new().forward(h);
        let h = self.dropout.forward(h);

        // GCN + Residual + Norm + ReLU + Dropout
        let h_in = h.clone();
        let h = self.conv_forward(&self.conv2, h, adj.clone());
        let h = h + h_in;
        let h = self.norm2.forward(h);
        let h = Relu::new().forward(h);
        let h = self.dropout.forward(h);

        // GCN + Residual + Norm + ReLU + Dropout
        let h_in = h.clone();
        let h = self.conv_forward(&self.conv3, h, adj);
        let h = h + h_in;
        let h = self.norm3.forward(h);
        let h = Relu::new().forward(h);
        let h = self.dropout.forward(h);

        let node_logits = self
            .policy_node
            .forward(h.clone())
            .reshape([batch_size, num_nodes]); // [1, N]

        let global = h.mean_dim(1).reshape([batch_size, self.config.hidden_dim]);

        let pass_logit = self.policy_pass.forward(global.clone()); // [1, 1]

        let policy = Tensor::cat(vec![node_logits, pass_logit], 1); // [1, N]

        let value = self.value.forward(global).tanh(); // [1, 1]

        ModelOutput { policy, value }
    }

    pub fn config(&self) -> GnnModelConfig {
        self.config.clone()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ai::backend::Backend;

    #[test]
    fn gnn_forward_produces_expected_output_shapes() {
        let device = Default::default();
        let config = GnnModelConfig::default();
        let model = GnnModel::<Backend>::new(config, &device);

        let x = Tensor::<Backend, 2>::zeros([88, FEATURE_DIM], &device).unsqueeze::<3>();
        let adj = Tensor::<Backend, 2>::zeros([88, 88], &device);

        let output = model.forward(x, adj);

        assert_eq!(output.policy.dims(), [1, 89]);
        assert_eq!(output.value.dims(), [1, 1]);
    }

    #[test]
    fn gnn_forward_handles_arbitrary_vertex_count() {
        let device = Default::default();
        let config = GnnModelConfig {
            feature_dim: FEATURE_DIM,
            hidden_dim: 32,
            num_vertices: 4,
        };
        let model = GnnModel::<Backend>::new(config, &device);

        let x = Tensor::<Backend, 2>::zeros([4, FEATURE_DIM], &device).unsqueeze::<3>();
        let adj = Tensor::<Backend, 2>::zeros([4, 4], &device);

        let output = model.forward(x, adj);

        assert_eq!(output.policy.dims(), [1, 5]);
        assert_eq!(output.value.dims(), [1, 1]);
    }
}
