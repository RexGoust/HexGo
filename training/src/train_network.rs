use burn::Tensor;
use burn::tensor::TensorData;
use burn::tensor::activation::softmax;
use burn::tensor::backend::Backend;
use hex_go::ai::encoder::{MLP_INPUT_SIZE, VERTEX_COUNT};

use hex_go::ai::model::HexGoModel;
use hex_go::ai::model::gnn::FEATURE_DIM;
use hex_go::game::action::{ACTION_SIZE, Action, PASS_INDEX};
use hex_go::game::board::VertexId;

pub fn forward_batch<B: Backend>(
    model: &HexGoModel<B>,
    device: &B::Device,
    adj: &Tensor<B, 2>,
    inputs: &[&Vec<f32>],
) -> (Vec<Vec<(Action, f32)>>, Vec<f32>) {
    let batch_size = inputs.len();

    let (policy_tensor, value_tensor) = match model {
        HexGoModel::Mlp(m) => {
            let mut flat = Vec::with_capacity(batch_size * MLP_INPUT_SIZE);
            for input in inputs {
                flat.extend_from_slice(input);
            }

            let x = Tensor::<B, 1>::from_data(
                TensorData::new(flat, [batch_size * MLP_INPUT_SIZE]),
                device,
            )
            .reshape([batch_size, MLP_INPUT_SIZE]);

            let out = m.forward(x);
            (softmax(out.policy, 1), out.value)
        }

        HexGoModel::Gnn(m) => {
            let state_size = FEATURE_DIM * VERTEX_COUNT;
            let mut flat = Vec::with_capacity(batch_size * state_size);
            for input in inputs {
                flat.extend_from_slice(input);
            }

            let x =
                Tensor::<B, 1>::from_data(TensorData::new(flat, [batch_size * state_size]), device)
                    .reshape([batch_size, VERTEX_COUNT, FEATURE_DIM]);

            let out = m.forward(x, adj.clone());
            (softmax(out.policy, 1), out.value)
        }
    };

    let policies_flat: Vec<f32> = policy_tensor.into_data().to_vec().unwrap();
    let values: Vec<f32> = value_tensor.into_data().to_vec().unwrap();

    let mut policies = Vec::with_capacity(batch_size);

    for i in 0..batch_size {
        let start = i * ACTION_SIZE;
        let p_slice = &policies_flat[start..start + ACTION_SIZE];

        let mut policy = Vec::with_capacity(ACTION_SIZE);
        for (idx, &p) in p_slice.iter().enumerate() {
            let action = if idx == PASS_INDEX {
                Action::Pass
            } else {
                Action::Move(VertexId::new(idx))
            };
            policy.push((action, p));
        }
        policies.push(policy);
    }

    (policies, values)
}
