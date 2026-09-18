#![allow(dead_code)]
use burn::{
    optim::{GradientsParams, Optimizer},
    tensor::{
        ElementConversion,
        backend::{AutodiffBackend, Backend},
    },
};

use crate::{
    dataset::TrainingSample,
    loss::{LossValue, pred_entropy, target_entropy, total_loss},
    tensor::{samples_to_gnn_tensors, samples_to_mlp_tensors},
};
use hex_go::{
    ai::{encoder::adjacency_tensor, model::HexGoModel},
    board_layout::BoardDefinition,
};

pub fn train_on_samples<B: AutodiffBackend>(
    model: HexGoModel<B>,
    optimizer: &mut impl Optimizer<HexGoModel<B>, B>,
    samples: &[TrainingSample],
    device: &B::Device,
    learning_rate: f64,
) -> (HexGoModel<B>, LossValue) {
    let (output, target_policy, target_value) = match &model {
        HexGoModel::Mlp(m) => {
            let (state, policy, value) = samples_to_mlp_tensors::<B>(samples, device);
            (m.forward(state), policy, value)
        }
        HexGoModel::Gnn(m) => {
            let (state, policy, value) = samples_to_gnn_tensors::<B>(samples, device);
            let board_def = BoardDefinition::compact();
            let adj = adjacency_tensor::<B>(board_def.graph(), device);
            (m.forward(state, adj), policy, value)
        }
    };

    let loss = total_loss(
        output.policy.clone(),
        target_policy.clone(),
        output.value,
        target_value,
    );

    let total = loss.total.clone().into_scalar().elem::<f32>();
    let policy = loss.policy.into_scalar().elem::<f32>();
    let value = loss.value.into_scalar().elem::<f32>();

    let grads = loss.total.backward();

    let grads = GradientsParams::from_grads(grads, &model);

    let model = optimizer.step(learning_rate, model, grads);
    //let target_entropy = target_entropy(target_policy);
    //let pred_entropy = pred_entropy(output.policy.detach());

    (
        model,
        LossValue {
            policy,
            value,
            total,
            target_entropy: None,
            pred_entropy: None,
        },
    )
}

pub fn validation_step<B: Backend>(
    model: &HexGoModel<B>,
    samples: &[TrainingSample],
    batch_size: usize,
    device: &B::Device,
) -> LossValue {
    if samples.is_empty() {
        return LossValue {
            policy: 0.0,
            value: 0.0,
            total: 0.0,
            target_entropy: None,
            pred_entropy: None,
        };
    }

    let board_def = BoardDefinition::compact();
    let adj = adjacency_tensor::<B>(board_def.graph(), device);

    let mut total_loss_sum = 0.0;
    let mut policy_loss_sum = 0.0;
    let mut value_loss_sum = 0.0;
    let mut target_entropy_sum = 0.0;
    let mut pred_entropy_sum = 0.0;
    let total_samples = samples.len() as f32;

    for batch in samples.chunks(batch_size.max(1)) {
        let batch_len = batch.len() as f32;

        let (output, target_policy, target_value) = match model {
            HexGoModel::Mlp(m) => {
                let (states, policies, values) = samples_to_mlp_tensors(batch, device);
                (m.forward(states), policies, values)
            }
            HexGoModel::Gnn(m) => {
                let (states, policies, values) = samples_to_gnn_tensors(batch, device);
                (m.forward(states, adj.clone()), policies, values)
            }
        };

        let loss = total_loss(
            output.policy.clone(),
            target_policy.clone(),
            output.value,
            target_value,
        );

        let total = loss.total.into_scalar().elem::<f32>();
        let policy = loss.policy.into_scalar().elem::<f32>();
        let value = loss.value.into_scalar().elem::<f32>();
        let t_entropy = target_entropy(target_policy);
        let p_entropy = pred_entropy(output.policy);

        total_loss_sum += total * batch_len;
        policy_loss_sum += policy * batch_len;
        value_loss_sum += value * batch_len;
        target_entropy_sum += t_entropy * batch_len;
        pred_entropy_sum += p_entropy * batch_len;
    }
    LossValue {
        policy: policy_loss_sum / total_samples,
        value: value_loss_sum / total_samples,
        total: total_loss_sum / total_samples,
        target_entropy: Some(target_entropy_sum / total_samples),
        pred_entropy: Some(pred_entropy_sum / total_samples),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dataset::TrainingSample;
    use burn::{
        backend::{Autodiff, Flex},
        optim::AdamWConfig,
    };
    use hex_go::ai::model::{MlpModel, MlpModelConfig};
    use hex_go::game::action::ACTION_SIZE;

    type Backend = Autodiff<Flex>;

    #[test]
    fn train_step_reduces_loss() {
        let device = Default::default();

        let samples = vec![TrainingSample {
            state: vec![0.0; 264],
            policy: {
                let mut p = vec![0.0; ACTION_SIZE];
                p[0] = 1.0;
                p
            },
            value: 1.0,
        }];

        let mut model =
            HexGoModel::Mlp(MlpModel::<Backend>::new(MlpModelConfig::default(), &device));
        let mut optimizer = AdamWConfig::new().with_weight_decay(1e-4).init();

        let mut initial_loss = None;
        let mut final_loss = 0.0;

        for step in 0..100 {
            let (new_model, loss) =
                train_on_samples(model, &mut optimizer, &samples, &device, 1e-3);

            model = new_model;

            if step == 0 {
                initial_loss = Some(loss);
            }

            final_loss = loss.total;
        }

        let initial_loss = initial_loss.unwrap();

        println!("initial loss: {initial_loss}");
        println!("final loss:   {final_loss}");

        assert!(
            final_loss < initial_loss.total,
            "loss did not decrease: initial={initial_loss}, final={final_loss}"
        );
    }

    type TestBackend = burn::backend::Autodiff<burn::backend::Flex>;

    const INPUT_SIZE: usize = 88 * 3;

    fn test_samples() -> Vec<TrainingSample> {
        vec![
            TrainingSample {
                state: vec![0.0; INPUT_SIZE],
                policy: {
                    let mut policy = vec![0.0; ACTION_SIZE];
                    policy[0] = 1.0;
                    policy
                },
                value: 1.0,
            },
            TrainingSample {
                state: vec![1.0; INPUT_SIZE],
                policy: {
                    let mut policy = vec![0.0; ACTION_SIZE];
                    policy[1] = 1.0;
                    policy
                },
                value: -1.0,
            },
        ]
    }

    #[test]
    fn validation_step_returns_finite_loss() {
        let device = Default::default();
        let model = HexGoModel::Mlp(MlpModel::<TestBackend>::new(
            MlpModelConfig::default(),
            &device,
        ));

        let samples = test_samples();

        let loss = validation_step(&model, &samples, 256, &device);

        assert!(
            loss.total.is_finite(),
            "validation loss must be finite, got {loss:?}"
        );
    }

    #[test]
    fn validation_step_does_not_require_optimizer() {
        let device = Default::default();
        let model = HexGoModel::Mlp(MlpModel::<TestBackend>::new(
            MlpModelConfig::default(),
            &device,
        ));
        let samples = test_samples();

        let loss = validation_step(&model, &samples, 256, &device);

        assert!(loss.total.is_finite());
    }

    #[test]
    fn saved_model_can_be_loaded() {
        use burn::{
            module::{AutodiffModule, Module},
            record::CompactRecorder,
        };

        type TestBackend = Autodiff<Flex>;
        type InferenceBackend = Flex;

        let device = Default::default();

        let model = MlpModel::<TestBackend>::new(MlpModelConfig::default(), &device);

        let inference_model: MlpModel<InferenceBackend> = model.valid();

        let path = std::env::temp_dir().join("hexgo-test-model");

        inference_model
            .save_file(&path, &CompactRecorder::new())
            .expect("failed to save model");

        let _loaded_model = MlpModel::<InferenceBackend>::new(MlpModelConfig::default(), &device)
            .load_file(&path, &CompactRecorder::new(), &device)
            .expect("failed to load model");

        let _ = std::fs::remove_file(format!("{}.mpk", path.display()));
    }
}
