#![allow(dead_code)]
use burn::{
    Tensor,
    optim::{GradientsParams, Optimizer},
    tensor::{ElementConversion, backend::AutodiffBackend},
};

use crate::{
    dataset::TrainingSample,
    loss::{LossValue, pred_entropy, target_entropy, total_loss},
    tensor::samples_to_tensors,
};
use hex_go::ai::model::HexGoModel;

pub fn train_step<B: AutodiffBackend>(
    model: HexGoModel<B>,
    input: Tensor<B, 2>,
    target_policy: Tensor<B, 2>,
    target_value: Tensor<B, 2>,
    optimizer: &mut impl Optimizer<HexGoModel<B>, B>,
    learning_rate: f64,
) -> (HexGoModel<B>, LossValue) {
    let output = model.forward(input);

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

pub fn train_on_samples<B: AutodiffBackend>(
    model: HexGoModel<B>,
    optimizer: &mut impl Optimizer<HexGoModel<B>, B>,
    samples: &[TrainingSample],
    device: &B::Device,
    learning_rate: f64,
) -> (HexGoModel<B>, LossValue) {
    let (state, policy, value) = samples_to_tensors::<B>(samples, device);

    train_step(model, state, policy, value, optimizer, learning_rate)
}

pub fn validation_step<B: AutodiffBackend>(
    model: &HexGoModel<B>,
    samples: &[TrainingSample],
    device: &B::Device,
) -> LossValue {
    let (states, policies, values) = samples_to_tensors(samples, device);

    let output = model.forward(states);

    let loss = total_loss(
        output.policy.clone(),
        policies.clone(),
        output.value,
        values,
    );

    let total = loss.total.into_scalar().elem::<f32>();
    let policy = loss.policy.into_scalar().elem::<f32>();
    let value = loss.value.into_scalar().elem::<f32>();
    let target_entropy = target_entropy(policies);
    let pred_entropy = pred_entropy(output.policy.detach());
    LossValue {
        policy,
        value,
        total,
        target_entropy: Some(target_entropy),
        pred_entropy: Some(pred_entropy),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{dataset::TrainingSample, tensor::samples_to_tensors};
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

        let (input, target_policy, target_value) = samples_to_tensors::<Backend>(&samples, &device);

        let mut model =
            HexGoModel::Mlp(MlpModel::<Backend>::new(MlpModelConfig::default(), &device));
        let mut optimizer = AdamWConfig::new().with_weight_decay(1e-4).init();

        let mut initial_loss = None;
        let mut final_loss = 0.0;

        for step in 0..100 {
            let (new_model, loss) = train_step(
                model,
                input.clone(),
                target_policy.clone(),
                target_value.clone(),
                &mut optimizer,
                1e-3,
            );

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

        let loss = validation_step(&model, &samples, &device);

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

        let loss = validation_step(&model, &samples, &device);

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
