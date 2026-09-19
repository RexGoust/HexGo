use burn::{
    module::Module,
    record::{BinBytesRecorder, FullPrecisionSettings, Recorder},
    tensor::backend::Backend,
};
use clap::ValueEnum;

use crate::ai::model::HexGoModel;

#[derive(Debug, Clone, ValueEnum, Copy, PartialEq, Eq)]
#[value(rename_all = "lower")]
pub enum BackendKind {
    Flex,
    Cuda,
}

pub fn switch_model_backend<B1, B2>(
    model: HexGoModel<B1>,
    target_device: &B2::Device,
) -> HexGoModel<B2>
where
    B1: Backend,
    B2: Backend,
{
    let config = model.config();

    let recorder = BinBytesRecorder::<FullPrecisionSettings>::default();
    let bytes = recorder
        .record(model.into_record(), ())
        .expect("failed to serialize model into bytes");

    let record2 = recorder
        .load(bytes, target_device)
        .expect("failed to load model record onto target device");

    HexGoModel::<B2>::new(config, target_device).load_record(record2)
}

#[cfg(feature = "cuda")]
pub type TrainBackend = burn::backend::Cuda<f32, i32>;

#[cfg(feature = "cuda")]
pub type TrainDevice = burn::backend::cuda::CudaDevice;

#[cfg(not(any(feature = "cuda")))]
pub type TrainBackend = burn::backend::Flex;

#[cfg(not(any(feature = "cuda")))]
pub type TrainDevice = burn::backend::flex::FlexDevice;

pub type InferBackend = TrainBackend;
pub type InferDevice = TrainDevice;

pub fn default_train_device() -> TrainDevice {
    Default::default()
}

pub fn default_infer_device() -> InferDevice {
    Default::default()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ai::{
        encoder::MLP_INPUT_SIZE,
        model::{
            ModelConfig,
            gnn::{FEATURE_DIM, GnnModelConfig},
            mlp::MlpModelConfig,
        },
    };
    use burn::Tensor;

    #[test]
    fn switch_model_backend_preserves_mlp_outputs() {
        let device = default_infer_device();
        let config = MlpModelConfig { hidden_size: 64 };
        let model = HexGoModel::<InferBackend>::new(ModelConfig::Mlp(config.clone()), &device);

        let input = Tensor::<InferBackend, 2>::zeros([1, MLP_INPUT_SIZE], &device);
        let expected_output = match &model {
            HexGoModel::Mlp(m) => m.forward(input.clone()),
            HexGoModel::Gnn(_) => unreachable!(),
        };

        let switched_model = switch_model_backend::<InferBackend, InferBackend>(model, &device);
        assert_eq!(switched_model.config(), ModelConfig::Mlp(config));

        let actual_output = match &switched_model {
            HexGoModel::Mlp(m) => m.forward(input),
            HexGoModel::Gnn(_) => unreachable!(),
        };

        let expected_val = expected_output.value.into_data().to_vec::<f32>().unwrap();
        let actual_val = actual_output.value.into_data().to_vec::<f32>().unwrap();
        assert_eq!(expected_val.len(), actual_val.len());
        for (e, a) in expected_val.iter().zip(&actual_val) {
            assert!(
                (e - a).abs() < 1e-4,
                "value mismatch: expected {e}, got {a}"
            );
        }

        let expected_policy = expected_output.policy.into_data().to_vec::<f32>().unwrap();
        let actual_policy = actual_output.policy.into_data().to_vec::<f32>().unwrap();
        assert_eq!(expected_policy.len(), actual_policy.len());
        for (e, a) in expected_policy.iter().zip(&actual_policy) {
            assert!(
                (e - a).abs() < 1e-4,
                "policy mismatch: expected {e}, got {a}"
            );
        }
    }

    #[test]
    fn switch_model_backend_preserves_gnn_outputs() {
        let device = default_infer_device();
        let config = GnnModelConfig {
            feature_dim: FEATURE_DIM,
            hidden_dim: 32,
            num_vertices: 4,
        };
        let model = HexGoModel::<InferBackend>::new(ModelConfig::Gnn(config.clone()), &device);

        let x = Tensor::<InferBackend, 2>::zeros([4, FEATURE_DIM], &device).unsqueeze::<3>();
        let adj = Tensor::<InferBackend, 2>::zeros([4, 4], &device);

        let expected_output = match &model {
            HexGoModel::Gnn(m) => m.forward(x.clone(), adj.clone()),
            HexGoModel::Mlp(_) => unreachable!(),
        };

        let switched_model = switch_model_backend::<InferBackend, InferBackend>(model, &device);
        assert_eq!(switched_model.config(), ModelConfig::Gnn(config));

        let actual_output = match &switched_model {
            HexGoModel::Gnn(m) => m.forward(x, adj),
            HexGoModel::Mlp(_) => unreachable!(),
        };

        let expected_val = expected_output.value.into_data().to_vec::<f32>().unwrap();
        let actual_val = actual_output.value.into_data().to_vec::<f32>().unwrap();
        assert_eq!(expected_val.len(), actual_val.len());
        for (e, a) in expected_val.iter().zip(&actual_val) {
            assert!(
                (e - a).abs() < 1e-4,
                "value mismatch: expected {e}, got {a}"
            );
        }

        let expected_policy = expected_output.policy.into_data().to_vec::<f32>().unwrap();
        let actual_policy = actual_output.policy.into_data().to_vec::<f32>().unwrap();
        assert_eq!(expected_policy.len(), actual_policy.len());
        for (e, a) in expected_policy.iter().zip(&actual_policy) {
            assert!(
                (e - a).abs() < 1e-4,
                "policy mismatch: expected {e}, got {a}"
            );
        }
    }
}
