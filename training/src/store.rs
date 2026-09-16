use burn::prelude::Backend;
use burn::tensor::Device;
use burn::{module::Module, record::CompactRecorder};
use hex_go::ai::model::HexGoModel;
use std::{fs, path::Path};
pub fn load_model<B: Backend>(path: impl AsRef<Path>, device: &Device<B>) -> HexGoModel<B> {
    HexGoModel::new(device)
        .load_file(path.as_ref().to_path_buf(), &CompactRecorder::new(), device)
        .unwrap_or_else(|_| panic!("failed to load checkpoint from {}", path.as_ref().display()))
}

pub fn save_model<B: Backend>(path: impl AsRef<Path>, model: HexGoModel<B>) {
    if let Some(parent) = path.as_ref().parent()
        && !parent.as_os_str().is_empty()
    {
        fs::create_dir_all(parent).expect("failed to create checkpoints directory");
    }

    model
        .save_file(path.as_ref().to_path_buf(), &CompactRecorder::new())
        .expect("failed to save model");

    println!("model saved to {}", path.as_ref().display());
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::Tensor;
    use hex_go::ai::backend::Backend as TestBackend;
    use hex_go::ai::encoder::INPUT_SIZE;
    use rand::RngExt;

    #[test]
    fn save_and_load_model_preserves_weights() {
        let device = Default::default();
        let model = HexGoModel::<TestBackend>::new(&device);

        let input = Tensor::<TestBackend, 2>::zeros([1, INPUT_SIZE], &device);
        let expected_output = model.forward(input.clone());

        let temp_dir =
            std::env::temp_dir().join(format!("hexgo-test-store-{}", rand::rng().random::<u64>()));
        let model_path = temp_dir.join("sub_dir").join("model");

        save_model(&model_path, model);
        let loaded_model = load_model::<TestBackend>(&model_path, &device);

        let actual_output = loaded_model.forward(input);

        let expected_val = expected_output.value.into_data().to_vec::<f32>().unwrap();
        let actual_val = actual_output.value.into_data().to_vec::<f32>().unwrap();
        assert_eq!(expected_val.len(), actual_val.len());
        for (e, a) in expected_val.iter().zip(&actual_val) {
            assert!(
                (e - a).abs() < 1e-3,
                "value mismatch: expected {e}, got {a}"
            );
        }

        let expected_policy = expected_output.policy.into_data().to_vec::<f32>().unwrap();
        let actual_policy = actual_output.policy.into_data().to_vec::<f32>().unwrap();
        assert_eq!(expected_policy.len(), actual_policy.len());
        for (e, a) in expected_policy.iter().zip(&actual_policy) {
            assert!(
                (e - a).abs() < 1e-3,
                "policy mismatch: expected {e}, got {a}"
            );
        }

        let _ = fs::remove_dir_all(temp_dir);
    }

    #[test]
    fn load_model_supports_explicit_mpk_extension() {
        let device = Default::default();
        let model = HexGoModel::<TestBackend>::new(&device);

        let temp_dir = std::env::temp_dir().join(format!(
            "hexgo-test-store-ext-{}",
            rand::rng().random::<u64>()
        ));
        let model_path = temp_dir.join("model");

        save_model(&model_path, model);

        let mpk_path = temp_dir.join("model.mpk");
        let _loaded_model = load_model::<TestBackend>(&mpk_path, &device);

        let _ = fs::remove_dir_all(temp_dir);
    }

    #[test]
    #[should_panic(expected = "failed to load checkpoint from")]
    fn load_model_panics_when_file_not_found() {
        let device = Default::default();
        let non_existent =
            std::env::temp_dir().join(format!("hexgo-missing-{}", rand::rng().random::<u64>()));
        let _model = load_model::<TestBackend>(&non_existent, &device);
    }
}
