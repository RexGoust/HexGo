use crate::ai::model::{HexGoModel, MlpModel, MlpModelConfig, ModelConfig};
use burn::prelude::Backend;
use burn::tensor::Device;
use burn::{module::Module, record::CompactRecorder};
use burn_store::{BurnpackStore, ModuleSnapshot};
use clap::ValueEnum;
use std::{fs, path::Path};
#[derive(Debug, Clone, ValueEnum, Copy, PartialEq, Eq)]
#[value(rename_all = "lower")]
pub enum StoreType {
    BPK,
    MPK,
}

pub fn get_store_type(path: &str) -> Option<StoreType> {
    let path_obj = Path::new(path);
    let ext = path_obj
        .extension()
        .and_then(|e| e.to_str())
        .unwrap_or("")
        .to_lowercase();

    match ext.as_str() {
        "bpk" => return Some(StoreType::BPK),
        "mpk" => return Some(StoreType::MPK),
        "" => {}
        _ => {
            println!("Unsupported model format: .{}", ext);
            return None;
        }
    }

    for suffix in ["mpk", "bpk"] {
        let candidate = path_obj.with_extension(suffix);
        if candidate.is_file() {
            return match suffix {
                "bpk" => Some(StoreType::BPK),
                "mpk" => Some(StoreType::MPK),
                _ => unreachable!(),
            };
        }
    }

    println!("No model file found for base path: {}", path);
    None
}

fn load_config(path: impl AsRef<Path>) -> ModelConfig {
    let json = match std::fs::read_to_string(path) {
        Ok(s) => s,
        Err(_) => {
            println!("can't read config file, use default config");
            return ModelConfig::Mlp(MlpModelConfig::default());
        }
    };

    match ModelConfig::parse_with_fallback(&json) {
        Ok(config) => config,
        Err(_) => {
            println!("can't parse config, use default config");
            ModelConfig::Mlp(MlpModelConfig::default())
        }
    }
}

pub fn load_model<B: Backend>(path: impl AsRef<Path>, device: &Device<B>) -> HexGoModel<B> {
    let model_type = get_store_type(path.as_ref().to_str().unwrap())
        .unwrap_or_else(|| panic!("failed to load checkpoint from {}", path.as_ref().display()));

    let config = load_config(path.as_ref().with_file_name("config.json"));

    match config {
        ModelConfig::Mlp(mlp_config) => {
            let model = match model_type {
                StoreType::MPK => {
                    println!("warning this store type(mpk) is departured");
                    MlpModel::new(mlp_config, device)
                        .load_file(path.as_ref().to_path_buf(), &CompactRecorder::new(), device)
                        .unwrap_or_else(|_| {
                            panic!("failed to load checkpoint from {}", path.as_ref().display())
                        })
                }
                StoreType::BPK => {
                    let mut store = BurnpackStore::from_file(path.as_ref());

                    let mut model = MlpModel::new(mlp_config, device);

                    let result = model.load_from(&mut store).unwrap_or_else(|_| {
                        panic!("failed to load checkpoint from {}", path.as_ref().display())
                    });

                    if !result.missing.is_empty() {
                        eprintln!("Missing tensors: {:?}", result.missing);
                    }
                    model
                }
            };

            HexGoModel::Mlp(model)
        }
    }
}

pub fn save_model<B: Backend>(path: impl AsRef<Path>, model: HexGoModel<B>, store_type: StoreType) {
    if let Some(parent) = path.as_ref().parent()
        && !parent.as_os_str().is_empty()
    {
        fs::create_dir_all(parent).expect("failed to create checkpoints directory");
    }

    let json = serde_json::to_string(&model.config()).unwrap();
    std::fs::write(path.as_ref().with_file_name("config.json"), json)
        .expect("failed to save config.json");

    match model {
        HexGoModel::Mlp(mlp) => match store_type {
            StoreType::BPK => {
                let mut store = BurnpackStore::from_file(path.as_ref()).overwrite(true);
                mlp.save_into(&mut store).expect("failed to save model");
            }
            StoreType::MPK => {
                println!("warning this type is departured");
                mlp.save_file(path.as_ref().to_path_buf(), &CompactRecorder::new())
                    .expect("failed to save model");
            }
        },
    }

    println!("model saved to {}", path.as_ref().display());
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ai::backend::Backend as TestBackend;
    use crate::ai::encoder::INPUT_SIZE;
    use burn::Tensor;
    use rand::RngExt;

    #[test]
    fn save_and_load_model_preserves_weights() {
        let device = Default::default();
        let model =
            HexGoModel::<TestBackend>::new(ModelConfig::Mlp(MlpModelConfig::default()), &device);

        let input = Tensor::<TestBackend, 2>::zeros([1, INPUT_SIZE], &device);
        let expected_output = model.forward(input.clone());

        let temp_dir =
            std::env::temp_dir().join(format!("hexgo-test-store-{}", rand::rng().random::<u64>()));
        let model_path = temp_dir.join("sub_dir").join("model");

        save_model(&model_path, model, StoreType::MPK);
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
        let model =
            HexGoModel::<TestBackend>::new(ModelConfig::Mlp(MlpModelConfig::default()), &device);

        let temp_dir = std::env::temp_dir().join(format!(
            "hexgo-test-store-ext-{}",
            rand::rng().random::<u64>()
        ));
        let model_path = temp_dir.join("model");

        save_model(&model_path, model, StoreType::MPK);

        let mpk_path = temp_dir.join("model.mpk");
        let _loaded_model = load_model::<TestBackend>(&mpk_path, &device);

        let _ = fs::remove_dir_all(temp_dir);
    }

    #[test]
    fn save_and_load_bpk_preserves_weights() {
        let device = Default::default();
        let config = MlpModelConfig { hidden_size: 64 };
        let model = HexGoModel::<TestBackend>::new(ModelConfig::Mlp(config.clone()), &device);

        let input = Tensor::<TestBackend, 2>::zeros([1, INPUT_SIZE], &device);
        let expected_output = model.forward(input.clone());

        let temp_dir = std::env::temp_dir().join(format!(
            "hexgo-test-store-bpk-{}",
            rand::rng().random::<u64>()
        ));
        let model_path = temp_dir.join("sub_dir").join("model");

        save_model(&model_path, model, StoreType::BPK);
        let loaded_model = load_model::<TestBackend>(&model_path, &device);

        assert_eq!(loaded_model.config(), ModelConfig::Mlp(config));

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
    fn load_model_supports_explicit_bpk_extension() {
        let device = Default::default();
        let model =
            HexGoModel::<TestBackend>::new(ModelConfig::Mlp(MlpModelConfig::default()), &device);

        let temp_dir = std::env::temp_dir().join(format!(
            "hexgo-test-store-bpk-ext-{}",
            rand::rng().random::<u64>()
        ));
        let model_path = temp_dir.join("model");

        save_model(&model_path, model, StoreType::BPK);

        let bpk_path = temp_dir.join("model.bpk");
        let _loaded_model = load_model::<TestBackend>(&bpk_path, &device);

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
