use std::path::Path;

use hex_go::ai::{
    backend::{InferBackend, default_infer_device},
    model::{
        HexGoModel,
        store::{get_store_type, load_model, save_model},
    },
};

pub fn convert_model(input: impl AsRef<Path>, output: impl AsRef<Path>) {
    let input = input.as_ref();
    let output = output.as_ref();

    let output_str = output.to_str().unwrap_or_else(|| {
        eprintln!("Error: output path contains invalid UTF-8 characters");
        std::process::exit(1);
    });

    let store_type = get_store_type(output_str).unwrap_or_else(|| {
            eprintln!(
                "Error: cannot determine storage format for output path '{}'. Please specify a valid extension (.bpk or .mpk).",
                output.display()
            );
            std::process::exit(1);
        });

    let model: HexGoModel<InferBackend> = load_model(input, &default_infer_device());

    save_model(output, model, store_type);
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::Tensor;
    use hex_go::ai::{
        encoder::MLP_INPUT_SIZE,
        model::{MlpModelConfig, ModelConfig, store::StoreType},
    };
    use rand::RngExt;
    use std::fs;

    #[test]
    fn test_convert_model_between_formats() {
        let device = default_infer_device();
        let config = MlpModelConfig { hidden_size: 64 };
        let model = HexGoModel::<InferBackend>::new(ModelConfig::Mlp(config.clone()), &device);

        let input = Tensor::<InferBackend, 2>::zeros([1, MLP_INPUT_SIZE], &device);
        let expected_output = match &model {
            HexGoModel::Mlp(m) => m.forward(input.clone()),
            HexGoModel::Gnn(_) => unreachable!(),
        };

        let temp_dir = std::env::temp_dir().join(format!(
            "hexgo-test-convert-{}",
            rand::rng().random::<u64>()
        ));
        let mpk_path = temp_dir.join("model.mpk");
        let bpk_path = temp_dir.join("model.bpk");
        let back_mpk_path = temp_dir.join("model_back.mpk");

        // Save original model as MPK
        save_model(&mpk_path, model, StoreType::MPK);

        // Convert MPK -> BPK
        convert_model(&mpk_path, &bpk_path);
        let bpk_model: HexGoModel<InferBackend> = load_model(&bpk_path, &device);
        assert_eq!(bpk_model.config(), ModelConfig::Mlp(config.clone()));

        let bpk_output = match &bpk_model {
            HexGoModel::Mlp(m) => m.forward(input.clone()),
            HexGoModel::Gnn(_) => unreachable!(),
        };
        let expected_val = expected_output.value.into_data().to_vec::<f32>().unwrap();
        let bpk_val = bpk_output.value.into_data().to_vec::<f32>().unwrap();
        for (e, a) in expected_val.iter().zip(&bpk_val) {
            assert!(
                (e - a).abs() < 1e-3,
                "BPK value mismatch: expected {e}, got {a}"
            );
        }

        // Convert BPK -> MPK
        convert_model(&bpk_path, &back_mpk_path);
        let back_model: HexGoModel<InferBackend> = load_model(&back_mpk_path, &device);
        assert_eq!(back_model.config(), ModelConfig::Mlp(config));

        let back_output = match &back_model {
            HexGoModel::Mlp(m) => m.forward(input),
            HexGoModel::Gnn(_) => unreachable!(),
        };
        let back_val = back_output.value.into_data().to_vec::<f32>().unwrap();
        for (e, a) in expected_val.iter().zip(&back_val) {
            assert!(
                (e - a).abs() < 1e-3,
                "Back MPK value mismatch: expected {e}, got {a}"
            );
        }

        let _ = fs::remove_dir_all(temp_dir);
    }
}
