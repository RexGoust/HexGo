use std::{fs, path::Path};

use burn::backend::Autodiff;
use burn::{module::Module, record::CompactRecorder};
use hex_go::ai::backend::{Backend as InnerBackend, Device};
use hex_go::ai::model::HexGoModel;

type Backend = Autodiff<InnerBackend>;

pub fn load_model(path: impl AsRef<Path>, device: &Device) -> HexGoModel<Backend> {
    HexGoModel::new(device)
        .load_file(path.as_ref().to_path_buf(), &CompactRecorder::new(), device)
        .unwrap_or_else(|_| panic!("failed to load checkpoint from {}", path.as_ref().display()))
}

pub fn save_model(path: impl AsRef<Path>, model: HexGoModel<Backend>) {
    fs::create_dir_all(path.as_ref().to_path_buf().parent().unwrap())
        .expect("failed to create checkpoints directory");

    model
        .save_file(path.as_ref().to_path_buf(), &CompactRecorder::new())
        .expect("failed to save model");

    println!("model saved to {}", path.as_ref().display());
}
