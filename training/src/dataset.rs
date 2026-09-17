#![allow(dead_code)]
use serde::{Deserialize, Serialize};
use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::Path;
use zstd::stream::read::Decoder;
use zstd::stream::write::Encoder;

#[derive(Serialize, Deserialize, Clone, Debug, PartialEq)]
pub struct TrainingSample {
    pub state: Vec<f32>,
    pub policy: Vec<f32>,
    pub value: f32,
}

pub fn save_samples(path: &str, samples: &[TrainingSample]) -> std::io::Result<()> {
    if let Some(parent) = Path::new(path).parent()
        && !parent.as_os_str().is_empty()
    {
        std::fs::create_dir_all(parent)?;
    }

    let file = File::create(path)?;
    let writer = BufWriter::new(file);

    let mut encoder = Encoder::new(writer, 3)?;

    bincode::serialize_into(&mut encoder, samples).map_err(std::io::Error::other)?;

    let mut writer = encoder.finish()?;
    writer.flush()?;
    Ok(())
}

pub fn load_samples(path: &str) -> std::io::Result<Vec<TrainingSample>> {
    let file = File::open(path)?;
    let mut decoder = Decoder::new(file)?;
    bincode::deserialize_from(&mut decoder).map_err(std::io::Error::other)
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::RngExt;

    #[test]
    fn save_and_load_samples_preserves_data() {
        let samples = vec![
            TrainingSample {
                state: vec![1.0, 0.0, -1.0],
                policy: vec![0.1, 0.9],
                value: 0.5,
            },
            TrainingSample {
                state: vec![0.0, 1.0, 0.0],
                policy: vec![0.5, 0.5],
                value: -1.0,
            },
        ];

        let temp_dir = std::env::temp_dir().join(format!(
            "hexgo-test-dataset-{}",
            rand::rng().random::<u64>()
        ));
        let file_path = temp_dir.join("sub_dir").join("samples.bin.zst");
        let path_str = file_path.to_str().unwrap();

        save_samples(path_str, &samples).expect("failed to save samples");
        let loaded = load_samples(path_str).expect("failed to load samples");

        assert_eq!(loaded, samples);

        let _ = std::fs::remove_dir_all(temp_dir);
    }

    #[test]
    fn load_samples_fails_on_non_existent_path() {
        let non_existent = format!(
            "/tmp/hexgo-missing-samples-{}.bin.zst",
            rand::rng().random::<u64>()
        );
        let result = load_samples(&non_existent);
        assert!(result.is_err());
    }
}
