#![allow(dead_code)]
use serde::{Deserialize, Serialize};
use std::fs::File;
use std::io::{BufReader, BufWriter, Write};
use std::path::Path;
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
    let mut writer = BufWriter::new(file);
    bincode::serialize_into(&mut writer, samples).map_err(std::io::Error::other)?;
    writer.flush()?;
    Ok(())
}

pub fn load_samples(path: &str) -> std::io::Result<Vec<TrainingSample>> {
    let file = File::open(path)?;
    let mut reader = BufReader::new(file);
    bincode::deserialize_from(&mut reader).map_err(std::io::Error::other)
}
