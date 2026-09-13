mod dataset;
mod loss;
mod self_play;
mod tensor;
mod train;
use std::fs;

use burn::{
    backend::{Autodiff, Flex},
    module::Module,
    optim::AdamConfig,
    record::CompactRecorder,
};
use hex_go::ai::model::HexGoModel;
use rand::seq::SliceRandom;

use crate::{
    self_play::generate_self_play_games,
    train::{train_on_samples, validation_step},
};
type Backend = Autodiff<Flex>;

const SELF_PLAY_GAMES: usize = 32;
const MCTS_ITERATIONS: usize = 32;

const TRAIN_RATIO: f32 = 0.9;
const BATCH_SIZE: usize = 256;
const EPOCHS: usize = 5;

fn main() {
    let device = Default::default();

    let mut samples = generate_self_play_games(SELF_PLAY_GAMES, MCTS_ITERATIONS);

    println!("generated {} samples", samples.len());

    // Shuffle before splitting to avoid keeping positions from the same games together.
    samples.shuffle(&mut rand::rng());

    let split_index = (samples.len() as f32 * TRAIN_RATIO) as usize;
    let (train_samples, validation_samples) = samples.split_at(split_index);

    println!(
        "train={}, validation={}",
        train_samples.len(),
        validation_samples.len()
    );

    let mut model = HexGoModel::<Backend>::new(&device);

    let mut optimizer = AdamConfig::new().init();

    for epoch in 0..EPOCHS {
        for (batch_index, batch) in train_samples.chunks(BATCH_SIZE).enumerate() {
            let (new_model, loss) = train_on_samples(model, &mut optimizer, batch, &device, 1e-3);

            model = new_model;

            println!("epoch={epoch}, batch={batch_index}, loss={loss}");
        }

        let validation_loss = validation_step(&model, validation_samples, &device);

        println!("epoch={epoch}, validation_loss={validation_loss}");
    }

    fs::create_dir_all("checkpoints").expect("failed to create checkpoints directory");

    model
        .save_file("checkpoints/hexgo", &CompactRecorder::new())
        .expect("failed to save model");

    println!("model saved to checkpoints/hexgo");
}
