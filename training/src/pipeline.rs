use std::fs;

use burn::{
    backend::{Autodiff, Flex, flex::FlexDevice},
    module::{AutodiffModule, Module},
    optim::AdamConfig,
    record::CompactRecorder,
};
use hex_go::ai::{
    burn_neural_network::BurnNeuralNetwork,
    mcts::Mcts,
    model::HexGoModel,
    neural_mcts::{NeuralConfig, NeuralMcts},
};
use rand::seq::SliceRandom;

use crate::{
    dataset::TrainingSample,
    self_play::generate_self_play_games,
    train::{train_on_samples, validation_step},
};
type Backend = Autodiff<Flex>;

const SELF_PLAY_GAMES: usize = 32;
const MCTS_ITERATIONS: usize = 32;

const TRAIN_RATIO: f32 = 0.9;
const BATCH_SIZE: usize = 256;
const EPOCHS: usize = 5;

fn train_version_v0(device: &FlexDevice) {
    let t = std::time::Instant::now();

    println!("v0: start training...");
    let mut samples = generate_self_play_games(SELF_PLAY_GAMES, MCTS_ITERATIONS, Mcts::new);

    println!("v0: generated {} samples", samples.len());

    // Shuffle before splitting to avoid keeping positions from the same games together.
    samples.shuffle(&mut rand::rng());

    let model = HexGoModel::<Backend>::new(device);

    let model = train(model, &samples, device);

    save_model(0, model);

    println!("v0: train finish, time comsume: {:?}", t.elapsed());
}

fn load_model(version: usize, device: &FlexDevice) -> HexGoModel<Backend> {
    let path = format!("checkpoints/v{}/model.mpk", version);

    HexGoModel::new(device)
        .load_file(path, &CompactRecorder::new(), device)
        .unwrap()
}

fn save_model(version: usize, model: HexGoModel<Backend>) {
    fs::create_dir_all(format!("checkpoints/v{}", version))
        .expect("failed to create checkpoints directory");

    model
        .save_file(
            format!("checkpoints/v{}/model", version),
            &CompactRecorder::new(),
        )
        .expect("failed to save model");

    println!("v{0}: model saved to checkpoints/v{0}/model", version);
}

fn generate_self_play_data(model: &HexGoModel<Backend>) -> Vec<TrainingSample> {
    let model = model.valid();

    let mut samples = generate_self_play_games(SELF_PLAY_GAMES, MCTS_ITERATIONS, || {
        NeuralMcts::new(
            BurnNeuralNetwork::from_model(&model),
            NeuralConfig { add_noise: true },
        )
    });

    samples.shuffle(&mut rand::rng());

    samples
}

fn split_samples(samples: &[TrainingSample]) -> (&[TrainingSample], &[TrainingSample]) {
    let split_index = (samples.len() as f32 * TRAIN_RATIO) as usize;
    samples.split_at(split_index)
}

fn train(
    mut model: HexGoModel<Backend>,
    samples: &[TrainingSample],
    device: &FlexDevice,
) -> HexGoModel<Backend> {
    let (train_samples, validation_samples) = split_samples(samples);

    println!(
        "train={}, validation={}",
        train_samples.len(),
        validation_samples.len()
    );

    let mut optimizer = AdamConfig::new().init();

    for epoch in 0..EPOCHS {
        for (batch_index, batch) in train_samples.chunks(BATCH_SIZE).enumerate() {
            let (new_model, loss) = train_on_samples(model, &mut optimizer, batch, device, 1e-3);

            model = new_model;

            println!("epoch={epoch}, batch={batch_index}, loss={loss}");
        }

        let validation_loss = validation_step(&model, validation_samples, device);

        println!("epoch={epoch}, validation_loss={validation_loss}");
    }

    model
}

pub fn run() {
    let device = &Default::default();

    train_version_v0(device);

    for version in 1..10 {
        let t = std::time::Instant::now();

        println!("v{}: start training...", version);

        let model = load_model(version - 1, device);

        let samples = generate_self_play_data(&model);

        println!("v{}: generated {} samples", version, samples.len());

        let model = train(model, &samples, device);

        save_model(version, model);

        println!(
            "v{}: train finish, time comsume: {:?}",
            version,
            t.elapsed()
        );
    }
}
