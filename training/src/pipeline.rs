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
    evaluation::evaluate_model,
    self_play::generate_self_play_games,
    train::{train_on_samples, validation_step},
};
type Backend = Autodiff<Flex>;

const SELF_PLAY_GAMES: usize = 1000;
const MCTS_ITERATIONS: usize = 800;

const EVALUATE_GAMES: usize = 200;
const EVALUATE_ITERATIONS: usize = 800;

const TRAIN_RATIO: f32 = 0.9;
const BATCH_SIZE: usize = 256;
const EPOCHS: usize = 10;

const MIN_SCORE_RATE: f32 = 0.55;

fn train_version_v0(device: &FlexDevice) {
    let t = std::time::Instant::now();

    println!("v0: start training...");
    let mut samples = generate_self_play_games(SELF_PLAY_GAMES, MCTS_ITERATIONS, Mcts::new);

    println!("v0: generated {} samples", samples.len());

    // Shuffle before splitting to avoid keeping positions from the same games together.
    samples.shuffle(&mut rand::rng());

    let model = HexGoModel::<Backend>::new(device);

    let model = train(model, samples, device);

    save_model(0, model);

    println!("v0: train finish, time comsumed: {:?}", t.elapsed());
}

fn load_model(version: usize, device: &FlexDevice) -> HexGoModel<Backend> {
    let path = format!("checkpoints/v{}/model", version);

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

fn split_samples(mut samples: Vec<TrainingSample>) -> (Vec<TrainingSample>, Vec<TrainingSample>) {
    let split_index = (samples.len() as f32 * TRAIN_RATIO) as usize;

    let validation_samples = samples.split_off(split_index);

    (samples, validation_samples)
}

fn train(
    mut model: HexGoModel<Backend>,
    samples: Vec<TrainingSample>,
    device: &FlexDevice,
) -> HexGoModel<Backend> {
    let (mut train_samples, validation_samples) = split_samples(samples);

    println!(
        "train={}, validation={}",
        train_samples.len(),
        validation_samples.len()
    );

    let mut optimizer = AdamConfig::new().init();

    for epoch in 0..EPOCHS {
        train_samples.shuffle(&mut rand::rng());
        for (batch_index, batch) in train_samples.chunks(BATCH_SIZE).enumerate() {
            let (new_model, loss) = train_on_samples(model, &mut optimizer, batch, device, 1e-3);

            model = new_model;

            println!("epoch={epoch}, batch={batch_index}, loss={loss}");
        }

        let validation_loss = validation_step(&model, &validation_samples, device);

        println!("epoch={epoch}, validation_loss={validation_loss}");
    }

    model
}

pub fn evaluate(candidate: &HexGoModel<Backend>, baseline: &HexGoModel<Backend>) -> bool {
    let candidate = candidate.valid();
    let baseline = baseline.valid();

    let result = evaluate_model(
        || {
            NeuralMcts::new(
                BurnNeuralNetwork::from_model(&candidate),
                NeuralConfig::default(),
            )
        },
        || {
            NeuralMcts::new(
                BurnNeuralNetwork::from_model(&baseline),
                NeuralConfig::default(),
            )
        },
        EVALUATE_GAMES,
        EVALUATE_ITERATIONS,
    );

    println!("{result}");

    result.score_rate >= MIN_SCORE_RATE
}

pub fn run(iterations: usize, start_version: usize) {
    let device = &Default::default();
    let mut version = start_version;
    if version == 0 {
        train_version_v0(device);
    }

    for _ in 0..iterations {
        let t = std::time::Instant::now();

        println!("v{}: start training...", version);

        let model = load_model(version - 1, device);
        let baseline = model.clone();
        let samples = generate_self_play_data(&model);

        println!("v{}: generated {} samples", version, samples.len());

        let candidate = train(model, samples, device);

        let sucess = evaluate(&candidate, &baseline);

        if sucess {
            save_model(version, candidate);
            version += 1;
        }

        println!(
            "v{}: train {}, time comsumed: {:?}",
            version,
            if sucess { "finished" } else { "failed" },
            t.elapsed()
        );
    }
}
