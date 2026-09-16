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
    argument::TrainArgs,
    dataset::TrainingSample,
    evaluation::evaluate_model,
    self_play::generate_self_play_games,
    train::{train_on_samples, validation_step},
};
type Backend = Autodiff<Flex>;

const EVALUATE_GAMES: usize = 200;
const EVALUATE_ITERATIONS: usize = 800;

const TRAIN_RATIO: f32 = 0.9;

const MIN_SCORE_RATE: f32 = 0.55;

pub struct TrainingConfig {
    pub games: usize,

    pub iterations: usize,

    pub runs: usize,

    pub start_version: usize,

    pub epochs: usize,

    pub batch_size: usize,
}

impl From<TrainArgs> for TrainingConfig {
    fn from(args: TrainArgs) -> Self {
        Self {
            games: args.games as usize,
            iterations: args.iterations as usize,
            runs: args.runs as usize,
            start_version: args.start_version,
            epochs: args.epochs,
            batch_size: args.batch_size,
        }
    }
}

pub struct Pipeline {
    config: TrainingConfig,
    current_version: usize,
}

impl Pipeline {
    pub fn new(config: TrainingConfig) -> Self {
        Self {
            current_version: config.start_version,
            config,
        }
    }
    fn train_version_v0(&self, device: &FlexDevice) {
        let t = std::time::Instant::now();

        println!("v0: start training...");
        let mut samples =
            generate_self_play_games(self.config.games, self.config.iterations, Mcts::new);

        println!("v0: generated {} samples", samples.len());

        // Shuffle before splitting to avoid keeping positions from the same games together.
        samples.shuffle(&mut rand::rng());

        let model = HexGoModel::<Backend>::new(device);

        let model = self.train(model, samples, device);

        Self::save_model(0, model);

        println!("v0: train finished, time consumed: {:?}", t.elapsed());
    }

    fn load_model(version: usize, device: &FlexDevice) -> HexGoModel<Backend> {
        let path = format!("checkpoints/v{}/model", version);

        HexGoModel::new(device)
            .load_file(&path, &CompactRecorder::new(), device)
            .unwrap_or_else(|_| panic!("failed to load checkpoint from {path}"))
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

    fn generate_self_play_data(&self, model: &HexGoModel<Backend>) -> Vec<TrainingSample> {
        let model = model.valid();

        let mut samples =
            generate_self_play_games(self.config.games, self.config.iterations, || {
                NeuralMcts::new(
                    BurnNeuralNetwork::from_model(&model),
                    NeuralConfig { add_noise: true },
                )
            });

        samples.shuffle(&mut rand::rng());

        samples
    }

    fn split_samples(
        mut samples: Vec<TrainingSample>,
    ) -> (Vec<TrainingSample>, Vec<TrainingSample>) {
        let split_index = (samples.len() as f32 * TRAIN_RATIO) as usize;

        let validation_samples = samples.split_off(split_index);

        (samples, validation_samples)
    }

    fn train(
        &self,
        mut model: HexGoModel<Backend>,
        samples: Vec<TrainingSample>,
        device: &FlexDevice,
    ) -> HexGoModel<Backend> {
        let (mut train_samples, validation_samples) = Self::split_samples(samples);

        println!(
            "train={}, validation={}",
            train_samples.len(),
            validation_samples.len()
        );

        let mut optimizer = AdamConfig::new().init();

        for epoch in 0..self.config.epochs {
            train_samples.shuffle(&mut rand::rng());
            for (batch_index, batch) in train_samples.chunks(self.config.batch_size).enumerate() {
                let (new_model, loss) =
                    train_on_samples(model, &mut optimizer, batch, device, 1e-3);

                model = new_model;

                println!("epoch={epoch}, batch={batch_index}, loss={loss}");
            }

            let validation_loss = validation_step(&model, &validation_samples, device);

            println!("epoch={epoch}, validation_loss={validation_loss}");
        }

        model
    }

    pub fn evaluate(
        &self,
        candidate: &HexGoModel<Backend>,
        baseline: &HexGoModel<Backend>,
    ) -> bool {
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

        println!("v{} evaluate result: {}", self.current_version, result);

        result.score_rate >= MIN_SCORE_RATE
    }

    pub fn run(&mut self) {
        let device = &Default::default();
        if self.current_version == 0 {
            self.train_version_v0(device);
            self.current_version += 1;
        }

        for _ in 0..self.config.runs {
            let t = std::time::Instant::now();

            println!("v{}: start training...", self.current_version);

            let model = Self::load_model(self.current_version - 1, device);
            let baseline = model.clone();
            let samples = self.generate_self_play_data(&model);

            println!(
                "v{}: generated {} samples",
                self.current_version,
                samples.len()
            );

            let candidate = self.train(model, samples, device);

            let success = self.evaluate(&candidate, &baseline);

            if success {
                Self::save_model(self.current_version, candidate);
                println!(
                    "v{}: train finished, time consumed: {:?}",
                    self.current_version,
                    t.elapsed()
                );
                self.current_version += 1;
            } else {
                println!(
                    "v{}: train failed, time consumed: {:?}",
                    self.current_version,
                    t.elapsed()
                );
            }
        }
    }
}
