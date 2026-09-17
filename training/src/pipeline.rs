use std::path::Path;

use burn::{backend::Autodiff, module::AutodiffModule, optim::AdamConfig};
use hex_go::ai::{
    backend::default_device,
    burn_neural_network::BurnNeuralNetwork,
    mcts::Mcts,
    model::{
        HexGoModel, ModelConfig,
        store::{self, StoreType},
    },
    neural_mcts::{NeuralConfig, NeuralMcts},
};
use rand::seq::SliceRandom;

use crate::{
    argument::TrainArgs,
    dataset::TrainingSample,
    evaluation::{self},
    self_play::generate_self_play_games,
    train::{train_on_samples, validation_step},
};
use hex_go::ai::backend::{Backend as InnerBackend, Device};

type Backend = Autodiff<InnerBackend>;

const EVALUATE_GAMES: usize = 200;
const EVALUATE_ITERATIONS: usize = 800;

const TRAIN_RATIO: f32 = 0.9;

const MIN_SCORE_RATE: f32 = 0.55;

const CURRENT_TRAIN_MODEL_CONFIG: ModelConfig = ModelConfig { hidden_size: 256 };

pub struct TrainingConfig {
    pub games: usize,

    pub iterations: usize,

    pub runs: usize,

    pub start_version: usize,

    pub epochs: usize,

    pub batch_size: usize,

    pub no_skip: bool,

    pub store_type: StoreType,
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
            no_skip: args.no_skip,
            store_type: args.store_type,
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
    fn train_version_v0(&self, device: &Device) {
        let t = std::time::Instant::now();

        println!("v0: start training...");
        let mut samples =
            generate_self_play_games(self.config.games, self.config.iterations, Mcts::new);

        println!("v0: generated {} samples", samples.len());

        // Shuffle before splitting to avoid keeping positions from the same games together.
        samples.shuffle(&mut rand::rng());

        let model = HexGoModel::<Backend>::new(CURRENT_TRAIN_MODEL_CONFIG, device);

        let model = self.train(model, samples, device);

        self.save_model(0, model);

        println!("v0: train finished, time consumed: {:?}", t.elapsed());
    }

    fn load_model(version: usize, device: &Device) -> HexGoModel<Backend> {
        let path = format!("checkpoints/v{}/model", version);
        store::load_model(path, device)
    }

    fn save_model(&self, version: usize, model: HexGoModel<Backend>) {
        let path = format!("checkpoints/v{}/model", version);
        store::save_model(path, model, self.config.store_type);
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
        device: &Device,
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
        let result = evaluation::start_evaluate(
            &candidate.valid(),
            &baseline.valid(),
            EVALUATE_GAMES,
            EVALUATE_ITERATIONS,
        );

        result.score_rate >= MIN_SCORE_RATE
    }

    fn should_skip_version(&self, version: usize) -> bool {
        if self.config.no_skip {
            return false;
        }

        let suffixes = ["mpk", "bpk"];
        let mut found: Option<String> = None;
        let mut error: Option<(String, std::io::Error)> = None;

        for suffix in &suffixes {
            let path_str = format!("checkpoints/v{}/model.{}", version, suffix);
            let path = Path::new(&path_str);

            match path.try_exists() {
                Ok(true) => {
                    found = Some(path_str);
                    break;
                }
                Ok(false) => {}
                Err(e) => {
                    error = Some((path_str, e));
                    break;
                }
            }
        }

        if let Some((_path_str, e)) = error {
            eprintln!("failed to check checkpoint for v{}: {}", version, e);
            panic!();
        }

        if let Some(path_str) = found {
            println!(
                "checkpoint {} already exists, skipping (use --no-skip to override)",
                path_str
            );
            true
        } else {
            false
        }
    }

    pub fn run(&mut self) {
        let device = &default_device();
        if self.current_version == 0 {
            if self.should_skip_version(0) {
                self.current_version = 1;
            } else {
                self.train_version_v0(device);
                self.current_version = 1;
            }
        }
        let mut trained = 0;
        while trained < self.config.runs {
            if self.should_skip_version(self.current_version) {
                self.current_version += 1;
                continue;
            }

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

            let model = if model.config() == CURRENT_TRAIN_MODEL_CONFIG {
                model
            } else {
                HexGoModel::new(CURRENT_TRAIN_MODEL_CONFIG, device)
            };

            let candidate = self.train(model, samples, device);

            let success = self.evaluate(&candidate, &baseline);

            if success {
                self.save_model(self.current_version, candidate);
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

            trained += 1;
        }
    }
}
