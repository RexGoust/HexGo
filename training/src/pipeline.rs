use std::path::Path;

use burn::{
    backend::Autodiff, module::AutodiffModule, optim::AdamConfig, tensor::backend::Backend,
};
use hex_go::ai::{
    backend::{
        InferBackend, InferDevice, TrainDevice, default_infer_device, default_train_device,
        switch_model_backend,
    },
    mcts::Mcts,
    model::{
        HexGoModel, MlpModel, MlpModelConfig, ModelConfig,
        gnn::{GnnModel, GnnModelConfig},
        store::{self, StoreType},
    },
};
use rand::seq::SliceRandom;

use crate::{
    argument::TrainArgs,
    dataset::{self, TrainingSample},
    evaluation::{self},
    model_type::ModelType,
    self_play::{generate_samples_batched, generate_samples_local},
    train::{train_on_samples, validation_step},
};
use hex_go::ai::backend::TrainBackend as InnerTrainBackend;

type TrainBackend = Autodiff<InnerTrainBackend>;

const EVALUATE_GAMES: usize = 200;
const EVALUATE_ITERATIONS: usize = 800;

const TRAIN_RATIO: f32 = 0.9;

const MIN_SCORE_RATE: f32 = 0.55;

const CURRENT_TRAIN_MODEL_CONFIG: MlpModelConfig = MlpModelConfig { hidden_size: 256 };

const RECENT_GENERATIONS: usize = 4;
const CURRENT_VERSION_SAMPLE_RATIO: f64 = 0.7;

pub struct TrainingConfig {
    pub games: usize,

    pub iterations: usize,

    pub runs: usize,

    pub start_version: usize,

    pub epochs: usize,

    pub batch_size: usize,

    pub no_skip: bool,

    pub store_type: StoreType,

    pub force_save: bool,

    pub model_type: ModelType,

    pub reuse_data: bool,

    pub no_eval: bool,

    pub infer_size: usize,
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
            force_save: args.force_save,
            model_type: args.model_type,
            reuse_data: args.reuse_data,
            no_eval: args.no_eval,
            infer_size: args.infer_size,
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
    fn train_version_v0(&self, device: &TrainDevice) {
        let t = std::time::Instant::now();

        println!("v0: start training...");
        let mut samples = generate_samples_local(
            self.config.model_type,
            self.config.games,
            self.config.iterations,
            Mcts::new,
        );

        println!("v0: generated {} samples", samples.len());

        // Shuffle before splitting to avoid keeping positions from the same games together.
        samples.shuffle(&mut rand::rng());

        let model = match self.config.model_type {
            ModelType::Mlp => HexGoModel::Mlp(MlpModel::<TrainBackend>::new(
                CURRENT_TRAIN_MODEL_CONFIG,
                device,
            )),
            ModelType::Gnn => HexGoModel::Gnn(GnnModel::<TrainBackend>::new(
                GnnModelConfig::default(),
                device,
            )),
        };

        let model = self.train(model, samples, device);

        self.save_model(0, model);

        println!("v0: train finished, time consumed: {:?}", t.elapsed());
    }

    fn load_model<B: Backend>(&self, version: usize, device: &B::Device) -> HexGoModel<B> {
        let path = format!("checkpoints/{}/v{}/model", self.config.model_type, version);
        store::load_model(path, device)
    }

    fn save_model<B: Backend>(&self, version: usize, model: HexGoModel<B>) {
        let path = format!("checkpoints/{}/v{}/model", self.config.model_type, version);
        store::save_model(path, model, self.config.store_type);
    }

    fn save_samples(&self, version: usize, sample: &[TrainingSample]) {
        let path = format!(
            "data/{}/v{}/self_play.bin.zst",
            self.config.model_type, version
        );

        let result = dataset::save_samples(&path, sample);

        if let Err(err) = result {
            println!("can't save sample, {}", err);
        }
    }

    fn load_samples(&self, version: usize) -> Vec<TrainingSample> {
        let path = format!(
            "data/{}/v{}/self_play.bin.zst",
            self.config.model_type, version
        );

        let result = dataset::load_samples(&path);

        match result {
            Err(err) => {
                println!("can't load sample, {}", err);
                Vec::new()
            }
            Ok(samples) => samples,
        }
    }

    pub fn load_recent_samples(&self, version: usize, generations: usize) -> Vec<TrainingSample> {
        if generations == 0 {
            return Vec::new();
        }

        let start = version.saturating_sub(generations - 1);
        let mut history_samples = Vec::new();
        for v in start..version {
            let mut samples = self.load_samples(v);
            history_samples.append(&mut samples);
        }

        let mut current_samples = self.load_samples(version);

        if !current_samples.is_empty() && !history_samples.is_empty() {
            let target_history = ((current_samples.len() as f64)
                * ((1.0 - CURRENT_VERSION_SAMPLE_RATIO) / CURRENT_VERSION_SAMPLE_RATIO))
                .round() as usize;
            let target_history = if target_history == 0 {
                1
            } else {
                target_history
            };
            if history_samples.len() > target_history {
                history_samples.shuffle(&mut rand::rng());
                history_samples.truncate(target_history);
            }
        }

        let current_count = current_samples.len();
        let history_count = history_samples.len();

        let mut all = history_samples;
        all.append(&mut current_samples);

        println!(
            "loaded {} samples from {} generations (v{}: {}, history: {})",
            all.len(),
            generations,
            version,
            current_count,
            history_count
        );
        all
    }

    fn generate_self_play_data(
        &self,
        model: &HexGoModel<InferBackend>,
        device: &InferDevice,
    ) -> Vec<TrainingSample> {
        let mut samples = generate_samples_batched(
            model.clone(),
            *device,
            self.config.model_type,
            self.config.infer_size,
            self.config.games,
            self.config.iterations,
        );

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
        mut model: HexGoModel<TrainBackend>,
        samples: Vec<TrainingSample>,
        device: &TrainDevice,
    ) -> HexGoModel<TrainBackend> {
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

            let valid_model = model.valid();
            let validation_loss = validation_step(
                &valid_model,
                &validation_samples,
                self.config.batch_size,
                device,
            );

            println!("epoch={epoch}, validation_loss={validation_loss}");
        }

        model
    }

    pub fn evaluate(
        &self,
        candidate: HexGoModel<InferBackend>,
        baseline: HexGoModel<InferBackend>,
        device: InferDevice,
    ) -> bool {
        let result = evaluation::start_evaluate(
            candidate,
            baseline,
            device,
            self.config.infer_size,
            EVALUATE_GAMES,
            EVALUATE_ITERATIONS,
        );

        result.score_rate >= MIN_SCORE_RATE || self.config.force_save
    }

    fn should_skip_version(&self, version: usize) -> bool {
        if self.config.no_skip {
            return false;
        }

        let suffixes = ["mpk", "bpk"];
        let mut found: Option<String> = None;
        let mut error: Option<(String, std::io::Error)> = None;

        for suffix in &suffixes {
            let path_str = format!(
                "checkpoints/{}/v{}/model.{}",
                self.config.model_type, version, suffix
            );
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
        let train_device = &default_train_device();
        let infer_device = &default_infer_device();

        if self.current_version == 0 {
            if self.should_skip_version(0) {
                self.current_version = 1;
            } else {
                self.train_version_v0(train_device);
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

            let model = self.load_model(self.current_version - 1, train_device);

            let baseline = switch_model_backend(model.clone(), infer_device);
            let path = format!(
                "data/{}/v{}/self_play.bin.zst",
                self.config.model_type,
                self.current_version.saturating_sub(1)
            );
            let file_exists = Path::new(&path).is_file();

            if !self.config.reuse_data || !file_exists {
                let infer_model = &baseline;
                let samples = self.generate_self_play_data(infer_model, infer_device);

                println!(
                    "v{}: generated {} samples",
                    self.current_version,
                    samples.len()
                );

                self.save_samples(self.current_version - 1, &samples);
            }

            let mut samples = self
                .load_recent_samples(self.current_version.saturating_sub(1), RECENT_GENERATIONS);

            println!(
                "v{}: loaded total {} samples",
                self.current_version,
                samples.len()
            );

            samples.shuffle(&mut rand::rng());

            let model = match self.config.model_type {
                ModelType::Mlp => {
                    if model.config() == ModelConfig::Mlp(CURRENT_TRAIN_MODEL_CONFIG) {
                        model
                    } else {
                        HexGoModel::Mlp(MlpModel::new(CURRENT_TRAIN_MODEL_CONFIG, train_device))
                    }
                }
                ModelType::Gnn => {
                    if model.config() == ModelConfig::Gnn(GnnModelConfig::default()) {
                        model
                    } else {
                        HexGoModel::Gnn(GnnModel::new(GnnModelConfig::default(), train_device))
                    }
                }
            };

            let candidate = self.train(model, samples, train_device);
            let candidate = switch_model_backend(candidate, infer_device);
            let success =
                self.config.no_eval || self.evaluate(candidate.clone(), baseline, *infer_device);

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
                    "v{}: candidate rejected (score rate < 55%), baseline retained, time consumed: {:?}",
                    self.current_version,
                    t.elapsed()
                );
            }

            trained += 1;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::RngExt;

    fn test_pipeline() -> Pipeline {
        Pipeline::new(TrainingConfig {
            games: 1,
            iterations: 1,
            runs: 1,
            start_version: 0,
            epochs: 1,
            batch_size: 1,
            no_skip: false,
            store_type: StoreType::BPK,
            force_save: false,
            model_type: ModelType::Mlp,
            reuse_data: false,
            no_eval: false,
            infer_size: 1,
        })
    }

    #[test]
    fn load_recent_samples_returns_empty_when_generations_is_zero() {
        let pipeline = test_pipeline();
        let samples = pipeline.load_recent_samples(10, 0);
        assert!(samples.is_empty());
    }

    #[test]
    fn test_load_recent_samples_window() {
        let pipeline = test_pipeline();
        let base_version: usize = 99000 + rand::rng().random_range(1000..9000);
        let sample_v0 = vec![TrainingSample {
            state: vec![0.0],
            policy: vec![1.0],
            value: 0.0,
        }];
        let sample_v1 = vec![TrainingSample {
            state: vec![1.0],
            policy: vec![1.0],
            value: 1.0,
        }];
        let sample_v2 = vec![TrainingSample {
            state: vec![2.0],
            policy: vec![1.0],
            value: 2.0,
        }];

        pipeline.save_samples(base_version, &sample_v0);
        pipeline.save_samples(base_version + 1, &sample_v1);
        pipeline.save_samples(base_version + 2, &sample_v2);

        // Load 2 recent generations ending at base_version + 2 -> should load v1 and v2
        let loaded_2 = pipeline.load_recent_samples(base_version + 2, 2);
        assert_eq!(loaded_2.len(), 2);
        assert_eq!(loaded_2[0], sample_v1[0]);
        assert_eq!(loaded_2[1], sample_v2[0]);

        // Load 5 recent generations ending at base_version + 1 -> should load v0 and v1
        let loaded_all = pipeline.load_recent_samples(base_version + 1, 5);
        assert_eq!(loaded_all.len(), 2);
        assert_eq!(loaded_all[0], sample_v0[0]);
        assert_eq!(loaded_all[1], sample_v1[0]);

        // Cleanup
        let _ = std::fs::remove_dir_all(format!(
            "data/{}/v{}",
            pipeline.config.model_type, base_version
        ));
        let _ = std::fs::remove_dir_all(format!(
            "data/{}/v{}",
            pipeline.config.model_type,
            base_version + 1
        ));
        let _ = std::fs::remove_dir_all(format!(
            "data/{}/v{}",
            pipeline.config.model_type,
            base_version + 2
        ));
    }

    #[test]
    fn test_load_recent_samples_ratio_70_30() {
        let pipeline = test_pipeline();
        let base_version: usize = 88000 + rand::rng().random_range(1000..9000);
        let sample = TrainingSample {
            state: vec![0.0],
            policy: vec![1.0],
            value: 0.0,
        };
        let current_samples = vec![sample.clone(); 70];
        let history_samples = vec![sample.clone(); 100];

        pipeline.save_samples(base_version, &history_samples);
        pipeline.save_samples(base_version + 1, &current_samples);

        let loaded = pipeline.load_recent_samples(base_version + 1, 2);
        // Target history: round(70 * 0.3 / 0.7) = 30
        // Total loaded: 70 (v1) + 30 (v0) = 100
        assert_eq!(loaded.len(), 100);

        let _ = std::fs::remove_dir_all(format!(
            "data/{}/v{}",
            pipeline.config.model_type, base_version
        ));
        let _ = std::fs::remove_dir_all(format!(
            "data/{}/v{}",
            pipeline.config.model_type,
            base_version + 1
        ));
    }
}
