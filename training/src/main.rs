mod argument;
mod dataset;
mod evaluation;
mod loss;
mod pipeline;
mod sampler;
mod self_play;
mod store;
mod tensor;
mod train;
use clap::Parser;

use crate::{
    argument::{Cli, Command},
    evaluation::EvaluationConfig,
    pipeline::{Pipeline, TrainingConfig},
};
fn main() {
    let cli = Cli::parse();

    match cli.command {
        Command::Train(args) => {
            let mut pipeline = Pipeline::new(TrainingConfig::from(args));
            pipeline.run();
        }
        Command::Evaluate(args) => {
            evaluation::create_evaluate(EvaluationConfig::from(args));
        }
    }
}
