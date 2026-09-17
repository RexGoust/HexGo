mod argument;
mod convert;
mod dataset;
mod evaluation;
mod loss;
mod pipeline;
mod play;
mod sampler;
mod self_play;
mod tensor;
mod train;

use clap::Parser;

use crate::{
    argument::{Cli, Command},
    convert::convert_model,
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
        Command::Convert(args) => {
            convert_model(&args.input, &args.output);
        }
        Command::Play(args) => {
            play::self_play(args);
        }
    }
}
