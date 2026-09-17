use clap::{Args, Parser, Subcommand};
use hex_go::ai::model::store::StoreType;
#[derive(Parser)]
#[command(styles = clap_cargo::style::CLAP_STYLING)]
#[command(name = "hexgo-train", version, about = "HexGo AI tools")]
pub struct Cli {
    #[command(subcommand)]
    pub command: Command,
}

#[derive(Subcommand)]
pub enum Command {
    /// Train the model through self-play and evaluation pipeline
    Train(TrainArgs),

    /// Evaluate the model
    #[command(alias = "eval")]
    Evaluate(EvaluateArgs),

    /// Convert model checkpoint between storage formats (e.g. .mpk ↔ .bpk)
    #[command(alias = "conv")]
    Convert(ConvertArgs),
}

#[derive(Args)]
pub struct TrainArgs {
    /// Number of self-play games
    #[arg(short, long, default_value_t = 1000)]
    pub games: u32,

    /// Number of MCTS iterations
    #[arg(short, long, default_value_t = 800)]
    pub iterations: u32,

    /// Number of times to run the pipeline
    #[arg(short, long, default_value_t = 1)]
    pub runs: u32,

    /// Model version to start training from
    #[arg(long, default_value_t = 0)]
    pub start_version: usize,

    /// Number of training epochs per pipeline run
    #[arg(long, default_value_t = 30)]
    pub epochs: usize,

    /// Number of samples per training batch
    #[arg(long, default_value_t = 256)]
    pub batch_size: usize,

    /// Retrain and overwrite the checkpoint even if it already exists
    #[arg(long)]
    pub no_skip: bool,

    /// Model storage format to use for saving checkpoints
    #[arg(long, value_enum, default_value_t = StoreType::BPK)]
    pub store_type: StoreType,
}

#[derive(Args)]
pub struct EvaluateArgs {
    /// The path to the candidate model checkpoint.
    #[arg(short, long)]
    pub candidate: String,

    /// The path to the baseline model checkpoint.
    #[arg(short, long)]
    pub baseline: String,

    /// Number of evaluation games
    #[arg(short, long, default_value_t = 200)]
    pub games: u32,

    /// Number of MCTS iterations
    #[arg(short, long, default_value_t = 800)]
    pub iterations: u32,
}

#[derive(Args)]
pub struct ConvertArgs {
    /// The path to the input model checkpoint.
    #[arg(short, long)]
    pub input: String,

    /// The path to the output model checkpoint.
    #[arg(short, long)]
    pub output: String,
}
