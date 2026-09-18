use crate::{argument::PlayArgs, dataset::save_samples, self_play::generate_self_play_games};
use hex_go::ai::{
    backend::*,
    burn_neural_network::BurnNeuralNetwork,
    model::{HexGoModel, store::load_model},
    neural_mcts::{NeuralConfig, NeuralMcts},
};
use rand::seq::SliceRandom;
pub fn self_play(args: PlayArgs) {
    let device = default_device();
    let model: HexGoModel<Backend> = load_model(args.model, &device);

    let mut samples = generate_self_play_games(
        args.model_type,
        args.games as usize,
        args.iterations as usize,
        || {
            NeuralMcts::new(
                BurnNeuralNetwork::from_model(&model),
                NeuralConfig { add_noise: true },
            )
        },
    );

    samples.shuffle(&mut rand::rng());

    let result = save_samples(&args.output, &samples);

    match result {
        Err(err) => {
            println!("can't save sample, {}", err);
        }

        Ok(_) => {
            println!(
                "generated {} samples, saved to {}",
                samples.len(),
                args.output
            );
        }
    }
}
