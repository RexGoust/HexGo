use crate::{
    argument::PlayArgs,
    dataset::save_samples,
    device::DeviceKind,
    self_play::{generate_samples_batched, generate_samples_local},
};
use hex_go::ai::{
    backend::*,
    burn_neural_network::BurnNeuralNetwork,
    model::{HexGoModel, store::load_model},
    neural_mcts::{NeuralConfig, NeuralMcts},
};
use rand::seq::SliceRandom;
pub fn self_play(args: PlayArgs) {
    let mut samples = match args.infer_device {
        DeviceKind::Cpu => {
            let device = cpu_device();
            let model: HexGoModel<CpuBackend> = load_model(args.model, &device);
            generate_samples_local(
                args.model_type,
                args.games as usize,
                args.iterations as usize,
                || {
                    NeuralMcts::new(
                        BurnNeuralNetwork::from_model(&model),
                        NeuralConfig { add_noise: true },
                    )
                },
            )
        }
        DeviceKind::Cuda => {
            let device = cuda_device();
            let model: HexGoModel<CudaBackend> = load_model(args.model, &device);
            generate_samples_batched(
                model,
                &device,
                args.model_type,
                args.infer_size,
                args.games as usize,
                args.iterations as usize,
            )
        }
    };

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
