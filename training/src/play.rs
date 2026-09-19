use crate::{argument::PlayArgs, dataset::save_samples, self_play::generate_samples_batched};
use hex_go::ai::{
    backend::*,
    model::{HexGoModel, store::load_model},
};
use rand::seq::SliceRandom;
pub fn self_play(args: PlayArgs) {
    let device = default_infer_device();
    let model: HexGoModel<InferBackend> = load_model(args.model, &device);

    let mut samples = generate_samples_batched(
        model,
        device,
        args.model_type,
        args.batch_size,
        args.games as usize,
        args.iterations as usize,
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
