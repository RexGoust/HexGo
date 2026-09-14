mod dataset;
mod evaluation;
mod loss;
mod pipeline;
mod sampler;
mod self_play;
mod tensor;
mod train;
fn main() {
    pipeline::run(1, 1);
}
