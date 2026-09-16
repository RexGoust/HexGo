use std::sync::Arc;

use burn::{
    Tensor,
    module::Module,
    record::{HalfPrecisionSettings, NamedMpkBytesRecorder, Recorder},
    tensor::{TensorData, activation::softmax},
};

use crate::{
    ai::backend::default_device,
    ai::backend::{Backend, Device},
};

use crate::{
    ai::{
        encoder::{INPUT_SIZE, encode_game},
        model::HexGoModel,
        neural_network::{Evaluation, NeuralNetwork},
    },
    game::{
        Game,
        action::{ACTION_SIZE, Action, PASS_INDEX},
        board::VertexId,
        player::Player,
    },
};

const MODEL: &[u8] = include_bytes!("../../assets/hexgo.mpk");

pub struct BurnNeuralNetwork {
    model: HexGoModel<Backend>,
    device: Device,
}

impl BurnNeuralNetwork {
    #[allow(clippy::clone_on_copy)]
    pub fn load() -> Self {
        let device: &Device = &default_device();

        let record = NamedMpkBytesRecorder::<HalfPrecisionSettings>::new()
            .load(MODEL.to_vec(), device)
            .expect("failed to load model");

        let model = HexGoModel::<Backend>::new(device).load_record(record);

        Self {
            model,
            // `Device` is a type alias: `FlexDevice` is `Copy`, `CudaDevice` is not.
            // Using `clone` uniformly keeps both backends working.
            device: device.clone(),
        }
    }
    #[allow(clippy::clone_on_copy)]
    pub fn from_model(model: &HexGoModel<Backend>) -> Self {
        let device: &Device = &Default::default();
        Self {
            model: model.clone(),
            // `Device` is a type alias: `FlexDevice` is `Copy`, `CudaDevice` is not.
            // Using `clone` uniformly keeps both backends working.
            device: device.clone(),
        }
    }
}

impl NeuralNetwork for BurnNeuralNetwork {
    fn evaluate(&self, game: &Game, player: Player) -> Evaluation {
        let input = encode_game(game, player);

        let input_tensor =
            Tensor::<Backend, 2>::from_data(TensorData::new(input, [1, INPUT_SIZE]), &self.device);

        let output = self.model.forward(input_tensor);

        let policy_probs = softmax(output.policy, 1);
        let policy_values: Vec<f32> = policy_probs.into_data().to_vec().unwrap();

        let mut policy = Vec::with_capacity(ACTION_SIZE);

        for (index, &probability) in policy_values.iter().enumerate().take(ACTION_SIZE) {
            let action = if index == PASS_INDEX {
                Action::Pass
            } else {
                Action::Move(VertexId::new(index))
            };
            policy.push((action, probability));
        }

        let value = output.value.into_data().to_vec::<f32>().unwrap()[0];

        Evaluation { policy, value }
    }
}

impl NeuralNetwork for Arc<BurnNeuralNetwork> {
    fn evaluate(&self, game: &Game, player: Player) -> Evaluation {
        self.as_ref().evaluate(game, player)
    }
}
