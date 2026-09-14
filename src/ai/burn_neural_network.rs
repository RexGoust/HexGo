use std::sync::Arc;

use burn::{
    Tensor,
    backend::{Flex, flex::FlexDevice},
    module::Module,
    record::{HalfPrecisionSettings, NamedMpkBytesRecorder, Recorder},
    tensor::{TensorData, activation::softmax},
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

type Backend = Flex;

const MODEL: &[u8] = include_bytes!("../../assets/hexgo.mpk");

pub struct BurnNeuralNetwork {
    model: HexGoModel<Backend>,
    device: FlexDevice,
}

impl BurnNeuralNetwork {
    pub fn load() -> Self {
        let device = &Default::default();

        let record = NamedMpkBytesRecorder::<HalfPrecisionSettings>::new()
            .load(MODEL.to_vec(), device)
            .expect("failed to load model");

        let model = HexGoModel::<Backend>::new(device).load_record(record);

        Self {
            model,
            device: *device,
        }
    }

    pub fn from_model(model: &HexGoModel<Backend>) -> Self {
        let device = &Default::default();
        Self {
            model: model.clone(),
            device: *device,
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
