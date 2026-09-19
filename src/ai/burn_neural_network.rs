use std::sync::Arc;

use burn::{
    Tensor,
    tensor::{TensorData, activation::softmax},
};
use burn_store::{BurnpackStore, ModuleSnapshot};

use crate::ai::{
    backend::{InferBackend, InferDevice, default_infer_device},
    encoder::{adjacency_tensor, encode_game_gnn_tensor},
    model::{HexGoModel, MlpModel, MlpModelConfig, ModelConfig, gnn::GnnModel},
};

use crate::{
    ai::{
        encoder::{INPUT_SIZE, encode_game_mlp},
        neural_network::{Evaluation, NeuralNetwork},
    },
    game::{
        Game,
        action::{ACTION_SIZE, Action, PASS_INDEX},
        board::VertexId,
        player::Player,
    },
};

const MODEL: &[u8] = include_bytes!("../../model/gnn/model.bpk");
const CONFIG: &[u8] = include_bytes!("../../model/gnn/config.json");
pub struct BurnNeuralNetwork {
    model: HexGoModel<InferBackend>,
    device: InferDevice,
}

impl BurnNeuralNetwork {
    #[allow(clippy::clone_on_copy)]
    pub fn load() -> Self {
        let device: &InferDevice = &default_infer_device();
        let s = std::str::from_utf8(CONFIG).unwrap();
        let config = match ModelConfig::parse_with_fallback(s) {
            Ok(config) => config,
            Err(_) => {
                println!("can't parse config, use default config");
                ModelConfig::Mlp(MlpModelConfig::default())
            }
        };
        let mut store = BurnpackStore::from_static(MODEL).zero_copy(false);
        let model = match config {
            ModelConfig::Mlp(cfg) => {
                let mut mlp = MlpModel::<InferBackend>::new(cfg, device);
                mlp.load_from(&mut store).expect("failed to load model");
                HexGoModel::Mlp(mlp)
            }
            ModelConfig::Gnn(cfg) => {
                let mut gnn = GnnModel::<InferBackend>::new(cfg, device);
                gnn.load_from(&mut store).expect("failed to load model");
                HexGoModel::Gnn(gnn)
            }
        };
        Self {
            model,
            // `Device` is a type alias: `FlexDevice` is `Copy`, `CudaDevice` is not.
            // Using `clone` uniformly keeps both backends working.
            device: device.clone(),
        }
    }
    #[allow(clippy::clone_on_copy)]
    pub fn from_model(model: &HexGoModel<InferBackend>) -> Self {
        let device: &InferDevice = &Default::default();
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
        let output = match &self.model {
            HexGoModel::Mlp(m) => {
                let input = encode_game_mlp(game, player);

                let input_tensor = Tensor::<InferBackend, 2>::from_data(
                    TensorData::new(input, [1, INPUT_SIZE]),
                    &self.device,
                );

                m.forward(input_tensor)
            }
            HexGoModel::Gnn(m) => {
                let x = encode_game_gnn_tensor::<InferBackend>(game, player, &self.device)
                    .unsqueeze::<3>();
                let adj = adjacency_tensor::<InferBackend>(game.board(), &self.device);
                m.forward(x, adj)
            }
        };

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
