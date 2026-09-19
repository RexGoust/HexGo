use burn::Tensor;
use burn::tensor::TensorData;
use burn::tensor::activation::softmax;
use burn::tensor::backend::Backend;
use crossbeam_channel::{Receiver, Sender, bounded};
use hex_go::ai::encoder::{MLP_INPUT_SIZE, VERTEX_COUNT, adjacency_tensor, encode_game_gnn};

use hex_go::ai::model::HexGoModel;
use hex_go::ai::model::gnn::FEATURE_DIM;
use hex_go::game::action::{ACTION_SIZE, Action, PASS_INDEX};
use hex_go::game::board::VertexId;
use hex_go::{
    ai::{
        encoder::encode_game_mlp,
        neural_network::{Evaluation, NeuralNetwork},
    },
    game::{Game, player::Player},
};

use crate::model_type::ModelType;

pub struct Task {
    pub input: Vec<f32>,
}

pub struct TaskResult {
    pub evaluation: Evaluation,
}

pub struct Job {
    pub task: Task,
    pub reply: Sender<TaskResult>,
}

pub struct TrainNetwork {
    tx: Sender<Job>,
    model_type: ModelType,
}

impl TrainNetwork {
    pub fn new(tx: Sender<Job>, model_type: ModelType) -> Self {
        Self { tx, model_type }
    }

    pub fn serve<B: Backend>(
        rx: Receiver<Job>,
        model: HexGoModel<B>,
        device: B::Device,
        max_batch_size: usize,
    ) {
        let mut jobs = Vec::with_capacity(max_batch_size);

        let adj = match &model {
            HexGoModel::Gnn(_) => {
                let board = hex_go::board_layout::BoardDefinition::compact()
                    .graph()
                    .clone();
                adjacency_tensor::<B>(&board, &device)
            }
            _ => Tensor::zeros([1, 1], &device),
        };

        while let Ok(first) = rx.recv() {
            jobs.push(first);
            while jobs.len() < max_batch_size {
                match rx.try_recv() {
                    Ok(job) => jobs.push(job),
                    Err(_) => break,
                }
            }

            let batch_size = jobs.len();

            let (policy, value) = match &model {
                HexGoModel::Mlp(m) => {
                    let mut flat = Vec::with_capacity(batch_size * MLP_INPUT_SIZE);
                    for job in &jobs {
                        flat.extend_from_slice(&job.task.input);
                    }

                    let input = Tensor::<B, 1>::from_data(
                        TensorData::new(flat, [batch_size * MLP_INPUT_SIZE]),
                        &device,
                    )
                    .reshape([batch_size, MLP_INPUT_SIZE]);

                    let out = m.forward(input);
                    (softmax(out.policy, 1), out.value)
                }
                HexGoModel::Gnn(m) => {
                    let state_size = FEATURE_DIM * VERTEX_COUNT;

                    let mut flat = Vec::with_capacity(batch_size * state_size);

                    for job in &jobs {
                        flat.extend_from_slice(&job.task.input);
                    }

                    let x = Tensor::<B, 1>::from_data(
                        TensorData::new(flat, [batch_size * state_size]),
                        &device,
                    )
                    .reshape([batch_size, VERTEX_COUNT, FEATURE_DIM]);

                    let out = m.forward(x, adj.clone());

                    (softmax(out.policy, 1), out.value)
                }
            };

            let policies: Vec<f32> = policy.into_data().to_vec().unwrap();
            let values: Vec<f32> = value.into_data().to_vec::<f32>().unwrap();

            for (i, job) in jobs.drain(..).enumerate() {
                let start = i * ACTION_SIZE;

                let p_slice = &policies[start..start + ACTION_SIZE];
                let mut policy = Vec::with_capacity(ACTION_SIZE);

                for (idx, &p) in p_slice.iter().enumerate() {
                    let action = if idx == PASS_INDEX {
                        Action::Pass
                    } else {
                        Action::Move(VertexId::new(idx))
                    };

                    policy.push((action, p));
                }

                let _ = job.reply.send(TaskResult {
                    evaluation: Evaluation {
                        policy,
                        value: values[i],
                    },
                });
            }
        }
    }
}

impl NeuralNetwork for TrainNetwork {
    fn evaluate(&self, game: &Game, player: Player) -> Evaluation {
        let task = match &self.model_type {
            ModelType::Mlp => {
                let input = encode_game_mlp(game, player);

                Task { input }
            }
            ModelType::Gnn => {
                let input = encode_game_gnn(game, player);

                Task { input }
            }
        };
        let (reply_tx, reply_rx) = bounded::<TaskResult>(1);
        self.tx
            .send(Job {
                task,
                reply: reply_tx,
            })
            .unwrap();

        let res = reply_rx.recv().unwrap();

        res.evaluation
    }
}
