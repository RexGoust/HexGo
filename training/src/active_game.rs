use hex_go::{
    ai::{
        dummy_network::DummyNetwork,
        encoder::{encode_game_gnn, encode_game_mlp},
        neural_mcts::{NeuralConfig, NeuralMcts, StepState},
    },
    board_layout::BoardDefinition,
    game::{
        Game,
        action::{ACTION_SIZE, Action, PASS_INDEX},
    },
};

use crate::{
    dataset::TrainingSample,
    model_type::ModelType,
    sampler::sample_action_by_temperature,
    self_play::{SelfPlayPosition, to_training_samples},
};

pub const MAX_ACTIONS: usize = 1000;

pub struct ActiveGame {
    pub game: Game,
    pub mcts: NeuralMcts<DummyNetwork>, // DummyNetwork as an empty placeholder.

    pub state: StepState,
    pub input_buf: Vec<f32>,

    pub positions: Vec<SelfPlayPosition>,
    pub moves: usize,
    pub passes: usize,
    pub actions: usize,
}

pub fn create_game() -> Game {
    let board = BoardDefinition::compact().graph().clone();

    Game::new(board)
}

pub fn policy_to_dense(policy: &[(Action, f32)]) -> Vec<f32> {
    let mut result = vec![0.0; ACTION_SIZE];

    for &(action, probability) in policy {
        result[action.index()] = probability;
    }

    result
}

impl ActiveGame {
    pub fn new() -> Self {
        let game = create_game();
        let mut mcts = NeuralMcts::new(DummyNetwork, NeuralConfig { add_noise: true });
        mcts.start_search(&game);
        Self {
            game,
            mcts,
            positions: Vec::new(),

            state: StepState::Initial,
            input_buf: Vec::new(),

            moves: 0,
            passes: 0,
            actions: 0,
        }
    }

    pub fn step_action(&mut self, model_type: ModelType) {
        if self.is_finished() {
            return;
        }

        let player = self.game.current_player();
        let state = match model_type {
            ModelType::Mlp => encode_game_mlp(&self.game, player),
            ModelType::Gnn => encode_game_gnn(&self.game, player),
        };
        let search = match self.mcts.finish_search() {
            Some(search) => search,
            None => {
                // No legal board move: pass.
                let mut policy = vec![0.0; ACTION_SIZE];
                policy[PASS_INDEX] = 1.0;

                self.positions.push(SelfPlayPosition {
                    state,
                    policy,
                    player,
                });
                self.game.pass_turn().unwrap();
                self.mcts.start_search(&self.game);
                return;
            }
        };

        let policy = policy_to_dense(&search.policy);

        let temperature = if self.actions < 30 { 1.0 } else { 0.0 };

        let action = sample_action_by_temperature(&search.policy, temperature);

        match action {
            Action::Move(vertex) => {
                self.moves += 1;

                self.game.play_move(vertex).unwrap();
            }
            Action::Pass => {
                self.passes += 1;

                self.game.pass_turn().unwrap();
            }
        }
        self.actions += 1;

        self.positions.push(SelfPlayPosition {
            state,
            policy,
            player,
        });

        if self.actions >= MAX_ACTIONS {
            println!("action over {} times, quitting game...", MAX_ACTIONS);
            let _ = self.game.pass_turn();
            let _ = self.game.pass_turn();
        }

        self.mcts.start_search(&self.game);
    }

    pub fn finish_game(&mut self) -> Vec<TrainingSample> {
        println!(
            "game finished: positions={}, moves={}, passes={}",
            self.positions.len(),
            self.moves,
            self.passes
        );

        let result = self.game.result().unwrap();

        to_training_samples(std::mem::take(&mut self.positions), result)
    }

    pub fn is_finished(&self) -> bool {
        self.game.result().is_some()
    }
}
