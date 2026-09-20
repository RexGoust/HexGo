use std::fmt;

use burn::{Tensor, tensor::backend::Backend};
use hex_go::{
    ai::{
        backend::{CpuBackend, CudaBackend, cpu_device, cuda_device, switch_model_backend},
        burn_neural_network::BurnNeuralNetwork,
        dummy_network::DummyNetwork,
        encoder::adjacency_tensor,
        model::{HexGoModel, store::*},
        neural_mcts::{NeuralConfig, NeuralMcts, StepState},
        search::Search,
    },
    board_layout::BoardDefinition,
    game::{
        Game, GameResult,
        action::Action::{self, Pass},
        player::Player,
    },
};
use rayon::iter::{IntoParallelIterator, ParallelIterator};

use crate::{
    argument::EvaluateArgs, device::DeviceKind, eval_slot::EvalSlot, model_type::ModelType,
    sampler::sample_action_by_temperature, train_network::forward_batch,
};

pub struct ActiveEvalGame {
    pub index: usize,
    pub game: Game,
    pub mcts: NeuralMcts<DummyNetwork>,
    pub state: StepState,
    pub input_buf: Vec<f32>,
    pub actions: usize,
}

impl ActiveEvalGame {
    pub fn new(index: usize) -> Self {
        let game = create_game();
        let mut mcts = NeuralMcts::new(DummyNetwork, NeuralConfig::default());
        mcts.start_search(&game);
        Self {
            index,
            game,
            mcts,
            state: StepState::Initial,
            input_buf: Vec::new(),
            actions: 0,
        }
    }

    pub fn is_candidate_turn(&self) -> bool {
        let candidate_black = self.index.is_multiple_of(2);
        match self.game.current_player() {
            Player::Black => candidate_black,
            Player::White => !candidate_black,
        }
    }

    pub fn is_finished(&self) -> bool {
        self.game.result().is_some()
    }

    pub fn step_action(&mut self) {
        if self.is_finished() {
            return;
        }

        let action = match self.mcts.finish_search() {
            Some(search) => {
                let temperature = if self.actions < 10 { 1.0 } else { 0.0 };
                sample_action_by_temperature(&search.policy, temperature)
            }
            None => Pass,
        };

        match action {
            Action::Move(vertex) => {
                self.game.play_move(vertex).unwrap();
            }
            Action::Pass => {
                self.game.pass_turn().unwrap();
            }
        }
        self.actions += 1;
        if self.actions >= MAX_ACTIONS {
            println!("action over {} times, quitting game...", MAX_ACTIONS);
            let _ = self.game.pass_turn();
            let _ = self.game.pass_turn();
        }

        self.mcts.start_search(&self.game);
    }
}

pub struct EvaluationResult {
    pub games: u32,
    pub wins: u32,
    pub losses: u32,
    pub draws: u32,
    pub win_rate: f32,
    pub score_rate: f32,
}

impl fmt::Display for EvaluationResult {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{}/{} wins ({:.2}%), {} losses, {} draws, score rate: {:.2}%",
            self.wins,
            self.games,
            self.win_rate * 100.0,
            self.losses,
            self.draws,
            self.score_rate * 100.0,
        )
    }
}
const MAX_ACTIONS: usize = 1000;

pub struct EvaluationConfig {
    pub candidate: String,

    pub baseline: String,

    pub games: usize,

    pub iterations: usize,

    pub infer_size: usize,

    pub infer_device: DeviceKind,
}

impl From<EvaluateArgs> for EvaluationConfig {
    fn from(args: EvaluateArgs) -> Self {
        Self {
            candidate: args.candidate,
            baseline: args.baseline,
            games: args.games as usize,
            iterations: args.iterations as usize,
            infer_size: args.infer_size,
            infer_device: args.infer_device,
        }
    }
}

pub fn create_evaluate(config: EvaluationConfig) {
    println!("starting evaluate");

    println!(
        "candidate: {}, baseline: {}",
        config.candidate, config.baseline
    );

    match config.infer_device {
        DeviceKind::Cpu => {
            let device = cpu_device();
            let candidate = load_model::<CpuBackend>(config.candidate, &device);
            let baseline = load_model::<CpuBackend>(config.baseline, &device);

            start_evaluate(
                candidate,
                baseline,
                &device,
                config.infer_size,
                config.games,
                config.iterations,
                config.infer_device,
            );
        }
        DeviceKind::Cuda => {
            let device = cuda_device();
            let candidate = load_model::<CudaBackend>(config.candidate, &device);
            let baseline = load_model::<CudaBackend>(config.baseline, &device);

            start_evaluate(
                candidate,
                baseline,
                &device,
                config.infer_size,
                config.games,
                config.iterations,
                config.infer_device,
            );
        }
    };
}

pub fn start_evaluate<B: Backend>(
    candidate: HexGoModel<B>,
    baseline: HexGoModel<B>,
    device: &B::Device,
    batch_size: usize,
    games: usize,
    iterations: usize,
    infer_device: DeviceKind,
) -> EvaluationResult {
    let result = match infer_device {
        DeviceKind::Cpu => {
            let cpu_dev = cpu_device();

            let candidate_cpu = switch_model_backend::<B, CpuBackend>(candidate, &cpu_dev);
            let baseline_cpu = switch_model_backend::<B, CpuBackend>(baseline, &cpu_dev);

            evaluate_model_cpu(
                || {
                    NeuralMcts::new(
                        BurnNeuralNetwork::from_model(&candidate_cpu),
                        NeuralConfig::default(),
                    )
                },
                || {
                    NeuralMcts::new(
                        BurnNeuralNetwork::from_model(&baseline_cpu),
                        NeuralConfig::default(),
                    )
                },
                games,
                iterations,
            )
        }
        DeviceKind::Cuda => {
            evaluate_model_cuda(candidate, baseline, device, batch_size, games, iterations)
        }
    };

    println!("evaluate result: {}", result);
    result
}

fn play_model(
    black: &mut dyn Search,
    white: &mut dyn Search,
    game: &mut Game,
    iterations: usize,
) -> GameResult {
    let mut actions = 0usize;

    while game.result().is_none() {
        let player = game.current_player();

        let search = match player {
            Player::Black => black.search(game, iterations),
            Player::White => white.search(game, iterations),
        };

        let action = match search {
            Some(search) => {
                let temperature = if actions < 10 { 1.0 } else { 0.0 };
                sample_action_by_temperature(&search.policy, temperature)
            }
            None => Pass,
        };

        match action {
            Action::Move(vertex) => {
                game.play_move(vertex).unwrap();
            }
            Action::Pass => {
                game.pass_turn().unwrap();
            }
        }
        actions += 1;
        if actions >= MAX_ACTIONS {
            println!("action over {} times, quitting game...", MAX_ACTIONS);
            let _ = game.pass_turn();
            let _ = game.pass_turn();
            break;
        }
    }

    game.result().unwrap()
}

fn evaluate_model_cpu<S, FC, FB>(
    candidate: FC,
    baseline: FB,
    games: usize,
    iterations: usize,
) -> EvaluationResult
where
    S: Search,
    FC: Fn() -> S + Sync,
    FB: Fn() -> S + Sync,
{
    let results: Vec<(usize, GameResult)> = (0..games)
        .into_par_iter()
        .map(|index| {
            let mut white = if index % 2 == 0 {
                baseline()
            } else {
                candidate()
            };

            let mut black = if index % 2 == 0 {
                candidate()
            } else {
                baseline()
            };

            let mut game = create_game();
            let result = play_model(&mut black, &mut white, &mut game, iterations);
            (index, result)
        })
        .collect();

    calculate_result(results)
}

fn create_game() -> Game {
    let board = BoardDefinition::compact().graph().clone();

    Game::new(board)
}

fn calculate_result(results: impl IntoIterator<Item = (usize, GameResult)>) -> EvaluationResult {
    let results = results.into_iter();

    let mut games = 0;
    let mut wins = 0;
    let mut losses = 0;
    let mut draws = 0;

    for (index, result) in results {
        games += 1;

        let candidate_black = index % 2 == 0;

        match result {
            GameResult::WinByScore { winner, .. } | GameResult::WinByResignation { winner } => {
                let candidate_won = (candidate_black && winner == Player::Black)
                    || (!candidate_black && winner == Player::White);

                if candidate_won {
                    wins += 1;
                } else {
                    losses += 1;
                }
            }
            GameResult::Draw => {
                draws += 1;
            }
        }
    }

    let win_rate = if games == 0 {
        0.0
    } else {
        wins as f32 / games as f32
    };

    let score_rate = if games == 0 {
        0.0
    } else {
        (wins as f32 + draws as f32 * 0.5) / games as f32
    };

    EvaluationResult {
        games,
        wins,
        losses,
        draws,
        win_rate,
        score_rate,
    }
}

fn get_model_type<B: Backend>(model: &HexGoModel<B>) -> ModelType {
    match &model {
        HexGoModel::Gnn(_) => ModelType::Gnn,
        HexGoModel::Mlp(_) => ModelType::Mlp,
    }
}

fn evaluate_model_cuda<B: Backend>(
    candidate: HexGoModel<B>,
    baseline: HexGoModel<B>,
    device: &B::Device,
    batch_size: usize,
    total_games: usize,
    iterations: usize,
) -> EvaluationResult
where
{
    let cand_type = get_model_type(&candidate);
    let base_type = get_model_type(&baseline);

    let cand_adj = match &candidate {
        HexGoModel::Gnn(_) => adjacency_tensor::<B>(BoardDefinition::compact().graph(), device),
        _ => Tensor::zeros([1, 1], device),
    };
    let base_adj = match &baseline {
        HexGoModel::Gnn(_) => adjacency_tensor::<B>(BoardDefinition::compact().graph(), device),
        _ => Tensor::zeros([1, 1], device),
    };

    let mut results: Vec<(usize, GameResult)> = Vec::new();
    let mut completed = 0;

    let initial_games = batch_size.min(total_games);
    let count_0 = initial_games.div_ceil(2);

    let mut slots = [
        EvalSlot::new(0..count_0),
        EvalSlot::new(count_0..initial_games),
    ];
    let mut started = slots[0].games.len() + slots[1].games.len();

    std::thread::scope(|s| {
        let (req_tx, req_rx) =
            std::sync::mpsc::sync_channel::<(usize, Vec<Vec<f32>>, Vec<Vec<f32>>)>(2);

        let (resp_tx, resp_rx) = std::sync::mpsc::sync_channel(2);
        // GPU worker
        s.spawn(move || {
            while let Ok((id, cand_inputs, base_inputs)) = req_rx.recv() {
                let cand_res = if !cand_inputs.is_empty() {
                    let refs: Vec<&Vec<f32>> = cand_inputs.iter().collect();
                    forward_batch(&candidate, device, &cand_adj, &refs)
                } else {
                    (Vec::new(), Vec::new())
                };

                let base_res = if !base_inputs.is_empty() {
                    let refs: Vec<&Vec<f32>> = base_inputs.iter().collect();
                    forward_batch(&baseline, device, &base_adj, &refs)
                } else {
                    (Vec::new(), Vec::new())
                };

                if resp_tx.send((id, cand_res, base_res)).is_err() {
                    break;
                }
            }
        });

        let mut in_flight = 0;
        for (id, slot) in slots.iter_mut().enumerate() {
            let (cand_inputs, base_inputs) = slot.select(cand_type, base_type);
            if !cand_inputs.is_empty() || !base_inputs.is_empty() {
                req_tx.send((id, cand_inputs, base_inputs)).unwrap();
                in_flight += 1;
            }
        }

        while in_flight > 0 {
            let (id, cand_res, base_res) = resp_rx.recv().unwrap();
            in_flight -= 1;

            let slot = &mut slots[id];
            slot.update(&cand_res, &base_res);
            slot.step += 1;

            if slot.step >= iterations {
                slot.commit_actions(&mut results, &mut completed, &mut started, total_games);
            }

            if !slot.games.is_empty() && completed < total_games {
                loop {
                    let (cand_inputs, base_inputs) = slot.select(cand_type, base_type);
                    if !cand_inputs.is_empty() || !base_inputs.is_empty() {
                        req_tx.send((id, cand_inputs, base_inputs)).unwrap();
                        in_flight += 1;
                        break;
                    }

                    slot.step += 1;
                    if slot.step >= iterations {
                        slot.commit_actions(
                            &mut results,
                            &mut completed,
                            &mut started,
                            total_games,
                        );
                    }

                    if slot.games.is_empty() || completed >= total_games {
                        break;
                    }
                }
            }
        }

        drop(req_tx);
    });

    calculate_result(results)
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn calculate_result_counts_candidate_wins() {
        let results = vec![
            (
                0,
                GameResult::WinByScore {
                    winner: Player::Black,
                    margin: 1.0,
                },
            ),
            (
                1,
                GameResult::WinByScore {
                    winner: Player::White,
                    margin: 1.0,
                },
            ),
        ];

        let result = calculate_result(results);

        assert_eq!(result.games, 2);
        assert_eq!(result.wins, 2);
        assert_eq!(result.losses, 0);
        assert_eq!(result.draws, 0);
        assert_eq!(result.win_rate, 1.0);
    }

    #[test]
    fn calculate_result_counts_baseline_wins() {
        let results = vec![
            (
                0,
                GameResult::WinByScore {
                    winner: Player::White,
                    margin: 1.0,
                },
            ),
            (
                1,
                GameResult::WinByScore {
                    winner: Player::Black,
                    margin: 1.0,
                },
            ),
        ];

        let result = calculate_result(results);

        assert_eq!(result.games, 2);
        assert_eq!(result.wins, 0);
        assert_eq!(result.losses, 2);
        assert_eq!(result.draws, 0);
        assert_eq!(result.win_rate, 0.0);
    }

    #[test]
    fn calculate_result_counts_mixed_results() {
        let results = vec![
            (
                0,
                GameResult::WinByScore {
                    winner: Player::Black,
                    margin: 1.0,
                },
            ),
            (
                1,
                GameResult::WinByScore {
                    winner: Player::Black,
                    margin: 1.0,
                },
            ),
            (2, GameResult::Draw),
            (3, GameResult::Draw),
        ];

        let result = calculate_result(results);

        assert_eq!(result.games, 4);
        assert_eq!(result.wins, 1);
        assert_eq!(result.losses, 1);
        assert_eq!(result.draws, 2);
        assert_eq!(result.win_rate, 0.25);
    }

    #[test]
    fn candidate_side_alternates_by_index() {
        let results = vec![
            (
                0,
                GameResult::WinByScore {
                    winner: Player::Black,
                    margin: 1.0,
                },
            ),
            (
                1,
                GameResult::WinByScore {
                    winner: Player::White,
                    margin: 1.0,
                },
            ),
            (
                2,
                GameResult::WinByScore {
                    winner: Player::Black,
                    margin: 1.0,
                },
            ),
            (
                3,
                GameResult::WinByScore {
                    winner: Player::White,
                    margin: 1.0,
                },
            ),
        ];

        let result = calculate_result(results);

        assert_eq!(result.wins, 4);
        assert_eq!(result.losses, 0);
    }
}
