use std::fmt;

use burn::tensor::backend::Backend;
use hex_go::{
    ai::{
        backend::{InferDevice, default_infer_device},
        model::{HexGoModel, store::*},
        neural_mcts::{NeuralConfig, NeuralMcts},
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
    argument::EvaluateArgs,
    model_type::ModelType,
    sampler::sample_action_by_temperature,
    train_network::{Job, TrainNetwork},
};

use hex_go::ai::backend::InferBackend;

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

    pub batch_size: usize,
}

impl From<EvaluateArgs> for EvaluationConfig {
    fn from(args: EvaluateArgs) -> Self {
        Self {
            candidate: args.candidate,
            baseline: args.baseline,
            games: args.games as usize,
            iterations: args.iterations as usize,
            batch_size: args.batch_size,
        }
    }
}

pub fn create_evaluate(config: EvaluationConfig) {
    println!("starting evaluate");

    println!(
        "candidate: {}, baseline: {}",
        config.candidate, config.baseline
    );

    let device = default_infer_device();
    let candidate = load_model::<InferBackend>(config.candidate, &device);
    let baseline = load_model::<InferBackend>(config.baseline, &device);

    start_evaluate(
        candidate,
        baseline,
        device,
        config.batch_size,
        config.games,
        config.iterations,
    );
}

pub fn start_evaluate(
    candidate: HexGoModel<InferBackend>,
    baseline: HexGoModel<InferBackend>,
    device: InferDevice,
    batch_size: usize,
    games: usize,
    iterations: usize,
) -> EvaluationResult {
    let result = evaluate_model(candidate, baseline, device, batch_size, games, iterations);

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

fn evaluate_model<B: Backend>(
    candidate: HexGoModel<B>,
    baseline: HexGoModel<B>,
    device: B::Device,
    batch_size: usize,
    games: usize,
    iterations: usize,
) -> EvaluationResult
where
{
    let cand_type = get_model_type(&candidate);
    let base_type = get_model_type(&baseline);

    let (cand_tx, cand_rx) = crossbeam_channel::bounded::<Job>(4096);
    let (base_tx, base_rx) = crossbeam_channel::bounded::<Job>(4096);

    let cand_device = device.clone();
    let cand_handle = std::thread::spawn(move || {
        TrainNetwork::serve(cand_rx, candidate, cand_device, batch_size);
    });
    let base_handle = std::thread::spawn(move || {
        TrainNetwork::serve(base_rx, baseline, device, batch_size);
    });

    let results: Vec<(usize, GameResult)> = (0..games)
        .into_par_iter()
        .map(|index| {
            let mut white = if index % 2 == 0 {
                NeuralMcts::new(
                    TrainNetwork::new(base_tx.clone(), base_type),
                    NeuralConfig::default(),
                )
            } else {
                NeuralMcts::new(
                    TrainNetwork::new(cand_tx.clone(), cand_type),
                    NeuralConfig::default(),
                )
            };

            let mut black = if index % 2 == 0 {
                NeuralMcts::new(
                    TrainNetwork::new(cand_tx.clone(), cand_type),
                    NeuralConfig::default(),
                )
            } else {
                NeuralMcts::new(
                    TrainNetwork::new(base_tx.clone(), base_type),
                    NeuralConfig::default(),
                )
            };

            let mut game = create_game();

            let result = play_model(&mut black, &mut white, &mut game, iterations);

            (index, result)
        })
        .collect();

    drop(cand_tx);
    drop(base_tx);
    cand_handle.join().unwrap();
    base_handle.join().unwrap();

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
