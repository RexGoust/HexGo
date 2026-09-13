#![allow(dead_code)]

use hex_go::{
    ai::{encoder::encode_game, mcts::Mcts, search::Search},
    board_layout::BoardDefinition,
    game::{
        Game, GameResult,
        action::{ACTION_SIZE, Action, PASS_INDEX},
        player::Player,
    },
};

use crate::dataset::TrainingSample;
use rayon::prelude::*;

pub struct SelfPlayPosition {
    pub state: Vec<f32>,
    pub policy: Vec<f32>,
    pub player: Player,
}

fn policy_to_dense(policy: &[(Action, f32)]) -> Vec<f32> {
    let mut result = vec![0.0; ACTION_SIZE];

    for &(action, probability) in policy {
        result[action.index()] = probability;
    }

    result
}

pub fn play_game<S: Search>(
    game: &mut Game,
    mcts: &mut S,
    iterations: usize,
) -> Vec<TrainingSample> {
    let mut positions = Vec::new();
    let mut moves = 0usize;
    let mut passes = 0usize;

    let mut actions = 0usize;
    while game.result().is_none() {
        let player = game.current_player();
        let state = encode_game(game, player);

        let search = match mcts.search(game, iterations) {
            Some(search) => search,
            None => {
                // No legal board move: pass.
                let mut policy = vec![0.0; ACTION_SIZE];
                policy[PASS_INDEX] = 1.0;

                positions.push(SelfPlayPosition {
                    state,
                    policy,
                    player,
                });
                // End the game immediately to avoid unnecessary passes.
                passes += 2;
                game.pass_turn().unwrap();
                game.pass_turn().unwrap();
                continue;
            }
        };
        let policy = policy_to_dense(&search.policy);

        match search.action {
            Action::Move(vertex) => {
                moves += 1;

                game.play_move(vertex).unwrap();
            }
            Action::Pass => {
                passes += 1;

                game.pass_turn().unwrap();
            }
        }
        actions += 1;

        positions.push(SelfPlayPosition {
            state,
            policy,
            player,
        });

        if actions.is_multiple_of(1000) {
            println!(
                "actions={}, legal_moves={}, result={:?}",
                actions,
                game.legal_moves().len(),
                game.result()
            );
        }
    }

    println!(
        "game finished: positions={}, moves={}, passes={}",
        positions.len(),
        moves,
        passes
    );

    let result = game.result().unwrap();

    to_training_samples(positions, result)
}

fn create_game() -> Game {
    let board = BoardDefinition::compact().graph().clone();

    Game::new(board)
}

pub fn generate_self_play_games(games: usize, iterations: usize) -> Vec<TrainingSample> {
    (0..games)
        .into_par_iter()
        .flat_map(|_| {
            let mut game = create_game();
            let mut mcts = Mcts::new();

            play_game(&mut game, &mut mcts, iterations)
        })
        .collect()
}

pub fn to_training_samples(
    positions: Vec<SelfPlayPosition>,
    result: GameResult,
) -> Vec<TrainingSample> {
    positions
        .into_iter()
        .map(|position| {
            let value = match result {
                GameResult::WinByScore { winner, .. } | GameResult::WinByResignation { winner } => {
                    if winner == position.player {
                        1.0
                    } else {
                        -1.0
                    }
                }
                GameResult::Draw => 0.0,
            };
            TrainingSample {
                state: position.state,
                policy: position.policy,
                value,
            }
        })
        .collect()
}
#[cfg(test)]
mod tests {
    use super::*;

    use hex_go::{
        ai::{
            neural_mcts::NeuralMcts,
            neural_network::{Evaluation, NeuralNetwork},
        },
        game::{
            Game, GameResult,
            board::{BoardGraph, VertexId},
            player::Player,
        },
    };

    const TEST_VERTEX_COUNT: usize = 4;
    const TEST_INPUT_SIZE: usize = TEST_VERTEX_COUNT * 3;

    const TEST_ITERATIONS: usize = 32;

    fn test_game() -> Game {
        let board = BoardGraph::from_edges(
            TEST_VERTEX_COUNT,
            [
                (VertexId::new(0), VertexId::new(1)),
                (VertexId::new(1), VertexId::new(2)),
                (VertexId::new(2), VertexId::new(3)),
            ],
        )
        .unwrap();

        Game::new(board)
    }

    /// Gives every legal move equal probability and only allows Pass
    /// when no board move is available.
    struct SelfPlayTestNetwork;

    impl NeuralNetwork for SelfPlayTestNetwork {
        fn evaluate(&self, game: &Game, _player: Player) -> Evaluation {
            let legal_moves = game.legal_moves();

            if legal_moves.is_empty() {
                return Evaluation {
                    policy: vec![(Action::Pass, 1.0)],
                    value: 0.0,
                };
            }

            let prior = 1.0 / legal_moves.len() as f32;

            let mut policy = legal_moves
                .into_iter()
                .map(|action| (Action::Move(action), prior))
                .collect::<Vec<_>>();

            // Pass is always part of the action space, but it is disabled
            // while legal board moves are available.
            policy.push((Action::Pass, 0.0));

            Evaluation { policy, value: 0.0 }
        }
    }

    #[test]
    fn self_play_generates_training_samples() {
        let mut game = test_game();
        let mut mcts = NeuralMcts::new(SelfPlayTestNetwork);

        let samples = play_game(&mut game, &mut mcts, TEST_ITERATIONS);

        assert!(!samples.is_empty());
        assert!(game.result().is_some());

        for sample in &samples {
            assert_eq!(sample.state.len(), TEST_INPUT_SIZE);
            assert_eq!(sample.policy.len(), ACTION_SIZE);

            let policy_sum: f32 = sample.policy.iter().sum();
            assert!((policy_sum - 1.0).abs() < 1e-5);

            assert!((-1.0..=1.0).contains(&sample.value));
        }
    }

    #[test]
    fn self_play_records_correct_player_perspective() {
        let mut game = test_game();
        let mut mcts = NeuralMcts::new(SelfPlayTestNetwork);

        let samples = play_game(&mut game, &mut mcts, TEST_ITERATIONS);

        assert!(!samples.is_empty());

        for sample in &samples {
            assert_eq!(sample.state.len(), TEST_INPUT_SIZE);
        }

        // Players alternate between training positions.
        for pair in samples.windows(2) {
            assert_ne!(pair[0].value, 2.0);
            assert_ne!(pair[1].value, 2.0);
        }
    }

    #[test]
    fn self_play_policy_contains_only_valid_actions() {
        let mut game = test_game();
        let mut mcts = NeuralMcts::new(SelfPlayTestNetwork);

        let samples = play_game(&mut game, &mut mcts, TEST_ITERATIONS);

        assert!(!samples.is_empty());

        for sample in &samples {
            let non_zero_actions = sample
                .policy
                .iter()
                .filter(|&&probability| probability > 0.0)
                .count();

            assert!(non_zero_actions > 0);
            assert!(non_zero_actions <= TEST_VERTEX_COUNT);
        }
    }

    #[test]
    fn self_play_ends_with_finished_game() {
        let mut game = test_game();
        let mut mcts = NeuralMcts::new(SelfPlayTestNetwork);

        let _samples = play_game(&mut game, &mut mcts, TEST_ITERATIONS);

        assert!(game.result().is_some());

        match game.result().unwrap() {
            GameResult::WinByScore { .. }
            | GameResult::WinByResignation { .. }
            | GameResult::Draw => {}
        }
    }

    #[test]
    fn to_training_samples_assigns_values_from_winner() {
        let positions = vec![
            SelfPlayPosition {
                state: vec![0.0; TEST_INPUT_SIZE],
                policy: vec![0.0; ACTION_SIZE],
                player: Player::Black,
            },
            SelfPlayPosition {
                state: vec![1.0; TEST_INPUT_SIZE],
                policy: vec![0.5; ACTION_SIZE],
                player: Player::White,
            },
        ];

        let result = GameResult::WinByScore {
            winner: Player::Black,
            margin: 1.0,
        };

        let samples = to_training_samples(positions, result);

        assert_eq!(samples.len(), 2);

        assert_eq!(samples[0].state, vec![0.0; TEST_INPUT_SIZE]);
        assert_eq!(samples[0].policy, vec![0.0; ACTION_SIZE]);
        assert_eq!(samples[0].value, 1.0);

        assert_eq!(samples[1].state, vec![1.0; TEST_INPUT_SIZE]);
        assert_eq!(samples[1].policy, vec![0.5; ACTION_SIZE]);
        assert_eq!(samples[1].value, -1.0);
    }

    #[test]
    fn to_training_samples_assigns_zero_for_draw() {
        let positions = vec![
            SelfPlayPosition {
                state: vec![0.0; TEST_INPUT_SIZE],
                policy: vec![0.0; ACTION_SIZE],
                player: Player::Black,
            },
            SelfPlayPosition {
                state: vec![0.0; TEST_INPUT_SIZE],
                policy: vec![0.0; ACTION_SIZE],
                player: Player::White,
            },
        ];

        let samples = to_training_samples(positions, GameResult::Draw);

        assert_eq!(samples.len(), 2);
        assert!(samples.iter().all(|sample| sample.value == 0.0));
    }
}
