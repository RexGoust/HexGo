#![allow(dead_code)]

use burn::{Tensor, tensor::backend::Backend};
use hex_go::{
    ai::{
        encoder::{adjacency_tensor, encode_game_gnn, encode_game_mlp},
        model::HexGoModel,
        search::Search,
    },
    board_layout::BoardDefinition,
    game::{
        Game, GameResult,
        action::{ACTION_SIZE, Action, PASS_INDEX},
        player::Player,
    },
};

use crate::{
    active_game::{MAX_ACTIONS, create_game, policy_to_dense},
    dataset::TrainingSample,
    game_slot::GameSlot,
    model_type::ModelType,
    sampler::sample_action_by_temperature,
    train_network::forward_batch,
};
use rayon::prelude::*;

pub struct SelfPlayPosition {
    pub state: Vec<f32>,
    pub policy: Vec<f32>,
    pub player: Player,
}

pub fn play_game<S: Search>(
    model_type: ModelType,
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
        let state = match model_type {
            ModelType::Mlp => encode_game_mlp(game, player),
            ModelType::Gnn => encode_game_gnn(game, player),
        };

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
                game.pass_turn().unwrap();
                continue;
            }
        };
        let policy = policy_to_dense(&search.policy);

        let temperature = if actions < 30 { 1.0 } else { 0.0 };

        let action = sample_action_by_temperature(&search.policy, temperature);

        match action {
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

        if actions >= MAX_ACTIONS {
            println!("action over {} times, quitting game...", MAX_ACTIONS);
            let _ = game.pass_turn();
            let _ = game.pass_turn();
            break;
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

pub fn generate_samples_local<S, F>(
    model_type: ModelType,
    games: usize,
    iterations: usize,
    create_mcts: F,
) -> Vec<TrainingSample>
where
    S: Search,
    F: Fn() -> S + Sync,
{
    (0..games)
        .into_par_iter()
        .flat_map(|_| {
            let mut game = create_game();
            let mut mcts = create_mcts();
            //let mut mcts = NeuralMcts::new(DummyNetwork);
            play_game(model_type, &mut game, &mut mcts, iterations)
        })
        .collect()
}

pub fn generate_samples_batched<B>(
    model: HexGoModel<B>,
    device: &B::Device,
    model_type: ModelType,
    batch_size: usize,
    total_games: usize,
    iterations: usize,
) -> Vec<TrainingSample>
where
    B: Backend,
{
    let mut samples = Vec::new();
    let mut completed = 0;
    let initial_games = batch_size.min(total_games);
    let count_0 = initial_games.div_ceil(2);
    let count_1 = initial_games - count_0;

    let mut slots = [GameSlot::new(count_0), GameSlot::new(count_1)];
    let mut started = slots[0].games.len() + slots[1].games.len();

    let adj = match &model {
        HexGoModel::Gnn(_) => {
            let board = BoardDefinition::compact().graph().clone();
            adjacency_tensor::<B>(&board, device)
        }
        _ => Tensor::zeros([1, 1], device),
    };

    std::thread::scope(|s| {
        let (req_tx, req_rx) = std::sync::mpsc::sync_channel::<(usize, Vec<Vec<f32>>)>(2);
        let (resp_tx, resp_rx) = std::sync::mpsc::sync_channel(2);

        s.spawn(move || {
            while let Ok((id, inputs)) = req_rx.recv() {
                let refs: Vec<&Vec<f32>> = inputs.iter().collect();
                let res = forward_batch(&model, device, &adj, &refs);
                if resp_tx.send((id, res)).is_err() {
                    break;
                }
            }
        });

        let mut in_flight = 0;
        for (id, slot) in slots.iter_mut().enumerate() {
            let inputs = slot.select(model_type);
            if !inputs.is_empty() {
                req_tx.send((id, inputs)).unwrap();
                in_flight += 1;
            }
        }

        while in_flight > 0 {
            let (id, (policies, values)) = resp_rx.recv().unwrap();
            in_flight -= 1;

            let slot = &mut slots[id];

            slot.update(&policies, &values);
            slot.step += 1;
            if slot.step >= iterations {
                slot.commit_actions(
                    model_type,
                    &mut samples,
                    &mut completed,
                    &mut started,
                    total_games,
                );
            }

            if !slot.games.is_empty() && completed < total_games {
                loop {
                    let next_inputs = slot.select(model_type);
                    if !next_inputs.is_empty() {
                        req_tx.send((id, next_inputs)).unwrap();
                        in_flight += 1;
                        break;
                    }

                    slot.step += 1;
                    if slot.step >= iterations {
                        slot.commit_actions(
                            model_type,
                            &mut samples,
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

    samples
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
            neural_mcts::{NeuralConfig, NeuralMcts},
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
        let mut mcts = NeuralMcts::new(SelfPlayTestNetwork, NeuralConfig::default());

        let samples = play_game(ModelType::Mlp, &mut game, &mut mcts, TEST_ITERATIONS);

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
        let mut mcts = NeuralMcts::new(SelfPlayTestNetwork, NeuralConfig::default());

        let samples = play_game(ModelType::Mlp, &mut game, &mut mcts, TEST_ITERATIONS);

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
        let mut mcts = NeuralMcts::new(SelfPlayTestNetwork, NeuralConfig::default());

        let samples = play_game(ModelType::Mlp, &mut game, &mut mcts, TEST_ITERATIONS);

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
        let mut mcts = NeuralMcts::new(SelfPlayTestNetwork, NeuralConfig::default());

        let _samples = play_game(ModelType::Mlp, &mut game, &mut mcts, TEST_ITERATIONS);

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

    #[test]
    fn temperature_sampling_zero_is_strictly_deterministic() {
        let action_a = Action::Move(VertexId::new(0));
        let action_b = Action::Move(VertexId::new(1));
        let action_c = Action::Move(VertexId::new(2));

        let policy = vec![
            (action_a, 0.2),
            (action_b, 0.5), // Highest probability candidate.
            (action_c, 0.3),
        ];

        // Zero temperature (and values below the threshold) must deterministically
        // select the argmax action without stochastic variation across repeated calls.
        for _ in 0..100 {
            let chosen = sample_action_by_temperature(&policy, 0.0);
            assert_eq!(chosen, action_b);

            let chosen_near_zero = sample_action_by_temperature(&policy, 1e-5);
            assert_eq!(chosen_near_zero, action_b);
        }
    }

    #[test]
    fn temperature_sampling_standard_distribution() {
        let action_a = Action::Move(VertexId::new(0));
        let action_b = Action::Move(VertexId::new(1));

        let policy = vec![(action_a, 0.7), (action_b, 0.3)];

        let mut count_a = 0;
        let mut count_b = 0;
        let trials = 1000;

        for _ in 0..trials {
            match sample_action_by_temperature(&policy, 1.0) {
                a if a == action_a => count_a += 1,
                b if b == action_b => count_b += 1,
                _ => panic!("sampled unexpected action"),
            }
        }

        // Both candidates must be explored across independent trials.
        assert!(count_a > 0);
        assert!(count_b > 0);

        // The empirical distribution should reflect the 70%/30% ratio within normal variance.
        assert!(count_a > count_b);
        assert!((600..=800).contains(&count_a));
    }

    #[test]
    fn temperature_sampling_low_temp_sharpens_distribution() {
        let action_a = Action::Move(VertexId::new(0));
        let action_b = Action::Move(VertexId::new(1));

        let policy = vec![(action_a, 0.6), (action_b, 0.4)];

        // At low temperature (tau = 0.2, exponent = 5.0), selection probability
        // heavily concentrates on the dominant candidate (~88% expected).
        let mut count_a = 0;
        let trials = 500;

        for _ in 0..trials {
            if sample_action_by_temperature(&policy, 0.2) == action_a {
                count_a += 1;
            }
        }

        assert!(
            count_a > 400,
            "low temperature must heavily favor the dominant candidate"
        );
    }

    #[test]
    fn temperature_sampling_handles_single_action() {
        let only_action = Action::Pass;
        let policy = vec![(only_action, 1.0)];

        assert_eq!(sample_action_by_temperature(&policy, 0.0), only_action);
        assert_eq!(sample_action_by_temperature(&policy, 1.0), only_action);
        assert_eq!(sample_action_by_temperature(&policy, 2.0), only_action);
    }
}
