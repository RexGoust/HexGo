use crate::{
    ai::{
        neural_network::{Evaluation, NeuralNetwork},
        search::Search,
        search_result::SearchResult,
    },
    game::{Game, GameResult, action::Action, player::Player},
};
pub struct DummyNetwork;

impl NeuralNetwork for DummyNetwork {
    fn evaluate(&self, game: &Game, _player: Player) -> Evaluation {
        let legal_moves = game.legal_moves();

        let mut policy = Vec::with_capacity(legal_moves.len() + 1);

        if legal_moves.is_empty() {
            policy.push((Action::Pass, 1.0));
        } else {
            let prior = 1.0 / legal_moves.len() as f32;

            for action in legal_moves {
                policy.push((Action::Move(action), prior));
            }

            policy.push((Action::Pass, 0.0));
        }

        Evaluation { policy, value: 0.0 }
    }
}

struct NeuralMctsNode {
    parent: Option<usize>,
    children: Vec<usize>,

    action: Option<Action>,

    prior: f32,
    visits: u32,
    value_sum: f32,
}

pub struct NeuralMcts<N> {
    nodes: Vec<NeuralMctsNode>,
    network: N,
}

impl<N: NeuralNetwork> NeuralMcts<N> {
    pub fn new(network: N) -> Self {
        Self {
            nodes: Vec::new(),
            network,
        }
    }
    fn puct_score(
        parent_visits: u32,
        child_visits: u32,
        value_sum: f32,
        prior: f32,
        c_puct: f32,
    ) -> f32 {
        let q = if child_visits == 0 {
            0.0
        } else {
            -value_sum / child_visits as f32
        };

        let u = c_puct * prior * (parent_visits as f32).sqrt() / (1.0 + child_visits as f32);

        q + u
    }

    fn select(&self, root_game: &Game) -> (usize, Game) {
        let mut node = 0;
        let mut game = root_game.clone();

        loop {
            let current = &self.nodes[node];

            if current.children.is_empty() {
                return (node, game);
            }

            node = *current
                .children
                .iter()
                .max_by(|&&a, &&b| {
                    let a = &self.nodes[a];
                    let b = &self.nodes[b];

                    Self::puct_score(current.visits, a.visits, a.value_sum, a.prior, 1.414)
                        .total_cmp(&Self::puct_score(
                            current.visits,
                            b.visits,
                            b.value_sum,
                            b.prior,
                            1.414,
                        ))
                })
                .unwrap();

            match self.nodes[node].action.unwrap() {
                Action::Move(vertex) => game.play_move(vertex).unwrap(),
                Action::Pass => game.pass_turn().unwrap(),
            }
        }
    }

    fn expand(&mut self, node: usize, game: &Game, evaluation: &Evaluation) {
        for (action, prior) in evaluation.policy.iter() {
            let legal = match action {
                Action::Move(vertex) => game.is_legal_move(*vertex),
                Action::Pass => true,
            };
            if !legal {
                continue;
            }
            let child_id = self.nodes.len();

            self.nodes.push(NeuralMctsNode {
                parent: Some(node),
                children: Vec::new(),
                action: Some(*action),
                prior: *prior,
                visits: 0,
                value_sum: 0.0,
            });

            self.nodes[node].children.push(child_id);
        }
    }

    fn backpropagate(&mut self, mut node: usize, mut value: f32) {
        loop {
            self.nodes[node].visits += 1;
            self.nodes[node].value_sum += value;

            let Some(parent) = self.nodes[node].parent else {
                break;
            };

            value = -value;
            node = parent;
        }
    }

    fn terminal_value(result: GameResult, player: Player) -> f32 {
        match result {
            GameResult::WinByScore { winner, .. } | GameResult::WinByResignation { winner } => {
                if winner == player {
                    1.0
                } else {
                    -1.0
                }
            }

            GameResult::Draw => 0.0,
        }
    }
}

impl<N: NeuralNetwork> Search for NeuralMcts<N> {
    fn search(&mut self, game: &Game, iterations: usize) -> Option<SearchResult> {
        if game.result().is_some() {
            return None;
        }

        self.nodes.clear();

        self.nodes.push(NeuralMctsNode {
            parent: None,
            children: Vec::new(),
            action: None,
            prior: 1.0,
            visits: 0,
            value_sum: 0.0,
        });

        for _ in 0..iterations {
            let (node, game) = self.select(game);

            if let Some(result) = game.result() {
                let player = game.current_player().opponent();
                let value = Self::terminal_value(result, player);

                self.backpropagate(node, value);
                continue;
            }

            let player = game.current_player();
            let evaluation = self.network.evaluate(&game, player);

            self.expand(node, &game, &evaluation);
            self.backpropagate(node, evaluation.value);
        }

        let total_visits: u32 = self.nodes[0]
            .children
            .iter()
            .map(|&child| self.nodes[child].visits)
            .sum();

        let policy = self.nodes[0]
            .children
            .iter()
            .map(|&child| {
                let node = &self.nodes[child];

                (
                    node.action.unwrap(),
                    node.visits as f32 / total_visits as f32,
                )
            })
            .collect();

        let best_child = self.nodes[0]
            .children
            .iter()
            .copied()
            .max_by_key(|&child| self.nodes[child].visits)?;

        let action = self.nodes[best_child].action?;

        Some(SearchResult { action, policy })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::game::{
        Game,
        board::{BoardGraph, VertexId},
    };

    fn test_game() -> Game {
        let board = BoardGraph::from_edges(
            4,
            [
                (VertexId::new(0), VertexId::new(1)),
                (VertexId::new(1), VertexId::new(2)),
                (VertexId::new(2), VertexId::new(3)),
            ],
        )
        .unwrap();

        Game::new(board)
    }

    #[test]
    fn neural_mcts_returns_legal_action() {
        let game = test_game();
        let mut mcts = NeuralMcts::new(DummyNetwork);

        let action = mcts.choose_action(&game, 100).unwrap();

        match action {
            Action::Move(vertex) => assert!(game.is_legal_move(vertex)),
            Action::Pass => {}
        }
    }

    #[test]
    fn neural_mcts_root_visits_match_iterations() {
        let game = test_game();
        let mut mcts = NeuralMcts::new(DummyNetwork);

        mcts.choose_action(&game, 100);

        assert_eq!(mcts.nodes[0].visits, 100);
    }

    #[test]
    fn neural_mcts_root_children_are_legal() {
        let game = test_game();
        let mut mcts = NeuralMcts::new(DummyNetwork);

        mcts.choose_action(&game, 100);

        assert!(!mcts.nodes[0].children.is_empty());

        for &child in &mcts.nodes[0].children {
            match mcts.nodes[child].action.unwrap() {
                Action::Move(vertex) => assert!(game.is_legal_move(vertex)),
                Action::Pass => {}
            }
        }
    }

    #[test]
    fn neural_mcts_children_have_prior() {
        let game = test_game();
        let mut mcts = NeuralMcts::new(DummyNetwork);

        mcts.choose_action(&game, 1);

        let root = &mcts.nodes[0];

        for &child in &root.children {
            assert!(mcts.nodes[child].prior >= 0.0);
        }
    }

    #[test]
    fn neural_mcts_root_prior_sums_to_one() {
        let game = test_game();
        let mut mcts = NeuralMcts::new(DummyNetwork);

        mcts.choose_action(&game, 1);

        let sum: f32 = mcts.nodes[0]
            .children
            .iter()
            .map(|&child| mcts.nodes[child].prior)
            .sum();

        assert!((sum - 1.0).abs() < 1e-6);
    }

    struct BiasedNetwork;

    impl NeuralNetwork for BiasedNetwork {
        fn evaluate(&self, game: &Game, _player: Player) -> Evaluation {
            let legal_moves = game.legal_moves();
            let preferred = VertexId::new(0);

            let mut policy = Vec::with_capacity(legal_moves.len() + 1);

            if legal_moves.contains(&preferred) {
                let other_count = legal_moves.len() - 1;

                if other_count == 0 {
                    policy.push((Action::Move(preferred), 1.0));
                } else {
                    for vertex in legal_moves {
                        let prior = if vertex == preferred {
                            0.9
                        } else {
                            0.1 / other_count as f32
                        };

                        policy.push((Action::Move(vertex), prior));
                    }
                }
            } else {
                let prior = if legal_moves.is_empty() {
                    0.0
                } else {
                    1.0 / legal_moves.len() as f32
                };

                for vertex in legal_moves {
                    policy.push((Action::Move(vertex), prior));
                }
            }

            policy.push((Action::Pass, 0.0));

            Evaluation { policy, value: 0.0 }
        }
    }

    #[test]
    fn neural_mcts_follows_policy_prior() {
        let game = test_game();
        let mut mcts = NeuralMcts::new(BiasedNetwork);

        mcts.choose_action(&game, 1000);

        let preferred = VertexId::new(0);

        let preferred_child = mcts.nodes[0]
            .children
            .iter()
            .copied()
            .find(|&child| mcts.nodes[child].action == Some(Action::Move(preferred)))
            .unwrap();

        let preferred_visits = mcts.nodes[preferred_child].visits;

        let other_visits: u32 = mcts.nodes[0]
            .children
            .iter()
            .copied()
            .filter(|&child| mcts.nodes[child].action != Some(Action::Move(preferred)))
            .map(|child| mcts.nodes[child].visits)
            .sum();

        assert!(preferred_visits > other_visits);
    }

    #[test]
    fn neural_mcts_backpropagates_terminal_value() {
        let board = BoardGraph::from_edges(
            5,
            [
                (VertexId::new(0), VertexId::new(1)),
                (VertexId::new(1), VertexId::new(2)),
                (VertexId::new(2), VertexId::new(3)),
                (VertexId::new(3), VertexId::new(4)),
            ],
        )
        .unwrap();

        let mut game = Game::new(board);

        // 0(B) --- 1(B) --- 2(.) --- 3(W) --- 4(W)
        game.play_move(VertexId::new(0)).unwrap();
        game.play_move(VertexId::new(4)).unwrap();
        game.play_move(VertexId::new(1)).unwrap();
        game.play_move(VertexId::new(3)).unwrap();

        game.pass_turn().unwrap();
        game.pass_turn().unwrap();

        assert_eq!(
            game.result(),
            Some(GameResult::WinByScore {
                winner: Player::White,
                margin: 0.5,
            })
        );

        // The successful last pass switches current_player to Black.
        assert_eq!(game.current_player(), Player::Black);

        let value = NeuralMcts::<DummyNetwork>::terminal_value(
            game.result().unwrap(),
            game.current_player().opponent(),
        );

        assert_eq!(value, 1.0);
    }

    #[test]
    fn puct_prefers_higher_value() {
        let board = BoardGraph::from_edges(
            3,
            [
                (VertexId::new(0), VertexId::new(1)),
                (VertexId::new(1), VertexId::new(2)),
            ],
        )
        .unwrap();

        let game = Game::new(board);
        let mut mcts = NeuralMcts::new(DummyNetwork);

        mcts.nodes.push(NeuralMctsNode {
            parent: None,
            children: vec![1, 2],
            action: None,
            prior: 0.5,
            visits: 20,
            value_sum: 0.0,
        });

        mcts.nodes.push(NeuralMctsNode {
            parent: Some(0),
            children: Vec::new(),
            action: Some(Action::Move(VertexId::new(0))),
            prior: 0.5,
            visits: 10,
            value_sum: -8.0,
        });

        mcts.nodes.push(NeuralMctsNode {
            parent: Some(0),
            children: Vec::new(),
            action: Some(Action::Move(VertexId::new(1))),
            prior: 0.5,
            visits: 10,
            value_sum: 0.0,
        });

        let (node, _) = mcts.select(&game);

        assert_eq!(node, 1);
    }
}
