use crate::{
    ai::neural_network::{Evaluation, NeuralNetwork},
    game::{Game, action::Action, player::Player},
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
