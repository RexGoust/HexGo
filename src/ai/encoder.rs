use crate::game::{Game, player::Player, state::VertexState};

pub const INPUT_SIZE: usize = 88 * 3;

/// Encodes the game state from the given player's perspective.
///
/// Each vertex is represented by three features:
/// - `[1, 0, 0]` if occupied by the player.
/// - `[0, 1, 0]` if occupied by the opponent.
/// - `[0, 0, 1]` if the vertex is empty.
///
/// Vertices are encoded in stable `VertexId` order, so the resulting
/// vector has `3 * vertex_count` elements.
pub fn encode_game(game: &Game, player: Player) -> Vec<f32> {
    let mut input = Vec::with_capacity(game.board().vertex_count() * 3);

    let opponent = player.opponent();

    for index in 0..game.board().vertex_count() {
        let vertex = crate::game::board::VertexId::new(index);

        match game.vertex_state(vertex) {
            Some(VertexState::Occupied(owner)) if owner == player => {
                input.extend_from_slice(&[1.0, 0.0, 0.0]);
            }

            Some(VertexState::Occupied(owner)) if owner == opponent => {
                input.extend_from_slice(&[0.0, 1.0, 0.0]);
            }

            Some(VertexState::Empty) => {
                input.extend_from_slice(&[0.0, 0.0, 1.0]);
            }

            Some(VertexState::Occupied(_)) => unreachable!(),

            None => unreachable!(),
        }
    }

    input
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
    fn encode_game_has_expected_size() {
        let game = test_game();
        let input = encode_game(&game, Player::Black);

        assert_eq!(input.len(), 4 * 3);
    }

    #[test]
    fn encode_game_uses_player_perspective() {
        let mut game = test_game();

        game.play_move(VertexId::new(0)).unwrap();

        let input = encode_game(&game, Player::Black);
        assert_eq!(&input[0..3], &[1.0, 0.0, 0.0]);

        let input = encode_game(&game, Player::White);
        assert_eq!(&input[0..3], &[0.0, 1.0, 0.0]);
    }
}
