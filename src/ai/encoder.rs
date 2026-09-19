use burn::{
    Tensor,
    tensor::{TensorData, backend::Backend},
};

use crate::{
    ai::model::gnn::FEATURE_DIM,
    game::{
        Game,
        board::{BoardGraph, VertexId},
        player::Player,
        state::VertexState,
    },
};

pub const MLP_INPUT_SIZE: usize = 88 * 3;
pub const VERTEX_COUNT: usize = 88;
/// Encodes the game state from the given player's perspective.
///
/// Each vertex is represented by three features:
/// - `[1, 0, 0]` if occupied by the player.
/// - `[0, 1, 0]` if occupied by the opponent.
/// - `[0, 0, 1]` if the vertex is empty.
///
/// Vertices are encoded in stable `VertexId` order, so the resulting
/// vector has `3 * vertex_count` elements.
pub fn encode_game_mlp(game: &Game, player: Player) -> Vec<f32> {
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

fn encode_vertex_gnn(
    game: &Game,
    player: Player,
    opponent: Player,
    vertex: VertexId,
) -> [f32; FEATURE_DIM] {
    let (is_player, is_opponent, is_empty) = match game.vertex_state(vertex) {
        Some(VertexState::Occupied(owner)) if owner == player => (1.0, 0.0, 0.0),
        Some(VertexState::Occupied(owner)) if owner == opponent => (0.0, 1.0, 0.0),
        Some(VertexState::Empty) => (0.0, 0.0, 1.0),
        Some(VertexState::Occupied(_)) | None => unreachable!(),
    };

    let liberty_count = game.liberty_count(vertex);
    let group_liberties = match liberty_count {
        Some(n) => (1.0 + n as f32).ln() / (1.0 + 16.0f32).ln(),
        None => 0.0,
    };

    let is_atari = matches!(liberty_count, Some(1));
    let is_my_atari = if is_player > 0.0 && is_atari {
        1.0
    } else {
        0.0
    };
    let is_opp_atari = if is_opponent > 0.0 && is_atari {
        1.0
    } else {
        0.0
    };

    let is_last_move = game.is_last_move(vertex) as i32 as f32;

    let max_deg = game.board().max_degree();
    let neighbor_count: f32 = if max_deg > 0 {
        game.board()
            .get_neighbors(vertex)
            .map(|s| s.len())
            .unwrap_or(0) as f32
            / max_deg as f32
    } else {
        0.0
    };

    let moves = (1.0 + game.move_number() as f32).ln();

    let mut f = [0.0f32; FEATURE_DIM];

    f[0] = is_player;
    f[1] = is_opponent;
    f[2] = is_empty;
    f[3] = group_liberties;
    f[4] = is_my_atari;
    f[5] = is_opp_atari;
    f[6] = is_last_move;
    f[7] = neighbor_count;
    f[8] = moves;
    f
}

pub fn encode_game_gnn(game: &Game, player: Player) -> Vec<f32> {
    let vertex_count = game.board().vertex_count();
    let opponent = player.opponent();

    (0..vertex_count)
        .flat_map(|index| {
            let vertex = crate::game::board::VertexId::new(index);
            encode_vertex_gnn(game, player, opponent, vertex)
        })
        .collect()
}

pub fn encode_game_gnn_tensor<B: Backend>(
    game: &Game,
    player: Player,
    device: &B::Device,
) -> Tensor<B, 2> {
    let data = encode_game_gnn(game, player);
    let n = data.len() / FEATURE_DIM;

    Tensor::<B, 2>::from_data(TensorData::new(data, [n, FEATURE_DIM]), device)
}

pub fn adjacency_tensor<B: Backend>(graph: &BoardGraph, device: &B::Device) -> Tensor<B, 2> {
    let adj = graph.normalized_adjacency();
    let n = adj.len();
    let flat: Vec<f32> = adj.into_iter().flatten().collect();

    Tensor::<B, 2>::from_data(TensorData::new(flat, [n, n]), device)
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
        let input = encode_game_mlp(&game, Player::Black);

        assert_eq!(input.len(), 4 * 3);
    }

    #[test]
    fn encode_game_uses_player_perspective() {
        let mut game = test_game();

        game.play_move(VertexId::new(0)).unwrap();

        let input = encode_game_mlp(&game, Player::Black);
        assert_eq!(&input[0..3], &[1.0, 0.0, 0.0]);

        let input = encode_game_mlp(&game, Player::White);
        assert_eq!(&input[0..3], &[0.0, 1.0, 0.0]);
    }

    #[test]
    fn encode_game_gnn_has_expected_shape() {
        let game = test_game();
        let features = encode_game_gnn(&game, Player::Black);

        assert_eq!(features.len(), 4 * FEATURE_DIM);
        for val in &features {
            assert!(!val.is_nan());
        }
    }

    #[test]
    fn encode_game_gnn_handles_zero_max_degree_without_nan() {
        let board = BoardGraph::from_edges(1, []).unwrap();
        let game = Game::new(board);
        let features = encode_game_gnn(&game, Player::Black);

        assert_eq!(features.len(), FEATURE_DIM);
        for val in &features {
            assert!(!val.is_nan());
        }
        assert_eq!(features[6], 0.0);
    }

    #[test]
    fn encode_game_gnn_tensors_have_expected_dims() {
        use crate::ai::backend::InferBackend;

        let game = test_game();
        let device = Default::default();
        let x = encode_game_gnn_tensor::<InferBackend>(&game, Player::Black, &device);
        assert_eq!(x.dims(), [4, FEATURE_DIM]);

        let adj = adjacency_tensor::<InferBackend>(game.board(), &device);
        assert_eq!(adj.dims(), [4, 4]);
    }
}
