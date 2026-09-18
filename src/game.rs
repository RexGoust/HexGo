#![allow(dead_code)]

pub mod action;
pub mod board;
pub mod error;
pub mod player;
pub mod region;
pub mod score;
pub mod state;

use std::collections::HashSet;

use crate::game::{
    board::{BoardGraph, VertexId},
    error::*,
    player::Player,
    region::*,
    score::Score,
    state::{GameStatus::Playing, *},
};

use std::collections::VecDeque;

const DEFAULT_KOMI: f64 = 0.5;

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum GameResult {
    WinByScore { winner: Player, margin: f64 },
    WinByResignation { winner: Player },
    Draw,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct BoardSnapshot {
    occupancy: Vec<VertexState>,
}
#[derive(Debug, Clone, PartialEq)]
pub struct Game {
    board: BoardGraph,
    occupancy: Vec<VertexState>,
    current_player: Player,
    snapshot_history: HashSet<BoardSnapshot>,
    consecutive_passes: u8,
    status: GameStatus,
    komi: f64,
    last_move: Option<VertexId>,
    moves: usize,
}

impl Game {
    pub fn new(board: BoardGraph) -> Self {
        let count = board.vertex_count();

        let occupancy = vec![VertexState::Empty; count];
        let initial_snapshot = BoardSnapshot {
            occupancy: occupancy.clone(),
        };

        let mut snapshot_history = HashSet::new();
        snapshot_history.insert(initial_snapshot);

        Self {
            board,
            occupancy,
            current_player: Player::Black,
            snapshot_history,
            consecutive_passes: 0,
            status: Playing,
            komi: DEFAULT_KOMI,
            last_move: None,
            moves: 0,
        }
    }

    pub fn status(&self) -> GameStatus {
        self.status
    }

    pub fn board(&self) -> &BoardGraph {
        &self.board
    }

    pub fn current_player(&self) -> Player {
        self.current_player
    }

    pub fn current_snapshot(&self) -> BoardSnapshot {
        BoardSnapshot {
            occupancy: self.occupancy.clone(),
        }
    }

    pub fn komi(&self) -> f64 {
        self.komi
    }

    pub fn vertex_state(&self, id: VertexId) -> Option<VertexState> {
        if !self.board().contains(id) {
            return None;
        }

        Some(self.occupancy[id.index()])
    }

    pub fn group(&self, start: VertexId) -> Option<Vec<VertexId>> {
        Self::gruop_in(&self.occupancy, &self.board, start)
    }

    fn gruop_in(
        occupancy: &[VertexState],
        board: &BoardGraph,
        start: VertexId,
    ) -> Option<Vec<VertexId>> {
        if !board.contains(start) {
            return None;
        }

        let state = occupancy[start.index()];

        if state == VertexState::Empty {
            return None;
        }

        let mut group = Vec::new();
        let mut visited = HashSet::new();
        let mut queue = VecDeque::new();

        queue.push_back(start);
        visited.insert(start);

        while let Some(cur) = queue.pop_front() {
            group.push(cur);

            for &neighbor in board.get_neighbors(cur)? {
                if !board.contains(neighbor) {
                    continue;
                }

                if occupancy[neighbor.index()] != state {
                    continue;
                }

                if visited.insert(neighbor) {
                    queue.push_back(neighbor);
                }
            }
        }

        Some(group)
    }

    // Returns the liberties of the group containing the vertex.
    pub fn liberties(&self, start: VertexId) -> Option<Vec<VertexId>> {
        let group = self.group(start)?;

        let mut liberties = HashSet::new();

        for vertex in group {
            for &neighbor in self.board().get_neighbors(vertex)? {
                if self.vertex_state(neighbor) == Some(VertexState::Empty) {
                    liberties.insert(neighbor);
                }
            }
        }

        Some(liberties.into_iter().collect())
    }

    pub fn liberty_count(&self, start: VertexId) -> Option<usize> {
        Some(self.liberties(start)?.len())
    }

    pub fn has_liberty(&self, start: VertexId) -> bool {
        Self::has_liberty_in(&self.occupancy, &self.board, start)
    }

    fn has_liberty_in(occupancy: &[VertexState], board: &BoardGraph, start: VertexId) -> bool {
        if !board.contains(start) {
            return false;
        }

        let state = occupancy[start.index()];

        if state == VertexState::Empty {
            return false;
        }

        let mut visited = HashSet::new();
        let mut queue = VecDeque::new();

        queue.push_back(start);
        visited.insert(start);

        while let Some(cur) = queue.pop_front() {
            let Some(neighbors) = board.get_neighbors(cur) else {
                return false;
            };

            for &neighbor in neighbors {
                if !board.contains(neighbor) {
                    continue;
                }
                match occupancy[neighbor.index()] {
                    VertexState::Empty => return true,
                    neighbor_state if neighbor_state == state && visited.insert(neighbor) => {
                        queue.push_back(neighbor);
                    }
                    _ => {}
                }
            }
        }

        false
    }

    fn remove_group(&mut self, start: VertexId) -> Option<Vec<VertexId>> {
        let gruop = self.group(start)?;

        Self::remove_group_in(&mut self.occupancy, gruop)
    }

    fn remove_group_in(
        occupancy: &mut [VertexState],
        gruop: Vec<VertexId>,
    ) -> Option<Vec<VertexId>> {
        for vertex in gruop.iter() {
            occupancy[vertex.index()] = VertexState::Empty;
        }

        Some(gruop)
    }

    pub fn legal_moves(&self) -> Vec<VertexId> {
        (0..self.occupancy.len())
            .map(VertexId::new)
            .filter(|&vertex| self.is_legal_move(vertex))
            .collect()
    }

    pub fn has_legal_moves(&self) -> bool {
        for vertex in 0..self.occupancy.len() {
            if self.is_legal_move(VertexId::new(vertex)) {
                return true;
            }
        }
        false
    }

    pub fn is_legal_move(&self, vertex: VertexId) -> bool {
        if self.status != GameStatus::Playing {
            return false;
        }

        if self.vertex_state(vertex) != Some(VertexState::Empty) {
            return false;
        }

        let player = self.current_player();
        let opponent = player.opponent();

        let mut occupancy = self.occupancy.clone();

        occupancy[vertex.index()] = VertexState::Occupied(player);

        let neighbors = match self.board().get_neighbors(vertex) {
            Some(neighbors) => neighbors,
            None => return false,
        };

        for &neighbor in neighbors {
            if occupancy[neighbor.index()] != VertexState::Occupied(opponent) {
                continue;
            }

            if !Self::has_liberty_in(&occupancy, self.board(), neighbor)
                && let Some(group) = Self::gruop_in(&occupancy, self.board(), neighbor)
            {
                Self::remove_group_in(&mut occupancy, group);
            }
        }

        // The move is suicide if our own group has no liberty.
        if !Self::has_liberty_in(&occupancy, self.board(), vertex) {
            return false;
        }

        // Superko check comes last.
        let snapshot = BoardSnapshot { occupancy };

        !self.snapshot_history.contains(&snapshot)
    }

    pub fn is_last_move(&self, vertex: VertexId) -> bool {
        match self.last_move {
            Some(v) => vertex == v,
            None => false,
        }
    }

    pub fn move_number(&self) -> usize {
        self.moves
    }

    fn play_move_internal(&mut self, vertex: VertexId) -> Result<(), MoveError> {
        if self.status != GameStatus::Playing {
            return Err(MoveError::GameOver);
        }

        let Some(state) = self.vertex_state(vertex) else {
            return Err(MoveError::InvalidVertex);
        };

        if state != VertexState::Empty {
            return Err(MoveError::Occupied);
        }

        let player = self.current_player();

        let opponent = player.opponent();

        let neighbors = self
            .board()
            .get_neighbors(vertex)
            .ok_or(MoveError::InvalidVertex)?
            .to_vec();

        self.occupancy[vertex.index()] = VertexState::Occupied(player);

        let mut captured = Vec::new();
        let mut checked = HashSet::new();
        // Capture opponent groups with no liberties.
        for neighbor in neighbors {
            if self.vertex_state(neighbor) != Some(VertexState::Occupied(opponent)) {
                continue;
            }

            if checked.contains(&neighbor) {
                continue;
            }

            if self.has_liberty(neighbor) {
                continue;
            }

            let Some(group) = self.group(neighbor) else {
                continue;
            };

            for &stone in &group {
                checked.insert(stone);
            }

            for stone in group {
                self.occupancy[stone.index()] = VertexState::Empty;
                captured.push(stone);
            }
        }

        // Prevent suicide moves.
        if !self.has_liberty(vertex) {
            self.occupancy[vertex.index()] = VertexState::Empty;

            // Restore captured stones.
            for &stone in &captured {
                self.occupancy[stone.index()] = VertexState::Occupied(opponent);
            }

            return Err(MoveError::Suicide);
        }

        let snapshot = self.current_snapshot();

        if self.snapshot_history.contains(&snapshot) {
            self.occupancy[vertex.index()] = VertexState::Empty;

            for &stone in &captured {
                self.occupancy[stone.index()] = VertexState::Occupied(opponent);
            }

            return Err(MoveError::Superko);
        }

        self.current_player = opponent;
        self.consecutive_passes = 0;

        self.snapshot_history.insert(snapshot);
        self.last_move = Some(vertex);
        self.moves += 1;
        if !self.has_legal_moves() {
            self.status = GameStatus::Finished(GameEndReason::NoLegalMoves);
        }

        Ok(())
    }

    pub fn play_move(&mut self, vertex: VertexId) -> Result<(), MoveError> {
        self.play_move_internal(vertex)
    }

    pub fn pass_turn(&mut self) -> Result<(), PassError> {
        if self.status != GameStatus::Playing {
            return Err(PassError::GameOver);
        }

        self.consecutive_passes += 1;

        if self.consecutive_passes >= 2 {
            self.status = GameStatus::Finished(GameEndReason::ConsecutivePasses);
        }

        self.current_player = self.current_player.opponent();

        Ok(())
    }

    pub fn resign(&mut self) -> Result<(), ResignError> {
        if self.status != GameStatus::Playing {
            return Err(ResignError::GameOver);
        }

        let resigned = self.current_player;
        let winner = resigned.opponent();

        self.status = GameStatus::Finished(GameEndReason::Resignation { resigned, winner });

        Ok(())
    }

    pub fn consecutive_passes(&self) -> u8 {
        self.consecutive_passes
    }

    pub fn both_players_passed(&self) -> bool {
        self.consecutive_passes >= 2
    }

    pub fn empty_region(&self, start: VertexId) -> Option<EmptyRegion> {
        if self.vertex_state(start)? != VertexState::Empty {
            return None;
        }

        let mut vertices = Vec::new();
        let mut bordering_players = HashSet::new();

        let mut visited = HashSet::new();
        let mut queue = VecDeque::new();

        visited.insert(start);
        queue.push_back(start);

        while let Some(current) = queue.pop_front() {
            vertices.push(current);

            for &neighbor in self.board().get_neighbors(current)? {
                match self.vertex_state(neighbor)? {
                    VertexState::Empty => {
                        if visited.insert(neighbor) {
                            queue.push_back(neighbor);
                        }
                    }

                    VertexState::Occupied(player) => {
                        bordering_players.insert(player);
                    }
                }
            }
        }

        Some(EmptyRegion::new(vertices, bordering_players))
    }

    pub fn all_empty_regions(&self) -> Vec<EmptyRegion> {
        let mut regions = Vec::new();
        let mut visited = HashSet::new();

        for vertex in self.board().vertices() {
            if visited.contains(&vertex) {
                continue;
            }

            let Some(region) = self.empty_region(vertex) else {
                continue;
            };

            for &region_vertex in region.vertices() {
                visited.insert(region_vertex);
            }

            regions.push(region);
        }

        regions
    }

    pub fn score(&self) -> Score {
        let mut black = 0.0;
        let mut white = 0.0;

        for vertex in self.board().vertices() {
            match self.vertex_state(vertex) {
                Some(VertexState::Occupied(Player::Black)) => black += 1.0,
                Some(VertexState::Occupied(Player::White)) => white += 1.0,
                _ => {}
            }
        }

        for region in self.all_empty_regions() {
            let size = region.vertices().len() as f64;

            match region.territory() {
                Territory::Owned(Player::Black) => {
                    black += size;
                }

                Territory::Owned(Player::White) => {
                    white += size;
                }

                Territory::Neutral => {}
            }
        }

        Score::new(black, white + self.komi)
    }

    pub fn score_result(&self) -> GameResult {
        let score = self.score();

        if score.black() > score.white() {
            GameResult::WinByScore {
                winner: Player::Black,
                margin: score.black() - score.white(),
            }
        } else if score.black() < score.white() {
            GameResult::WinByScore {
                winner: Player::White,
                margin: score.white() - score.black(),
            }
        } else {
            GameResult::Draw
        }
    }

    pub fn result(&self) -> Option<GameResult> {
        match self.status {
            GameStatus::Playing => None,

            GameStatus::Finished(GameEndReason::ConsecutivePasses)
            | GameStatus::Finished(GameEndReason::NoLegalMoves) => Some(self.score_result()),

            GameStatus::Finished(GameEndReason::Resignation { winner, .. }) => {
                Some(GameResult::WinByResignation { winner })
            }
        }
    }
}

#[cfg(test)]
mod test {
    use crate::game::{board::VertexId, state::GameEndReason::ConsecutivePasses};

    use super::*;

    #[derive(Debug, Clone, PartialEq)]
    struct CompleteGameState {
        board: BoardGraph,
        snapshot: BoardSnapshot,
        current_player: Player,
        snapshot_history: HashSet<BoardSnapshot>,
        consecutive_passes: u8,
        status: GameStatus,
        komi: f64,
    }

    fn complete_state(game: &Game) -> CompleteGameState {
        CompleteGameState {
            board: game.board.clone(),
            snapshot: game.current_snapshot(),
            current_player: game.current_player,
            snapshot_history: game.snapshot_history.clone(),
            consecutive_passes: game.consecutive_passes,
            status: game.status,
            komi: game.komi,
        }
    }

    fn set_position(game: &mut Game, stones: &[(usize, Player)]) {
        game.occupancy.fill(VertexState::Empty);

        for &(index, player) in stones {
            game.occupancy[index] = VertexState::Occupied(player);
        }

        game.snapshot_history.clear();
        game.snapshot_history.insert(game.current_snapshot());
    }

    fn create_test_game() -> Game {
        // 0 --- 1 --- 2
        //       |
        //       3 --- 4
        let edges = vec![
            (VertexId::new(0), VertexId::new(1)),
            (VertexId::new(1), VertexId::new(2)),
            (VertexId::new(1), VertexId::new(3)),
            (VertexId::new(3), VertexId::new(4)),
        ];

        let board = BoardGraph::from_edges(5, edges).unwrap();

        Game::new(board)
    }

    #[test]
    fn test_new_game() {
        let game = create_test_game();

        assert_eq!(game.current_player(), Player::Black);

        for vertex in game.board().vertices() {
            assert_eq!(game.vertex_state(vertex), Some(VertexState::Empty));
        }

        assert_eq!(game.vertex_state(VertexId::new(5)), None);
    }

    #[test]
    fn test_group() {
        let mut game = create_test_game();

        // 0(B) --- 1(B) --- 2(W)
        //           |
        //          3(B) --- 4(Empty)

        game.occupancy[0] = VertexState::Occupied(Player::Black);
        game.occupancy[1] = VertexState::Occupied(Player::Black);
        game.occupancy[2] = VertexState::Occupied(Player::White);
        game.occupancy[3] = VertexState::Occupied(Player::Black);

        let group = game.group(VertexId::new(0)).unwrap();

        assert_eq!(group.len(), 3);
        assert!(group.contains(&VertexId::new(0)));
        assert!(group.contains(&VertexId::new(1)));
        assert!(group.contains(&VertexId::new(3)));

        assert!(!group.contains(&VertexId::new(2)));
        assert!(!group.contains(&VertexId::new(4)));
    }

    #[test]
    fn test_group_single_vertex() {
        let mut game = create_test_game();

        game.occupancy[2] = VertexState::Occupied(Player::White);

        let group = game.group(VertexId::new(2)).unwrap();

        assert_eq!(group, vec![VertexId::new(2)]);
    }

    #[test]
    fn test_group_empty_vertex() {
        let game = create_test_game();

        assert_eq!(game.group(VertexId::new(0)), None);
    }

    #[test]
    fn test_group_invalid_vertex() {
        let game = create_test_game();

        assert_eq!(game.group(VertexId::new(100)), None);
    }

    #[test]
    fn test_liberties() {
        let mut game = create_test_game();

        // 0(B) --- 1(B) --- 2(Empty)
        //           |
        //          3(B) --- 4(Empty)

        game.occupancy[0] = VertexState::Occupied(Player::Black);
        game.occupancy[1] = VertexState::Occupied(Player::Black);
        game.occupancy[3] = VertexState::Occupied(Player::Black);

        let liberties = game.liberties(VertexId::new(0)).unwrap();

        assert_eq!(liberties.len(), 2);
        assert!(liberties.contains(&VertexId::new(2)));
        assert!(liberties.contains(&VertexId::new(4)));

        assert_eq!(game.liberty_count(VertexId::new(0)), Some(2));

        assert!(game.has_liberty(VertexId::new(0)));
    }

    #[test]
    fn test_no_liberties() {
        let mut game = create_test_game();

        // full
        //
        // 0(B) --- 1(B) --- 2(W)
        //           |
        //          3(B) --- 4(W)

        game.occupancy[0] = VertexState::Occupied(Player::Black);
        game.occupancy[1] = VertexState::Occupied(Player::Black);
        game.occupancy[2] = VertexState::Occupied(Player::White);
        game.occupancy[3] = VertexState::Occupied(Player::Black);
        game.occupancy[4] = VertexState::Occupied(Player::White);

        let liberties = game.liberties(VertexId::new(0)).unwrap();

        assert!(liberties.is_empty());

        assert_eq!(game.liberty_count(VertexId::new(0)), Some(0));

        assert!(!game.has_liberty(VertexId::new(0)));
    }

    #[test]
    fn test_liberties_empty_vertex() {
        let game = create_test_game();

        assert_eq!(game.liberties(VertexId::new(0)), None);

        assert_eq!(game.liberty_count(VertexId::new(0)), None);

        assert!(!game.has_liberty(VertexId::new(0)));
    }

    #[test]
    fn test_remove_group() {
        let mut game = create_test_game();

        // 0(B) --- 1(B) --- 2(W)
        //           |
        //          3(B) --- 4(W)

        game.occupancy[0] = VertexState::Occupied(Player::Black);
        game.occupancy[1] = VertexState::Occupied(Player::Black);
        game.occupancy[2] = VertexState::Occupied(Player::White);
        game.occupancy[3] = VertexState::Occupied(Player::Black);
        game.occupancy[4] = VertexState::Occupied(Player::White);

        let removed = game.remove_group(VertexId::new(0)).unwrap();

        assert_eq!(removed.len(), 3);

        assert!(removed.contains(&VertexId::new(0)));
        assert!(removed.contains(&VertexId::new(1)));
        assert!(removed.contains(&VertexId::new(3)));

        assert_eq!(
            game.vertex_state(VertexId::new(0)),
            Some(VertexState::Empty)
        );

        assert_eq!(
            game.vertex_state(VertexId::new(1)),
            Some(VertexState::Empty)
        );

        assert_eq!(
            game.vertex_state(VertexId::new(3)),
            Some(VertexState::Empty)
        );

        // White stones should not be removed accidentally.
        assert_eq!(
            game.vertex_state(VertexId::new(2)),
            Some(VertexState::Occupied(Player::White))
        );

        assert_eq!(
            game.vertex_state(VertexId::new(4)),
            Some(VertexState::Occupied(Player::White))
        );
    }

    #[test]
    fn test_remove_single_vertex_group() {
        let mut game = create_test_game();

        game.occupancy[2] = VertexState::Occupied(Player::White);

        let removed = game.remove_group(VertexId::new(2)).unwrap();

        assert_eq!(removed, vec![VertexId::new(2)]);

        assert_eq!(
            game.vertex_state(VertexId::new(2)),
            Some(VertexState::Empty)
        );
    }

    #[test]
    fn test_remove_empty_vertex() {
        let mut game = create_test_game();

        let removed = game.remove_group(VertexId::new(0));

        assert_eq!(removed, None);

        assert_eq!(
            game.vertex_state(VertexId::new(0)),
            Some(VertexState::Empty)
        );
    }

    #[test]
    fn test_remove_invalid_vertex() {
        let mut game = create_test_game();

        let removed = game.remove_group(VertexId::new(100));

        assert_eq!(removed, None);
    }

    #[test]
    fn test_play_move() {
        let edges = vec![
            (VertexId::new(0), VertexId::new(1)),
            (VertexId::new(1), VertexId::new(2)),
        ];

        let board = BoardGraph::from_edges(3, edges).unwrap();
        let mut game = Game::new(board);

        assert_eq!(game.play_move(VertexId::new(1)), Ok(()));

        assert_eq!(
            game.vertex_state(VertexId::new(1)),
            Some(VertexState::Occupied(Player::Black))
        );

        // Switch to the other player after a valid move.
        assert_eq!(game.current_player(), Player::White);
    }

    #[test]
    fn test_play_move_on_occupied_vertex() {
        let edges = vec![(VertexId::new(0), VertexId::new(1))];

        let board = BoardGraph::from_edges(2, edges).unwrap();
        let mut game = Game::new(board);

        game.occupancy[0] = VertexState::Occupied(Player::White);

        assert_eq!(game.play_move(VertexId::new(0)), Err(MoveError::Occupied));

        // Player should not change after an invalid move.
        assert_eq!(game.current_player(), Player::Black);
    }

    #[test]
    fn test_play_move_invalid_vertex() {
        let board = BoardGraph::from_edges(2, vec![]).unwrap();
        let mut game = Game::new(board);

        assert_eq!(
            game.play_move(VertexId::new(100)),
            Err(MoveError::InvalidVertex)
        );

        assert_eq!(game.current_player(), Player::Black);
    }

    #[test]
    fn test_play_move_captures_opponent() {
        //       0(B)
        //        |
        // 2(B) - 1(W) - 3(?)
        //
        // Black plays at 3, leaving White at 1 with no liberties.

        let edges = vec![
            (VertexId::new(1), VertexId::new(0)),
            (VertexId::new(1), VertexId::new(2)),
            (VertexId::new(1), VertexId::new(3)),
        ];

        let board = BoardGraph::from_edges(4, edges).unwrap();
        let mut game = Game::new(board);

        game.occupancy[0] = VertexState::Occupied(Player::Black);
        game.occupancy[1] = VertexState::Occupied(Player::White);
        game.occupancy[2] = VertexState::Occupied(Player::Black);

        assert_eq!(game.play_move(VertexId::new(3)), Ok(()));

        // The white stone should be captured.
        assert_eq!(
            game.vertex_state(VertexId::new(1)),
            Some(VertexState::Empty)
        );

        assert_eq!(
            game.vertex_state(VertexId::new(3)),
            Some(VertexState::Occupied(Player::Black))
        );

        assert_eq!(game.current_player(), Player::White);
    }

    #[test]
    fn test_move_captures_multiple_enemy_groups_and_gains_liberties() {
        // 1(W)   2(W)
        //     \ /
        //      0(?) --- 3(W)
        //
        // The three white stones are separate groups whose only liberty is 0.
        let board = BoardGraph::from_edges(
            4,
            [
                (VertexId::new(0), VertexId::new(1)),
                (VertexId::new(0), VertexId::new(2)),
                (VertexId::new(0), VertexId::new(3)),
            ],
        )
        .unwrap();
        let mut game = Game::new(board);
        set_position(
            &mut game,
            &[(1, Player::White), (2, Player::White), (3, Player::White)],
        );

        assert_eq!(game.play_move(VertexId::new(0)), Ok(()));

        assert_eq!(
            game.vertex_state(VertexId::new(0)),
            Some(VertexState::Occupied(Player::Black))
        );
        for index in 1..=3 {
            assert_eq!(
                game.vertex_state(VertexId::new(index)),
                Some(VertexState::Empty)
            );
        }
        assert_eq!(game.liberty_count(VertexId::new(0)), Some(3));
        assert_eq!(game.current_player(), Player::White);
        assert_eq!(game.snapshot_history.len(), 2);
    }

    #[test]
    fn test_play_move_prevents_suicide() {
        //     4
        //     |
        //     0(W)
        //     |
        // 2(W)-1(?)-3(W)
        // |         |
        // 5         6
        //
        // Black plays at 1.
        // All neighboring white stones still have liberties,
        // so none are captured, while Black has no liberties.

        let edges = vec![
            (VertexId::new(1), VertexId::new(0)),
            (VertexId::new(1), VertexId::new(2)),
            (VertexId::new(1), VertexId::new(3)),
            (VertexId::new(0), VertexId::new(4)),
            (VertexId::new(2), VertexId::new(5)),
            (VertexId::new(3), VertexId::new(6)),
        ];

        let board = BoardGraph::from_edges(7, edges).unwrap();
        let mut game = Game::new(board);

        game.occupancy[0] = VertexState::Occupied(Player::White);
        game.occupancy[2] = VertexState::Occupied(Player::White);
        game.occupancy[3] = VertexState::Occupied(Player::White);

        let state_before_move = complete_state(&game);

        assert_eq!(game.play_move(VertexId::new(1)), Err(MoveError::Suicide));

        // The attempted move must be rolled back.
        assert_eq!(
            game.vertex_state(VertexId::new(1)),
            Some(VertexState::Empty)
        );

        // Player must not change after an illegal move.
        assert_eq!(game.current_player(), Player::Black);

        // White stones must remain untouched.
        for id in [0, 2, 3] {
            assert_eq!(
                game.vertex_state(VertexId::new(id)),
                Some(VertexState::Occupied(Player::White))
            );
        }

        assert_eq!(complete_state(&game), state_before_move);
    }

    #[test]
    fn test_initial_board_snapshot() {
        let game = create_test_game();

        let snapshot = game.current_snapshot();

        assert_eq!(
            snapshot.occupancy,
            vec![VertexState::Empty; game.board().vertex_count()]
        );
    }

    #[test]
    fn test_same_board_snapshots_are_equal() {
        let mut game = create_test_game();

        game.occupancy[0] = VertexState::Occupied(Player::Black);
        game.occupancy[2] = VertexState::Occupied(Player::White);

        let first = game.current_snapshot();
        let second = game.current_snapshot();

        assert_eq!(first, second);
    }

    #[test]
    fn test_different_board_snapshots_are_not_equal() {
        let mut game = create_test_game();

        let empty = game.current_snapshot();

        game.occupancy[0] = VertexState::Occupied(Player::Black);

        let occupied = game.current_snapshot();

        assert_ne!(empty, occupied);
    }

    #[test]
    fn test_initial_snapshot_is_recorded() {
        let game = create_test_game();

        let snapshot = game.current_snapshot();

        assert!(game.snapshot_history.contains(&snapshot));
        assert_eq!(game.snapshot_history.len(), 1);
    }

    #[test]
    fn test_recorded_snapshot_is_a_snapshot() {
        let mut game = create_test_game();

        let initial = game.current_snapshot();

        game.occupancy[0] = VertexState::Occupied(Player::Black);

        assert!(game.snapshot_history.contains(&initial));
        assert_ne!(game.current_snapshot(), initial);
    }

    #[test]
    fn test_play_move_prevents_superko() {
        let edges = vec![
            (VertexId::new(0), VertexId::new(1)),
            (VertexId::new(0), VertexId::new(2)),
            (VertexId::new(0), VertexId::new(3)),
            (VertexId::new(1), VertexId::new(4)),
            (VertexId::new(1), VertexId::new(5)),
            // Extra liberties for the surrounding stones.
            (VertexId::new(2), VertexId::new(6)),
            (VertexId::new(3), VertexId::new(7)),
            (VertexId::new(4), VertexId::new(8)),
            (VertexId::new(5), VertexId::new(9)),
            // Additional vertices so White retains legal moves and does not trigger NoLegalMoves.
            (VertexId::new(10), VertexId::new(11)),
        ];

        let board = BoardGraph::from_edges(12, edges).unwrap();
        let mut game = Game::new(board);

        game.occupancy[0] = VertexState::Occupied(Player::White);
        game.occupancy[2] = VertexState::Occupied(Player::Black);
        game.occupancy[3] = VertexState::Occupied(Player::Black);
        game.occupancy[4] = VertexState::Occupied(Player::White);
        game.occupancy[5] = VertexState::Occupied(Player::White);

        game.snapshot_history.clear();
        game.snapshot_history.insert(game.current_snapshot());

        // Black captures White at 0.
        assert_eq!(game.play_move(VertexId::new(1)), Ok(()));

        let state_after_capture = complete_state(&game);

        // White tries to recapture Black at 1,
        // which would recreate the previous board snapshot.
        assert_eq!(game.play_move(VertexId::new(0)), Err(MoveError::Superko));

        // Illegal move must leave the state unchanged.
        assert_eq!(complete_state(&game), state_after_capture);
    }

    #[test]
    fn test_actions_rejected_after_game_ends_with_no_legal_moves() {
        let edges = [
            (VertexId::new(0), VertexId::new(1)),
            (VertexId::new(0), VertexId::new(2)),
            (VertexId::new(0), VertexId::new(3)),
            (VertexId::new(1), VertexId::new(4)),
            (VertexId::new(1), VertexId::new(5)),
            (VertexId::new(2), VertexId::new(6)),
            (VertexId::new(3), VertexId::new(7)),
            (VertexId::new(4), VertexId::new(8)),
            (VertexId::new(5), VertexId::new(9)),
        ];

        let board = BoardGraph::from_edges(10, edges).unwrap();
        let mut game = Game::new(board);

        game.occupancy[0] = VertexState::Occupied(Player::White);
        game.occupancy[2] = VertexState::Occupied(Player::Black);
        game.occupancy[3] = VertexState::Occupied(Player::Black);
        game.occupancy[4] = VertexState::Occupied(Player::White);
        game.occupancy[5] = VertexState::Occupied(Player::White);

        game.snapshot_history.clear();
        game.snapshot_history.insert(game.current_snapshot());

        game.play_move(VertexId::new(1)).unwrap();
        assert!(matches!(
            game.status(),
            GameStatus::Finished(GameEndReason::NoLegalMoves)
        ));

        let finished_state = complete_state(&game);

        assert_eq!(game.play_move(VertexId::new(0)), Err(MoveError::GameOver));
        assert_eq!(complete_state(&game), finished_state);

        assert_eq!(game.pass_turn(), Err(PassError::GameOver));
        assert_eq!(complete_state(&game), finished_state);

        assert_eq!(game.resign(), Err(ResignError::GameOver));
        assert_eq!(complete_state(&game), finished_state);
    }

    #[test]
    fn test_occupied_and_invalid_vertex_rejections_leave_all_state_unchanged() {
        let board = BoardGraph::from_edges(3, [(VertexId::new(0), VertexId::new(1))]).unwrap();
        let mut game = Game::new(board);

        game.play_move(VertexId::new(0)).unwrap();
        game.pass_turn().unwrap();

        let state_before_occupied = complete_state(&game);
        assert_eq!(game.play_move(VertexId::new(0)), Err(MoveError::Occupied));
        assert_eq!(complete_state(&game), state_before_occupied);

        let state_before_invalid = complete_state(&game);
        assert_eq!(
            game.play_move(VertexId::new(99)),
            Err(MoveError::InvalidVertex)
        );
        assert_eq!(complete_state(&game), state_before_invalid);
    }

    #[test]
    fn test_legal_move_records_snapshot() {
        let board = BoardGraph::from_edges(2, vec![(VertexId::new(0), VertexId::new(1))]).unwrap();

        let mut game = Game::new(board);

        let history_len = game.snapshot_history.len();

        assert_eq!(game.play_move(VertexId::new(0)), Ok(()));

        assert_eq!(game.snapshot_history.len(), history_len + 1);
        assert!(game.snapshot_history.contains(&game.current_snapshot()));
    }

    #[test]
    fn test_pass_switches_player() {
        let mut game = create_test_game();

        assert!(game.pass_turn().is_ok());

        assert_eq!(game.current_player(), Player::White);
    }

    #[test]
    fn test_pass_increments_consecutive_passes() {
        let mut game = create_test_game();

        assert_eq!(game.consecutive_passes(), 0);

        assert!(game.pass_turn().is_ok());

        assert_eq!(game.consecutive_passes(), 1);

        assert!(game.pass_turn().is_ok());

        assert_eq!(game.consecutive_passes(), 2);
    }

    #[test]
    fn test_two_consecutive_passes() {
        let mut game = create_test_game();

        assert!(game.pass_turn().is_ok());

        assert!(!game.both_players_passed());

        assert!(game.pass_turn().is_ok());

        assert!(game.both_players_passed());
    }

    #[test]
    fn test_move_resets_consecutive_passes() {
        let edges = vec![(VertexId::new(0), VertexId::new(1))];

        let board = BoardGraph::from_edges(2, edges).unwrap();
        let mut game = Game::new(board);

        assert!(game.pass_turn().is_ok());

        assert_eq!(game.consecutive_passes(), 1);

        assert_eq!(game.play_move(VertexId::new(0)), Ok(()));

        assert_eq!(game.consecutive_passes(), 0);
    }

    #[test]
    fn test_pass_bypasses_superko_and_does_not_record_board_snapshot() {
        let mut game = create_test_game();

        let snapshot = game.current_snapshot();
        let history = game.snapshot_history.clone();

        assert!(game.pass_turn().is_ok());
        assert!(game.pass_turn().is_ok());

        assert_eq!(game.current_snapshot(), snapshot);
        assert_eq!(game.snapshot_history, history);
        assert_eq!(game.status(), GameStatus::Finished(ConsecutivePasses));
    }

    #[test]
    fn test_new_game_is_playing() {
        let game = create_test_game();

        assert_eq!(game.status(), GameStatus::Playing);
    }

    #[test]
    fn test_one_pass_does_not_finish_game() {
        let mut game = create_test_game();

        assert_eq!(game.pass_turn(), Ok(()));

        assert_eq!(game.status(), GameStatus::Playing);
        assert_eq!(game.consecutive_passes(), 1);
    }

    #[test]
    fn test_two_consecutive_passes_finish_game() {
        let mut game = create_test_game();

        assert_eq!(game.pass_turn(), Ok(()));
        assert_eq!(game.status(), GameStatus::Playing);

        assert_eq!(game.pass_turn(), Ok(()));
        assert_eq!(game.status(), GameStatus::Finished(ConsecutivePasses));
    }

    #[test]
    fn test_move_between_passes_does_not_finish_game() {
        let board = BoardGraph::from_edges(2, vec![(VertexId::new(0), VertexId::new(1))]).unwrap();

        let mut game = Game::new(board);

        game.pass_turn().unwrap();

        game.play_move(VertexId::new(0)).unwrap();

        game.pass_turn().unwrap();

        assert_eq!(game.consecutive_passes(), 1);
        assert_eq!(game.status(), GameStatus::Playing);
    }

    #[test]
    fn test_pass_move_pass_sequence_does_not_end_game() {
        let board = BoardGraph::from_edges(3, [(VertexId::new(0), VertexId::new(1))]).unwrap();
        let mut game = Game::new(board);

        game.pass_turn().unwrap();
        game.play_move(VertexId::new(0)).unwrap();
        game.pass_turn().unwrap();

        assert_eq!(game.status(), GameStatus::Playing);
        assert_eq!(game.consecutive_passes(), 1);
        assert_eq!(game.current_player(), Player::White);
        assert_eq!(game.snapshot_history.len(), 2);
    }

    #[test]
    fn test_move_is_rejected_after_game_finished() {
        let mut game = create_test_game();

        game.pass_turn().unwrap();
        game.pass_turn().unwrap();

        assert_eq!(
            game.status(),
            GameStatus::Finished(GameEndReason::ConsecutivePasses)
        );

        let snapshot = game.current_snapshot();
        let player = game.current_player();

        assert_eq!(game.play_move(VertexId::new(0)), Err(MoveError::GameOver));

        assert_eq!(game.current_snapshot(), snapshot);
        assert_eq!(game.current_player(), player);
    }

    #[test]
    fn test_pass_is_rejected_after_game_finished() {
        let mut game = create_test_game();

        game.pass_turn().unwrap();
        game.pass_turn().unwrap();

        let player = game.current_player();

        assert_eq!(game.pass_turn(), Err(PassError::GameOver));

        assert_eq!(game.current_player(), player);
        assert_eq!(game.consecutive_passes(), 2);
    }

    #[test]
    fn test_all_actions_are_rejected_atomically_after_game_finished() {
        let mut game = create_test_game();
        game.pass_turn().unwrap();
        game.pass_turn().unwrap();

        let finished_state = complete_state(&game);

        assert_eq!(game.play_move(VertexId::new(0)), Err(MoveError::GameOver));
        assert_eq!(complete_state(&game), finished_state);

        assert_eq!(game.pass_turn(), Err(PassError::GameOver));
        assert_eq!(complete_state(&game), finished_state);

        assert_eq!(game.resign(), Err(ResignError::GameOver));
        assert_eq!(complete_state(&game), finished_state);
    }

    #[test]
    fn test_black_resigns() {
        let mut game = create_test_game();

        assert_eq!(game.current_player(), Player::Black);

        assert_eq!(game.resign(), Ok(()));

        assert_eq!(
            game.status(),
            GameStatus::Finished(GameEndReason::Resignation {
                resigned: Player::Black,
                winner: Player::White,
            })
        );
    }

    #[test]
    fn test_white_resigns() {
        let mut game = create_test_game();

        game.play_move(VertexId::new(0)).unwrap();

        assert_eq!(game.current_player(), Player::White);

        game.resign().unwrap();

        assert_eq!(
            game.status(),
            GameStatus::Finished(GameEndReason::Resignation {
                resigned: Player::White,
                winner: Player::Black,
            })
        );
    }

    #[test]
    fn test_resign_does_not_change_board() {
        let mut game = create_test_game();

        let snapshot = game.current_snapshot();

        game.resign().unwrap();

        assert_eq!(game.current_snapshot(), snapshot);
    }

    #[test]
    fn test_cannot_resign_after_game_finished() {
        let mut game = create_test_game();

        game.resign().unwrap();

        let status = game.status();

        assert_eq!(game.resign(), Err(ResignError::GameOver));
        assert_eq!(game.status(), status);
    }

    #[test]
    fn test_cannot_resign_after_two_passes() {
        let mut game = create_test_game();

        game.pass_turn().unwrap();
        game.pass_turn().unwrap();

        assert_eq!(game.resign(), Err(ResignError::GameOver));
    }

    #[test]
    fn test_single_vertex_empty_region() {
        let board = BoardGraph::from_edges(2, vec![(VertexId::new(0), VertexId::new(1))]).unwrap();

        let mut game = Game::new(board);

        game.occupancy[1] = VertexState::Occupied(Player::Black);

        let region = game.empty_region(VertexId::new(0)).unwrap();

        assert_eq!(region.vertices().len(), 1);
        assert!(region.vertices().contains(&VertexId::new(0)));

        assert_eq!(region.bordering_players().len(), 1);
        assert!(region.bordering_players().contains(&Player::Black));
    }

    #[test]
    fn test_connected_empty_region() {
        let mut game = create_test_game();

        game.occupancy[0] = VertexState::Occupied(Player::Black);
        game.occupancy[4] = VertexState::Occupied(Player::Black);

        let region = game.empty_region(VertexId::new(1)).unwrap();

        assert_eq!(region.vertices().len(), 3);

        for id in [1, 2, 3] {
            assert!(region.vertices().contains(&VertexId::new(id)));
        }
        assert_eq!(region.bordering_players().len(), 1);
        assert!(region.bordering_players().contains(&Player::Black));
    }

    #[test]
    fn test_empty_region_with_mixed_border() {
        let mut game = create_test_game();

        game.occupancy[0] = VertexState::Occupied(Player::Black);
        game.occupancy[4] = VertexState::Occupied(Player::White);

        let region = game.empty_region(VertexId::new(1)).unwrap();

        assert_eq!(region.bordering_players().len(), 2);
        assert!(region.bordering_players().contains(&Player::Black));
        assert!(region.bordering_players().contains(&Player::White));
    }

    #[test]
    fn test_empty_region_on_occupied_vertex() {
        let mut game = create_test_game();

        game.occupancy[0] = VertexState::Occupied(Player::Black);

        assert_eq!(game.empty_region(VertexId::new(0)), None);
    }

    #[test]
    fn test_empty_region_invalid_vertex() {
        let game = create_test_game();

        assert_eq!(game.empty_region(VertexId::new(100)), None);
    }

    #[test]
    fn test_all_empty_board_has_one_region() {
        let edges = vec![
            (VertexId::new(0), VertexId::new(1)),
            (VertexId::new(1), VertexId::new(2)),
        ];

        let board = BoardGraph::from_edges(3, edges).unwrap();
        let game = Game::new(board);

        let regions = game.all_empty_regions();

        assert_eq!(regions.len(), 1);
        assert_eq!(regions[0].vertices().len(), 3);
    }

    #[test]
    fn test_multiple_empty_regions() {
        // 0(.) --- 1(.)
        //
        // 2(B)
        //
        // 3(.) --- 4(.)
        let edges = vec![
            (VertexId::new(0), VertexId::new(1)),
            (VertexId::new(1), VertexId::new(2)),
            (VertexId::new(2), VertexId::new(3)),
            (VertexId::new(3), VertexId::new(4)),
        ];

        let board = BoardGraph::from_edges(5, edges).unwrap();
        let mut game = Game::new(board);

        game.occupancy[2] = VertexState::Occupied(Player::Black);

        let regions = game.all_empty_regions();

        assert_eq!(regions.len(), 2);
    }

    #[test]
    fn test_empty_regions_exclude_occupied_vertices() {
        let mut game = create_test_game();

        game.occupancy[0] = VertexState::Occupied(Player::Black);
        game.occupancy[4] = VertexState::Occupied(Player::White);

        let regions = game.all_empty_regions();

        for region in &regions {
            assert!(!region.vertices().contains(&VertexId::new(0)));
            assert!(!region.vertices().contains(&VertexId::new(4)));
        }
    }

    #[test]
    fn test_each_empty_vertex_appears_once() {
        let mut game = create_test_game();

        game.occupancy[0] = VertexState::Occupied(Player::Black);

        let regions = game.all_empty_regions();

        let mut vertices = Vec::new();

        for region in &regions {
            vertices.extend_from_slice(region.vertices());
        }

        vertices.sort_unstable();

        let expected = vec![
            VertexId::new(1),
            VertexId::new(2),
            VertexId::new(3),
            VertexId::new(4),
        ];

        assert_eq!(vertices, expected);
    }

    #[test]
    fn test_full_board_has_no_empty_regions() {
        let mut game = create_test_game();

        for index in 0..game.board().vertex_count() {
            game.occupancy[index] = VertexState::Occupied(Player::Black);
        }

        assert!(game.all_empty_regions().is_empty());
    }

    #[test]
    fn test_score_counts_stones() {
        // 0(B) --- 1(W)

        let board = BoardGraph::from_edges(2, vec![(VertexId::new(0), VertexId::new(1))]).unwrap();

        let mut game = Game::new(board);

        game.occupancy[0] = VertexState::Occupied(Player::Black);
        game.occupancy[1] = VertexState::Occupied(Player::White);

        let score = game.score();

        assert_eq!(score.black(), 1.0);
        assert_eq!(score.white(), 1.0 + game.komi());
    }

    #[test]
    fn test_score_counts_black_territory() {
        // 0(B) --- 1(.) --- 2(B)

        let board = BoardGraph::from_edges(
            3,
            vec![
                (VertexId::new(0), VertexId::new(1)),
                (VertexId::new(1), VertexId::new(2)),
            ],
        )
        .unwrap();

        let mut game = Game::new(board);

        game.occupancy[0] = VertexState::Occupied(Player::Black);
        game.occupancy[2] = VertexState::Occupied(Player::Black);

        let score = game.score();

        // 2 black stones + vertex 1 territory
        assert_eq!(score.black(), 3.0);
        assert_eq!(score.white(), game.komi());
    }

    #[test]
    fn test_score_counts_white_territory() {
        // 0(W) --- 1(.) --- 2(W)

        let board = BoardGraph::from_edges(
            3,
            vec![
                (VertexId::new(0), VertexId::new(1)),
                (VertexId::new(1), VertexId::new(2)),
            ],
        )
        .unwrap();

        let mut game = Game::new(board);

        game.occupancy[0] = VertexState::Occupied(Player::White);
        game.occupancy[2] = VertexState::Occupied(Player::White);

        let score = game.score();

        assert_eq!(score.black(), 0.0);

        // 2 white stones + 1 territory + komi
        assert_eq!(score.white(), 3.0 + game.komi());
    }

    #[test]
    fn test_neutral_region_does_not_score() {
        // 0(B) --- 1(.) --- 2(W)

        let board = BoardGraph::from_edges(
            3,
            vec![
                (VertexId::new(0), VertexId::new(1)),
                (VertexId::new(1), VertexId::new(2)),
            ],
        )
        .unwrap();

        let mut game = Game::new(board);

        game.occupancy[0] = VertexState::Occupied(Player::Black);
        game.occupancy[2] = VertexState::Occupied(Player::White);

        let score = game.score();

        // vertex 1 touches both colors, so it is neutral.
        assert_eq!(score.black(), 1.0);
        assert_eq!(score.white(), 1.0 + game.komi());
    }

    #[test]
    fn test_score_combines_black_white_and_neutral_regions() {
        // 0(B)-1(.)-2(B)-3(W)-4(.)-5(W)-6(B)-7(.)-8(W)
        //      Black             White             Neutral
        let board = BoardGraph::from_edges(
            9,
            [
                (VertexId::new(0), VertexId::new(1)),
                (VertexId::new(1), VertexId::new(2)),
                (VertexId::new(2), VertexId::new(3)),
                (VertexId::new(3), VertexId::new(4)),
                (VertexId::new(4), VertexId::new(5)),
                (VertexId::new(5), VertexId::new(6)),
                (VertexId::new(6), VertexId::new(7)),
                (VertexId::new(7), VertexId::new(8)),
            ],
        )
        .unwrap();
        let mut game = Game::new(board);
        set_position(
            &mut game,
            &[
                (0, Player::Black),
                (2, Player::Black),
                (3, Player::White),
                (5, Player::White),
                (6, Player::Black),
                (8, Player::White),
            ],
        );

        assert_eq!(
            game.empty_region(VertexId::new(1)).unwrap().territory(),
            Territory::Owned(Player::Black)
        );
        assert_eq!(
            game.empty_region(VertexId::new(4)).unwrap().territory(),
            Territory::Owned(Player::White)
        );
        assert_eq!(
            game.empty_region(VertexId::new(7)).unwrap().territory(),
            Territory::Neutral
        );

        let score = game.score();
        assert_eq!(score.black(), 4.0);
        assert_eq!(score.white(), 4.0 + game.komi());
    }

    #[test]
    fn test_komi_is_added_only_to_white() {
        let board = BoardGraph::from_edges(1, vec![]).unwrap();

        let mut game = Game::new(board);

        game.occupancy[0] = VertexState::Occupied(Player::Black);

        let score = game.score();

        assert_eq!(score.black(), 1.0);
        assert_eq!(score.white(), game.komi());
    }

    #[test]
    fn test_result_is_none_while_playing() {
        let game = create_test_game();

        assert_eq!(game.result(), None);
    }

    #[test]
    fn test_result_black_wins_by_score() {
        let board = BoardGraph::from_edges(1, vec![]).unwrap();
        let mut game = Game::new(board);

        game.komi = 0.0;
        game.occupancy[0] = VertexState::Occupied(Player::Black);

        game.pass_turn().unwrap();
        game.pass_turn().unwrap();

        assert_eq!(
            game.result(),
            Some(GameResult::WinByScore {
                winner: Player::Black,
                margin: 1.0,
            })
        );
    }

    #[test]
    fn test_result_white_wins_by_score() {
        let board = BoardGraph::from_edges(1, vec![]).unwrap();
        let mut game = Game::new(board);

        game.komi = 0.0;
        game.occupancy[0] = VertexState::Occupied(Player::White);

        game.pass_turn().unwrap();
        game.pass_turn().unwrap();

        assert_eq!(
            game.result(),
            Some(GameResult::WinByScore {
                winner: Player::White,
                margin: 1.0,
            })
        );
    }

    #[test]
    fn test_result_draw() {
        let board = BoardGraph::from_edges(2, vec![(VertexId::new(0), VertexId::new(1))]).unwrap();

        let mut game = Game::new(board);

        game.komi = 0.0;
        game.occupancy[0] = VertexState::Occupied(Player::Black);
        game.occupancy[1] = VertexState::Occupied(Player::White);

        game.pass_turn().unwrap();
        game.pass_turn().unwrap();

        assert_eq!(game.result(), Some(GameResult::Draw));
    }

    #[test]
    fn test_komi_changes_the_final_winner() {
        fn finished_game_with_komi(komi: f64) -> Game {
            let board = BoardGraph::from_edges(
                3,
                [
                    (VertexId::new(0), VertexId::new(1)),
                    (VertexId::new(1), VertexId::new(2)),
                ],
            )
            .unwrap();
            let mut game = Game::new(board);
            game.komi = komi;
            set_position(
                &mut game,
                &[(0, Player::Black), (1, Player::Black), (2, Player::White)],
            );
            game.pass_turn().unwrap();
            game.pass_turn().unwrap();
            game
        }

        assert_eq!(
            finished_game_with_komi(0.5).result(),
            Some(GameResult::WinByScore {
                winner: Player::Black,
                margin: 0.5,
            })
        );
        assert_eq!(
            finished_game_with_komi(1.5).result(),
            Some(GameResult::WinByScore {
                winner: Player::White,
                margin: 0.5,
            })
        );
    }

    #[test]
    fn test_complete_game_from_first_move_through_scored_result() {
        // 0(B) --- 1(B) --- 2(.) --- 3(W) --- 4(W)
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

        assert_eq!(game.result(), None);
        assert_eq!(game.play_move(VertexId::new(0)), Ok(()));
        assert_eq!(game.play_move(VertexId::new(4)), Ok(()));
        assert_eq!(game.play_move(VertexId::new(1)), Ok(()));
        assert_eq!(game.play_move(VertexId::new(3)), Ok(()));
        assert_eq!(game.status(), GameStatus::Playing);
        assert_eq!(game.snapshot_history.len(), 5);

        assert_eq!(game.pass_turn(), Ok(()));
        assert_eq!(game.status(), GameStatus::Playing);
        assert_eq!(game.result(), None);

        assert_eq!(game.pass_turn(), Ok(()));
        assert_eq!(game.status(), GameStatus::Finished(ConsecutivePasses));
        assert_eq!(game.score(), Score::new(2.0, 2.5));
        assert_eq!(
            game.result(),
            Some(GameResult::WinByScore {
                winner: Player::White,
                margin: 0.5,
            })
        );
    }

    #[test]
    fn test_result_white_wins_by_resignation() {
        let mut game = create_test_game();

        // Black is the current player and resigns.
        game.resign().unwrap();

        assert_eq!(
            game.result(),
            Some(GameResult::WinByResignation {
                winner: Player::White,
            })
        );
    }

    #[test]
    fn test_result_black_wins_by_resignation() {
        let board = BoardGraph::from_edges(2, vec![(VertexId::new(0), VertexId::new(1))]).unwrap();

        let mut game = Game::new(board);

        // Black makes a legal move, so it becomes White's turn.
        game.play_move(VertexId::new(0)).unwrap();

        assert_eq!(game.current_player(), Player::White);

        game.resign().unwrap();

        assert_eq!(
            game.result(),
            Some(GameResult::WinByResignation {
                winner: Player::Black,
            })
        );
    }

    #[test]
    fn test_resignation_result_ignores_score() {
        let board = BoardGraph::from_edges(2, vec![(VertexId::new(0), VertexId::new(1))]).unwrap();

        let mut game = Game::new(board);

        game.occupancy[0] = VertexState::Occupied(Player::Black);
        game.occupancy[1] = VertexState::Occupied(Player::Black);

        // Black is massively ahead, but resigns anyway.
        game.resign().unwrap();

        assert_eq!(
            game.result(),
            Some(GameResult::WinByResignation {
                winner: Player::White,
            })
        );
    }
}
