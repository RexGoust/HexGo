#![allow(dead_code)]
use std::process::abort;

use crate::{
    board_layout::BoardDefinition,
    game::{
        Game, GameResult,
        board::VertexId,
        error::{MoveError, PassError, ResignError},
        player::Player,
        region::Territory,
        state::{GameStatus, VertexState},
    },
};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SessionCommand {
    Place(VertexId),
    Pass,
    Resign,
    Restart,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SessionError {
    InvalidVertex,
    Occupied,
    Suicide,
    Superko,
    GameOver,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GameMode {
    Local,
    AI(Player),
    Network(Player),
    SelfPlay,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ScoreBreakdown {
    pub black_stones: usize,
    pub black_territory: usize,
    pub black_total: f64,
    pub white_stones: usize,
    pub white_territory: usize,
    pub komi: f64,
    pub white_total: f64,
}

pub struct GameSession {
    definition: BoardDefinition,
    game: Game,
    last_move: Option<VertexId>,
    mode: GameMode,
}

impl GameSession {
    pub fn new(definition: BoardDefinition, mode: GameMode) -> Self {
        let game = Game::new(definition.graph().clone());
        Self {
            definition,
            game,
            last_move: None,
            mode,
        }
    }

    pub fn compact(mode: GameMode) -> Self {
        if mode == GameMode::Network(Player::Black) || mode == GameMode::Network(Player::White) {
            abort();
        }

        Self::new(BoardDefinition::compact(), mode)
    }

    pub fn submit(&mut self, command: SessionCommand) -> Result<(), SessionError> {
        match command {
            SessionCommand::Place(vertex) => {
                self.game.play_move(vertex).map_err(SessionError::from)?;
                self.last_move = Some(vertex);
            }
            SessionCommand::Pass => {
                self.game.pass_turn().map_err(SessionError::from)?;
                self.last_move = None;
            }
            SessionCommand::Resign => {
                self.game.resign().map_err(SessionError::from)?;
            }
            SessionCommand::Restart => {
                self.game = Game::new(self.definition.graph().clone());
                self.last_move = None;
            }
        }

        Ok(())
    }

    pub fn definition(&self) -> &BoardDefinition {
        &self.definition
    }

    pub fn current_player(&self) -> Player {
        self.game.current_player()
    }

    pub fn status(&self) -> GameStatus {
        self.game.status()
    }

    pub fn consecutive_passes(&self) -> u8 {
        self.game.consecutive_passes()
    }

    pub fn vertex_state(&self, vertex: VertexId) -> Option<VertexState> {
        self.game.vertex_state(vertex)
    }

    pub fn last_move(&self) -> Option<VertexId> {
        self.last_move
    }

    pub fn result(&self) -> Option<GameResult> {
        self.game.result()
    }

    pub fn mode(&self) -> GameMode {
        self.mode
    }

    pub fn game(&self) -> &Game {
        &self.game
    }

    pub fn score_breakdown(&self) -> ScoreBreakdown {
        let mut black_stones = 0;
        let mut white_stones = 0;

        for vertex in self.game.board().vertices() {
            match self.game.vertex_state(vertex) {
                Some(VertexState::Occupied(Player::Black)) => black_stones += 1,
                Some(VertexState::Occupied(Player::White)) => white_stones += 1,
                _ => {}
            }
        }

        let mut black_territory = 0;
        let mut white_territory = 0;
        for region in self.game.all_empty_regions() {
            match region.territory() {
                Territory::Owned(Player::Black) => black_territory += region.vertices().len(),
                Territory::Owned(Player::White) => white_territory += region.vertices().len(),
                Territory::Neutral => {}
            }
        }

        let score = self.game.score();
        ScoreBreakdown {
            black_stones,
            black_territory,
            black_total: score.black(),
            white_stones,
            white_territory,
            komi: self.game.komi(),
            white_total: score.white(),
        }
    }
}

impl From<MoveError> for SessionError {
    fn from(error: MoveError) -> Self {
        match error {
            MoveError::InvalidVertex => Self::InvalidVertex,
            MoveError::Occupied => Self::Occupied,
            MoveError::Suicide => Self::Suicide,
            MoveError::Superko => Self::Superko,
            MoveError::GameOver => Self::GameOver,
        }
    }
}

impl From<PassError> for SessionError {
    fn from(_: PassError) -> Self {
        Self::GameOver
    }
}

impl From<ResignError> for SessionError {
    fn from(_: ResignError) -> Self {
        Self::GameOver
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn accepted_moves_and_passes_update_the_session() {
        let mut session = GameSession::compact(GameMode::Local);
        let vertex = VertexId::new(0);

        assert_eq!(session.submit(SessionCommand::Place(vertex)), Ok(()));
        assert_eq!(
            session.vertex_state(vertex),
            Some(VertexState::Occupied(Player::Black))
        );
        assert_eq!(session.current_player(), Player::White);
        assert_eq!(session.last_move(), Some(vertex));

        assert_eq!(session.submit(SessionCommand::Pass), Ok(()));
        assert_eq!(session.consecutive_passes(), 1);
        assert_eq!(session.last_move(), None);
    }

    #[test]
    fn rejected_moves_leave_the_session_view_unchanged() {
        let mut session = GameSession::compact(GameMode::Local);
        let vertex = VertexId::new(0);
        session.submit(SessionCommand::Place(vertex)).unwrap();
        let player = session.current_player();

        assert_eq!(
            session.submit(SessionCommand::Place(vertex)),
            Err(SessionError::Occupied)
        );
        assert_eq!(session.current_player(), player);
        assert_eq!(session.last_move(), Some(vertex));
    }

    #[test]
    fn two_passes_finish_and_restart_resets_the_game() {
        let mut session = GameSession::compact(GameMode::Local);
        session.submit(SessionCommand::Pass).unwrap();
        session.submit(SessionCommand::Pass).unwrap();

        assert!(matches!(session.status(), GameStatus::Finished(_)));
        assert!(session.result().is_some());
        assert_eq!(
            session.submit(SessionCommand::Place(VertexId::new(0))),
            Err(SessionError::GameOver)
        );

        session.submit(SessionCommand::Restart).unwrap();
        assert_eq!(session.status(), GameStatus::Playing);
        assert_eq!(session.current_player(), Player::Black);
        assert_eq!(session.consecutive_passes(), 0);
    }

    #[test]
    fn resignation_and_score_breakdown_are_exposed() {
        let mut session = GameSession::compact(GameMode::Local);
        session
            .submit(SessionCommand::Place(VertexId::new(0)))
            .unwrap();
        let score = session.score_breakdown();

        assert_eq!(score.black_stones, 1);
        assert_eq!(score.white_stones, 0);
        assert_eq!(score.komi, 0.5);

        session.submit(SessionCommand::Resign).unwrap();
        assert_eq!(
            session.result(),
            Some(GameResult::WinByResignation {
                winner: Player::Black
            })
        );
    }
}
