#![allow(dead_code)]

use crate::game::player::Player;
#[derive(Debug, PartialEq, Eq, Clone, Copy, Hash)]
pub enum VertexState {
    Empty,
    Occupied(Player),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GameStatus {
    Playing,
    Finished(GameEndReason),
}
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GameEndReason {
    ConsecutivePasses,
    Resignation { resigned: Player, winner: Player },
    NoLegalMoves,
}
