use hex_go::{
    ai::{
        encoder::{encode_game_gnn, encode_game_mlp},
        neural_mcts::StepState,
    },
    game::{GameResult, action::Action},
};
use rayon::iter::{
    IndexedParallelIterator, IntoParallelIterator, IntoParallelRefMutIterator, ParallelIterator,
};

use crate::{evaluation::ActiveEvalGame, model_type::ModelType};
pub type ModelOutput = (Vec<Vec<(Action, f32)>>, Vec<f32>);
pub struct EvalSlot {
    pub games: Vec<ActiveEvalGame>,
    pub step: usize,
}

impl EvalSlot {
    pub fn new(indices: impl Iterator<Item = usize>) -> Self {
        Self {
            games: indices.map(ActiveEvalGame::new).collect(),
            step: 0,
        }
    }

    pub fn select(
        &mut self,
        cand_type: ModelType,
        base_type: ModelType,
    ) -> (Vec<Vec<f32>>, Vec<Vec<f32>>) {
        self.games.par_iter_mut().for_each(|g| {
            g.state = g.mcts.step_select(&g.game);
            if let StepState::NeedsEvaluation { leaf_game, .. } = &g.state {
                let model_type = if g.is_candidate_turn() {
                    cand_type
                } else {
                    base_type
                };
                g.input_buf = match model_type {
                    ModelType::Mlp => encode_game_mlp(leaf_game, leaf_game.current_player()),
                    ModelType::Gnn => encode_game_gnn(leaf_game, leaf_game.current_player()),
                };
            }
        });

        let mut cand_inputs = Vec::new();
        let mut base_inputs = Vec::new();

        for g in self.games.iter_mut() {
            if matches!(&g.state, StepState::NeedsEvaluation { .. }) {
                if g.is_candidate_turn() {
                    cand_inputs.push(std::mem::take(&mut g.input_buf));
                } else {
                    base_inputs.push(std::mem::take(&mut g.input_buf));
                }
            }
        }

        (cand_inputs, base_inputs)
    }

    pub fn update(&mut self, cand_res: &ModelOutput, base_res: &ModelOutput) {
        let mut cand_games: Vec<&mut ActiveEvalGame> = Vec::new();
        let mut base_games: Vec<&mut ActiveEvalGame> = Vec::new();

        for g in self.games.iter_mut() {
            if matches!(&g.state, StepState::NeedsEvaluation { .. }) {
                if g.is_candidate_turn() {
                    cand_games.push(g);
                } else {
                    base_games.push(g);
                }
            }
        }

        cand_games.into_par_iter().enumerate().for_each(|(i, g)| {
            if let StepState::NeedsEvaluation { node, leaf_game } = &g.state {
                g.mcts
                    .step_update(*node, leaf_game, &cand_res.0[i], cand_res.1[i]);
            }
        });

        base_games.into_par_iter().enumerate().for_each(|(i, g)| {
            if let StepState::NeedsEvaluation { node, leaf_game } = &g.state {
                g.mcts
                    .step_update(*node, leaf_game, &base_res.0[i], base_res.1[i]);
            }
        });
    }

    pub fn commit_actions(
        &mut self,
        results: &mut Vec<(usize, GameResult)>,
        completed: &mut usize,
        started: &mut usize,
        total: usize,
    ) {
        self.step = 0;
        self.games.par_iter_mut().for_each(|g| g.step_action());

        for g in self.games.iter_mut() {
            if g.is_finished() {
                results.push((g.index, g.game.result().unwrap()));
                *completed += 1;
                if *started < total {
                    let next_idx = *started;
                    *started += 1;
                    *g = ActiveEvalGame::new(next_idx);
                }
            }
        }
        self.games.retain(|g| !g.is_finished());
    }
}
