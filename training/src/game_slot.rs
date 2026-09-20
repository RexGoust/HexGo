use hex_go::{
    ai::{
        encoder::{encode_game_gnn, encode_game_mlp},
        neural_mcts::StepState,
    },
    game::action::Action,
};
use rayon::iter::{IndexedParallelIterator, IntoParallelRefMutIterator, ParallelIterator};

use crate::{active_game::ActiveGame, dataset::TrainingSample, model_type::ModelType};

pub struct GameSlot {
    pub games: Vec<ActiveGame>,
    pub step: usize,
}

impl GameSlot {
    pub fn new(count: usize) -> Self {
        Self {
            games: (0..count).map(|_| ActiveGame::new()).collect(),
            step: 0,
        }
    }

    pub fn select(&mut self, model_type: ModelType) -> Vec<Vec<f32>> {
        self.games.par_iter_mut().for_each(|g| {
            g.state = g.mcts.step_select(&g.game);
            if let StepState::NeedsEvaluation { leaf_game, .. } = &g.state {
                g.input_buf = match model_type {
                    ModelType::Mlp => encode_game_mlp(leaf_game, leaf_game.current_player()),
                    ModelType::Gnn => encode_game_gnn(leaf_game, leaf_game.current_player()),
                };
            }
        });

        self.games
            .iter_mut()
            .filter(|g| matches!(&g.state, StepState::NeedsEvaluation { .. }))
            .map(|g| std::mem::take(&mut g.input_buf))
            .collect()
    }

    pub fn update(&mut self, policies: &[Vec<(Action, f32)>], values: &[f32]) {
        self.games
            .iter_mut()
            .filter(|g| matches!(&g.state, StepState::NeedsEvaluation { .. }))
            .collect::<Vec<_>>()
            .par_iter_mut()
            .enumerate()
            .for_each(|(i, g)| {
                if let StepState::NeedsEvaluation { node, leaf_game } = &g.state {
                    g.mcts
                        .step_update(*node, leaf_game, &policies[i], values[i]);
                }
            });
    }

    pub fn commit_actions(
        &mut self,
        model_type: ModelType,
        samples: &mut Vec<TrainingSample>,
        completed: &mut usize,
        started: &mut usize,
        total: usize,
    ) {
        self.step = 0;
        self.games
            .par_iter_mut()
            .for_each(|g| g.step_action(model_type));

        for g in self.games.iter_mut() {
            if g.is_finished() {
                samples.extend(g.finish_game());
                *completed += 1;
                if *started < total {
                    *started += 1;
                    *g = ActiveGame::new();
                }
            }
        }
        self.games.retain(|g| !g.is_finished());
    }
}
