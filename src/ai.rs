use std::sync::Arc;

use bevy::{
    ecs::{
        resource::Resource,
        system::{Res, ResMut},
    },
    log::{error, info},
};

use crate::{
    ai::{
        burn_neural_network::BurnNeuralNetwork,
        neural_mcts::{NeuralConfig, NeuralMcts},
        search::Search,
    },
    client::{SessionResource, WorkerResource},
    game::{Game, action::Action, board::VertexId, state::GameStatus},
    session::{GameMode, SessionCommand},
    time::Timer,
    worker::Future,
};

pub mod backend;
pub mod burn_neural_network;
pub mod dummy_network;
pub mod encoder;
pub mod mcts;
pub mod model;
pub mod neural_mcts;
pub mod neural_network;
pub mod search;
pub mod search_result;

#[derive(Resource, Default)]
pub struct AiState {
    future: Option<Future<Option<VertexId>>>,
}

#[derive(Resource)]
pub struct NeuralNetworkResource {
    pub network: Arc<BurnNeuralNetwork>,
}

fn choose_ai_move(game: Game, network: Arc<BurnNeuralNetwork>) -> Option<VertexId> {
    //let mut mcts = Mcts::new();
    let mut mcts = NeuralMcts::new(network, NeuralConfig { add_noise: true });
    let t = Timer::now();
    let action = mcts.choose_action(&game, 1000);
    if game.status() == GameStatus::Playing {
        info!("MCTS took {:.2} ms", t.elapsed_ms());
    }

    let act = action?;

    match act {
        Action::Move(vertex) => Some(vertex),
        Action::Pass => None,
    }
}

pub(crate) fn update_ai(
    mut session: ResMut<SessionResource>,
    mut ai: ResMut<AiState>,
    worker: Res<WorkerResource>,
    network: Res<NeuralNetworkResource>,
) {
    if session.0.status() != GameStatus::Playing {
        ai.future = None;
        return;
    }

    match session.0.mode() {
        GameMode::AI(player) if session.0.current_player() == player => {
            ai.future = None;
            return;
        }
        GameMode::AI(player) if session.0.current_player() != player => {}
        GameMode::Local => {
            ai.future = None;
            return;
        }

        GameMode::Network(_) => {
            ai.future = None;
            return;
        }

        GameMode::SelfPlay => {}
        GameMode::AI(_) => {
            ai.future = None;
            return;
        }
    }

    if let Some(future) = &ai.future {
        let Some(vertex) = future.try_get() else {
            return;
        };

        ai.future = None;

        let result = match vertex {
            Some(vertex) => session.0.submit(SessionCommand::Place(vertex)),
            None => session.0.submit(SessionCommand::Pass),
        };

        if let Err(err) = result {
            error!("AI fail: {:?}", err);
        }
        return;
    }

    let game = session.0.game().clone();

    let network = network.network.clone();

    ai.future = Some(worker.0.execute(move || choose_ai_move(game, network)));
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ai::model::ModelConfig;
    use crate::game::player::Player::{Black, White};
    use crate::session::GameSession;
    use crate::worker::Worker;
    use bevy::prelude::*;
    use std::thread::sleep;
    use std::time::Duration;

    #[test]
    fn test_ai_plays_after_human_move() {
        let mut app = App::new();
        app.insert_resource(SessionResource(GameSession::compact(GameMode::AI(Black))))
            .insert_resource(WorkerResource(Worker::new()))
            .insert_resource(NeuralNetworkResource {
                network: Arc::new(BurnNeuralNetwork::load(ModelConfig::default())),
            })
            .init_resource::<AiState>()
            .add_systems(Update, update_ai);

        // Frame 0: Human turn (Black). AI should NOT move.
        app.update();
        assert_eq!(
            app.world().resource::<SessionResource>().0.current_player(),
            Black
        );
        assert!(app.world().resource::<AiState>().future.is_none());

        // Human plays a move as Black.
        app.world_mut()
            .resource_mut::<SessionResource>()
            .0
            .submit(SessionCommand::Place(VertexId::new(0)))
            .unwrap();
        assert_eq!(
            app.world().resource::<SessionResource>().0.current_player(),
            White
        );

        // Frame 1: AI turn (White). AI should start calculation.
        app.update();
        assert!(app.world().resource::<AiState>().future.is_some());

        // Wait for AI to finish calculation.
        for _ in 0..100 {
            sleep(Duration::from_millis(100));
            app.update();
            if app.world().resource::<SessionResource>().0.current_player() == Black {
                break;
            }
        }

        // AI should have made its move (White placed stone or passed), returning turn to Black.
        let session = app.world().resource::<SessionResource>();
        assert_eq!(session.0.current_player(), Black);
        assert!(session.0.last_move().is_some());
    }
}
