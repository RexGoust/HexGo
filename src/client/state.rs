use bevy::prelude::*;

/// High-level lifecycle states for HexGo application.
#[derive(States, Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum AppState {
    /// Active on startup and when browsing menus or configuring a match.
    #[default]
    MainMenu,
    /// Active during an ongoing local or AI game session.
    InGame,
}

/// Marker component attached to root in-game entities to facilitate cleanup on state exit.
#[derive(Component, Debug, Clone, Copy, PartialEq, Eq)]
pub struct InGameEntity;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_app_state_is_main_menu() {
        assert_eq!(AppState::default(), AppState::MainMenu);
    }

    #[test]
    fn state_transition_triggers_enter_and_exit_schedules() {
        let mut app = App::new();
        app.add_plugins(bevy::state::app::StatesPlugin);
        app.init_state::<AppState>();

        #[derive(Resource, Default)]
        struct TransitionLog {
            entered_ingame: usize,
            exited_ingame: usize,
        }

        app.init_resource::<TransitionLog>();

        app.add_systems(
            OnEnter(AppState::InGame),
            |mut log: ResMut<TransitionLog>| {
                log.entered_ingame += 1;
            },
        );

        app.add_systems(
            OnExit(AppState::InGame),
            |mut log: ResMut<TransitionLog>| {
                log.exited_ingame += 1;
            },
        );

        app.update();
        assert_eq!(app.world().resource::<TransitionLog>().entered_ingame, 0);

        app.world_mut()
            .resource_mut::<NextState<AppState>>()
            .set(AppState::InGame);
        app.update();
        assert_eq!(app.world().resource::<TransitionLog>().entered_ingame, 1);
        assert_eq!(app.world().resource::<TransitionLog>().exited_ingame, 0);

        app.world_mut()
            .resource_mut::<NextState<AppState>>()
            .set(AppState::MainMenu);
        app.update();
        assert_eq!(app.world().resource::<TransitionLog>().entered_ingame, 1);
        assert_eq!(app.world().resource::<TransitionLog>().exited_ingame, 1);
    }
}
