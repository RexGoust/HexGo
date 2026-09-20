use std::sync::Arc;

use bevy::prelude::*;

use crate::{
    ai::{self, AiConfig, AiState, NeuralNetworkResource, burn_neural_network::BurnNeuralNetwork},
    game::{board::VertexId, player::Player::Black, state::GameStatus},
    session::{GameMode, GameSession, SessionCommand, SessionError},
    worker::Worker,
};

mod board;
pub mod i18n;
mod input;
mod layout;
mod materials;
pub mod menu;
pub mod state;
mod style;
mod sync;
mod ui;
pub use state::{AppState, InGameEntity};

const RULES_SCROLL_LINE: f32 = 28.0;

#[derive(SystemSet, Debug, Clone, Copy, PartialEq, Eq, Hash)]
enum GameSystemSet {
    Layout,
    Input,
    Sync,
    Style,
}

#[derive(Resource)]
pub struct WorkerResource(pub Worker);

pub struct ClientPlugin;

impl Plugin for ClientPlugin {
    fn build(&self, app: &mut App) {
        app.init_state::<AppState>()
            .add_plugins(menu::MenuPlugin)
            .insert_resource(ClearColor(board::BOARD_BACKGROUND))
            .insert_resource(SessionResource(GameSession::compact(GameMode::AI(Black))))
            .insert_resource(WorkerResource(Worker::new()))
            .insert_resource(NeuralNetworkResource {
                network: Arc::new(BurnNeuralNetwork::load()),
            })
            .init_resource::<AiConfig>()
            .init_resource::<AiState>()
            .init_resource::<UiState>()
            .init_resource::<i18n::CurrentLanguage>()
            .init_resource::<i18n::I18nStore>()
            .add_systems(Update, i18n::sync_i18n_static_texts);

        setup(app);

        app.configure_sets(
            Update,
            (
                GameSystemSet::Layout,
                GameSystemSet::Input,
                GameSystemSet::Sync,
                GameSystemSet::Style,
            )
                .chain()
                .run_if(in_state(AppState::InGame)),
        );

        add_layout_system(app);
        add_input_system(app);
        add_sync_system(app);
        add_style_system(app);
    }
}

fn setup(app: &mut App) {
    app.add_systems(
        Startup,
        (setup_camera, materials::setup_stone_materials).chain(),
    );

    app.add_systems(
        OnEnter(AppState::InGame),
        (board::setup_board, ui::setup_ui).chain(),
    );

    app.add_systems(OnExit(AppState::InGame), cleanup_in_game);
}

fn cleanup_in_game(
    mut commands: Commands,
    query: Query<Entity, With<InGameEntity>>,
    mut ui_state: ResMut<UiState>,
    mut ai_state: ResMut<AiState>,
) {
    for entity in &query {
        commands.entity(entity).despawn();
    }
    *ui_state = UiState::default();
    *ai_state = AiState::default();
}

fn add_layout_system(app: &mut App) {
    app.add_systems(
        Update,
        (
            layout::layout_control_panel,
            layout::layout_mobile_content,
            layout::fit_board_to_window,
        )
            .in_set(GameSystemSet::Layout),
    );
}

fn add_input_system(app: &mut App) {
    app.add_systems(
        Update,
        (
            input::update_pointer_target,
            input::handle_pointer_place,
            input::handle_keyboard,
            input::handle_buttons,
            input::scroll_rules,
            ai::update_ai,
        )
            .in_set(GameSystemSet::Input)
            .chain(),
    );
}

fn add_sync_system(app: &mut App) {
    app.add_systems(
        Update,
        (
            sync::sync_game_mode,
            sync::sync_stones,
            sync::sync_preview,
            sync::sync_focus_marker,
            sync::sync_last_move_marker,
            sync::sync_current_player,
            sync::sync_pass_count,
            sync::sync_result,
            sync::sync_feedback,
            sync::sync_modal,
            sync::sync_rules_modal,
            sync::sync_result_modal,
        )
            .in_set(GameSystemSet::Sync),
    );
}

fn add_style_system(app: &mut App) {
    app.add_systems(Update, (style::style_buttons).in_set(GameSystemSet::Style));
}

#[derive(Resource)]
pub struct SessionResource(pub(crate) GameSession);

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
enum FocusTarget {
    #[default]
    Board,
    Pass,
    Resign,
    Restart,
    Rules,
    MainMenu,
}

impl FocusTarget {
    fn next(self, reverse: bool) -> Self {
        const ORDER: [FocusTarget; 6] = [
            FocusTarget::Board,
            FocusTarget::Pass,
            FocusTarget::Resign,
            FocusTarget::Restart,
            FocusTarget::Rules,
            FocusTarget::MainMenu,
        ];
        let index = ORDER
            .iter()
            .position(|candidate| *candidate == self)
            .unwrap();
        let offset = if reverse { ORDER.len() - 1 } else { 1 };
        ORDER[(index + offset) % ORDER.len()]
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ModalKind {
    Resign,
    Restart,
    Rules,
    Result,
}

#[derive(Resource, Default)]
pub(crate) struct UiState {
    hovered: Option<VertexId>,
    focused_vertex: Option<VertexId>,
    focus: FocusTarget,
    modal: Option<ModalKind>,
    feedback: String,
    pub(crate) feedback_key: Option<&'static str>,
    feedback_is_error: bool,
    result_modal_seen: bool,
}

pub fn setup_camera(mut commands: Commands) {
    commands.spawn(Camera2d);
}

pub(crate) fn can_do_game_action(session: &GameSession, ui: &UiState) -> bool {
    if ui.modal.is_some() {
        return false;
    }

    let current_player = session.current_player();

    match session.mode() {
        GameMode::Local => true,

        GameMode::Network(player) => player == current_player,

        GameMode::AI(player) => player == current_player,

        GameMode::SelfPlay => false,
    }
}

pub(crate) fn command_feedback_key(command: SessionCommand) -> &'static str {
    match command {
        SessionCommand::Place(_) => "feedback.placed",
        SessionCommand::Pass => "feedback.passed",
        SessionCommand::Resign => "feedback.resigned",
        SessionCommand::Restart => "feedback.restarted",
    }
}

pub(crate) fn error_key(error: SessionError) -> &'static str {
    match error {
        SessionError::InvalidVertex => "error.invalid_vertex",
        SessionError::Occupied => "error.occupied",
        SessionError::Suicide => "error.suicide",
        SessionError::Superko => "error.superko",
        SessionError::GameOver => "error.game_over",
    }
}

fn submit_command(session: &mut GameSession, ui: &mut UiState, command: SessionCommand) {
    if ui.modal.is_some() {
        return;
    }

    if !can_do_game_action(session, ui) && command != SessionCommand::Restart {
        return;
    }

    if command == SessionCommand::Restart && session.game().status() == GameStatus::Playing {
        return;
    }

    match session.submit(command) {
        Ok(()) => {
            ui.feedback_is_error = false;
            let key = command_feedback_key(command);
            ui.feedback_key = Some(key);
            ui.feedback = match key {
                "feedback.placed" => "落子成功".into(),
                "feedback.passed" => "已停着".into(),
                "feedback.resigned" => "对局因认输结束".into(),
                "feedback.restarted" => "已开始新对局".into(),
                _ => String::new(),
            };
        }
        Err(error) => {
            ui.feedback_is_error = true;
            let key = error_key(error);
            ui.feedback_key = Some(key);
            ui.feedback = error_message(error).into();
        }
    }
}

fn error_message(error: SessionError) -> &'static str {
    match error {
        SessionError::InvalidVertex => "该交点不在棋盘上",
        SessionError::Occupied => "该处已有棋子",
        SessionError::Suicide => "禁止自杀落子",
        SessionError::Superko => "该落子违反全局同形规则",
        SessionError::GameOver => "对局已经结束",
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn focus_cycle_is_reversible() {
        assert_eq!(FocusTarget::Board.next(false), FocusTarget::Pass);
        assert_eq!(FocusTarget::Board.next(true), FocusTarget::MainMenu);
        assert_eq!(FocusTarget::MainMenu.next(false), FocusTarget::Board);
    }

    #[test]
    fn every_session_error_has_user_feedback() {
        let errors = [
            SessionError::InvalidVertex,
            SessionError::Occupied,
            SessionError::Suicide,
            SessionError::Superko,
            SessionError::GameOver,
        ];

        assert!(
            errors
                .into_iter()
                .all(|error| !error_message(error).is_empty())
        );
    }
}
