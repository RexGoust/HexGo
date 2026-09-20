use crate::client::ui::RulesScroll;
use crate::client::{RULES_SCROLL_LINE, can_do_game_action, error_key, layout};
use crate::game::state::GameStatus;
use crate::session::{GameMode, GameSession, SessionError};
use crate::{
    board_layout::BoardDefinition,
    client::{FocusTarget, ModalKind, SessionResource, UiState, board::BoardRoot, submit_command},
    game::board::VertexId,
    session::SessionCommand,
};
use bevy::input::mouse::{MouseScrollUnit, MouseWheel};
use bevy::input::touch::TouchPhase;
use bevy::{prelude::*, window::PrimaryWindow};

use crate::client::state::AppState;

pub(super) const HIT_RADIUS: f32 = 0.46;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Component)]
pub enum ButtonAction {
    Pass,
    Resign,
    Restart,
    Rules,
    Confirm,
    Cancel,
    CloseRules,
    MainMenu,
    CloseResult,
    RestartDirect,
}

pub fn vertex_at_screen_position(
    position: Vec2,
    window: &Window,
    camera: (&Camera, &GlobalTransform),
    root: &Transform,
    definition: &BoardDefinition,
) -> Option<VertexId> {
    let window_size = Vec2::new(window.width(), window.height());
    if !layout::screen_position_is_on_board(window_size, position) {
        return None;
    }
    let world = camera.0.viewport_to_world_2d(camera.1, position).ok()?;
    let local = (world - root.translation.xy()) / root.scale.x;
    nearest_vertex(definition, local, HIT_RADIUS)
}

pub fn nearest_vertex(
    definition: &BoardDefinition,
    position: Vec2,
    radius: f32,
) -> Option<VertexId> {
    definition
        .positions()
        .iter()
        .enumerate()
        .filter_map(|(index, point)| {
            let distance = Vec2::from_array(*point).distance_squared(position);
            (distance <= radius * radius).then_some((distance, VertexId::new(index)))
        })
        .min_by(|left, right| left.0.total_cmp(&right.0))
        .map(|(_, vertex)| vertex)
}

pub(super) fn update_pointer_target(
    window: Single<&Window, With<PrimaryWindow>>,
    camera: Single<(&Camera, &GlobalTransform), With<Camera2d>>,
    root: Single<&Transform, With<BoardRoot>>,
    session: Res<SessionResource>,
    mut ui: ResMut<UiState>,
) {
    if ui.modal.is_some() {
        ui.hovered = None;
        return;
    }
    let Some(cursor) = window.cursor_position() else {
        ui.hovered = None;
        return;
    };
    ui.hovered = vertex_at_screen_position(cursor, &window, *camera, &root, session.0.definition());
}

pub(super) fn handle_pointer_place(
    mouse: Res<ButtonInput<MouseButton>>,
    touches: Res<Touches>,
    window: Single<&Window, With<PrimaryWindow>>,
    camera: Single<(&Camera, &GlobalTransform), With<Camera2d>>,
    root: Single<&Transform, With<BoardRoot>>,
    mut session: ResMut<SessionResource>,
    mut ui: ResMut<UiState>,
) {
    if ui.modal.is_some() {
        return;
    }

    let touched_vertex = touches.iter_just_pressed().find_map(|touch| {
        vertex_at_screen_position(
            touch.position(),
            &window,
            *camera,
            &root,
            session.0.definition(),
        )
    });
    let vertex = touched_vertex.or_else(|| {
        (mouse.just_pressed(MouseButton::Left) && !touches.any_just_pressed())
            .then_some(ui.hovered)
            .flatten()
    });

    if let Some(vertex) = vertex {
        ui.focus = FocusTarget::Board;
        ui.focused_vertex = Some(vertex);
        submit_command(&mut session.0, &mut ui, SessionCommand::Place(vertex));
    }
}

pub(super) fn handle_keyboard(
    keyboard: Res<ButtonInput<KeyCode>>,
    mut session: ResMut<SessionResource>,
    mut ui: ResMut<UiState>,
    mut next_state: ResMut<NextState<AppState>>,
) {
    if let Some(modal) = ui.modal {
        if keyboard.just_pressed(KeyCode::Escape) {
            ui.modal = None;
        } else if keyboard.just_pressed(KeyCode::Enter) || keyboard.just_pressed(KeyCode::Space) {
            confirm_modal(&mut session.0, &mut ui, modal);
        }
        return;
    }

    if keyboard.just_pressed(KeyCode::Tab) {
        ui.focus = ui
            .focus
            .next(keyboard.pressed(KeyCode::ShiftLeft) || keyboard.pressed(KeyCode::ShiftRight));
        if ui.focus == FocusTarget::Board && ui.focused_vertex.is_none() {
            ui.focused_vertex = Some(VertexId::new(0));
        }
    }

    if ui.focus == FocusTarget::Board {
        let direction = if keyboard.just_pressed(KeyCode::ArrowLeft) {
            Some(Vec2::NEG_X)
        } else if keyboard.just_pressed(KeyCode::ArrowRight) {
            Some(Vec2::X)
        } else if keyboard.just_pressed(KeyCode::ArrowUp) {
            Some(Vec2::Y)
        } else if keyboard.just_pressed(KeyCode::ArrowDown) {
            Some(Vec2::NEG_Y)
        } else {
            None
        };

        if let Some(direction) = direction {
            let current = ui.focused_vertex.unwrap_or(VertexId::new(0));
            ui.focused_vertex = navigate_vertex(session.0.definition(), current, direction);
        }

        if keyboard.just_pressed(KeyCode::Enter) || keyboard.just_pressed(KeyCode::Space) {
            let vertex = ui.focused_vertex.unwrap_or(VertexId::new(0));
            ui.focused_vertex = Some(vertex);
            submit_command(&mut session.0, &mut ui, SessionCommand::Place(vertex));
        }
    } else if keyboard.just_pressed(KeyCode::Enter) || keyboard.just_pressed(KeyCode::Space) {
        match ui.focus {
            FocusTarget::Pass => submit_command(&mut session.0, &mut ui, SessionCommand::Pass),
            FocusTarget::Resign => request_resign(&session.0, &mut ui),
            FocusTarget::Restart => ui.modal = Some(ModalKind::Restart),
            FocusTarget::Rules => open_rules(&mut ui),
            FocusTarget::MainMenu => next_state.set(AppState::MainMenu),
            FocusTarget::Board => {}
        }
    }
}
pub(super) fn navigate_vertex(
    definition: &BoardDefinition,
    current: VertexId,
    direction: Vec2,
) -> Option<VertexId> {
    let origin = Vec2::from_array(definition.position(current)?);
    definition
        .positions()
        .iter()
        .enumerate()
        .filter_map(|(index, point)| {
            let candidate = VertexId::new(index);
            if candidate == current {
                return None;
            }
            let delta = Vec2::from_array(*point) - origin;
            let forward = delta.dot(direction);
            if forward <= 0.01 {
                return None;
            }
            let sideways = delta.perp_dot(direction).abs();
            let score = sideways / forward * 100.0 + delta.length();
            Some((score, candidate))
        })
        .min_by(|left, right| left.0.total_cmp(&right.0))
        .map(|(_, vertex)| vertex)
        .or(Some(current))
}

pub(super) fn handle_buttons(
    interactions: Query<(&Interaction, &ButtonAction), Changed<Interaction>>,
    mut session: ResMut<SessionResource>,
    mut ui: ResMut<UiState>,
    mut next_state: ResMut<NextState<AppState>>,
) {
    for (interaction, action) in &interactions {
        if *interaction != Interaction::Pressed {
            continue;
        }
        match action {
            ButtonAction::Pass if can_do_game_action(&session.0, &ui) => {
                ui.focus = FocusTarget::Pass;
                submit_command(&mut session.0, &mut ui, SessionCommand::Pass);
            }
            ButtonAction::Resign if can_do_game_action(&session.0, &ui) => {
                ui.focus = FocusTarget::Resign;
                request_resign(&session.0, &mut ui);
            }
            ButtonAction::Restart
                if ui.modal.is_none() && !matches!(session.0.mode(), GameMode::Network(_)) =>
            {
                ui.focus = FocusTarget::Restart;
                ui.modal = Some(ModalKind::Restart);
            }
            ButtonAction::Rules if ui.modal.is_none() => {
                ui.focus = FocusTarget::Rules;
                open_rules(&mut ui);
            }
            ButtonAction::MainMenu if ui.modal.is_none() || ui.modal == Some(ModalKind::Result) => {
                ui.focus = FocusTarget::MainMenu;
                ui.modal = None;
                next_state.set(AppState::MainMenu);
            }
            ButtonAction::CloseResult => ui.modal = None,
            ButtonAction::RestartDirect => {
                ui.result_modal_seen = false;
                ui.modal = None;
                submit_command(&mut session.0, &mut ui, SessionCommand::Restart);
            }
            ButtonAction::Confirm => {
                if let Some(modal) = ui.modal {
                    confirm_modal(&mut session.0, &mut ui, modal);
                }
            }
            ButtonAction::Cancel => ui.modal = None,
            ButtonAction::CloseRules => ui.modal = None,
            _ => {}
        }
    }
}

fn open_rules(ui: &mut UiState) {
    ui.modal = Some(ModalKind::Rules);
}

fn request_resign(session: &GameSession, ui: &mut UiState) {
    if session.status() == GameStatus::Playing {
        ui.modal = Some(ModalKind::Resign);
    } else {
        ui.feedback_is_error = true;
        ui.feedback_key = Some(error_key(SessionError::GameOver));
    }
}

fn confirm_modal(session: &mut GameSession, ui: &mut UiState, modal: ModalKind) {
    ui.modal = None;
    match modal {
        ModalKind::Resign => submit_command(session, ui, SessionCommand::Resign),
        ModalKind::Restart => {
            ui.result_modal_seen = false;
            submit_command(session, ui, SessionCommand::Restart);
        }
        ModalKind::Rules | ModalKind::Result => {}
    }
}

pub(super) fn scroll_rules(
    mut mouse_wheel: MessageReader<MouseWheel>,
    mut touch_input: MessageReader<TouchInput>,
    keyboard: Res<ButtonInput<KeyCode>>,
    touches: Res<Touches>,
    ui: Res<UiState>,
    mut scroll: Single<(&mut ScrollPosition, &ComputedNode), With<RulesScroll>>,
) {
    if ui.modal != Some(ModalKind::Rules) {
        mouse_wheel.clear();
        touch_input.clear();
        return;
    }

    let wheel_delta = mouse_wheel.read().fold(0.0, |total, event| {
        let scale = match event.unit {
            MouseScrollUnit::Line => RULES_SCROLL_LINE,
            MouseScrollUnit::Pixel => 1.0,
        };
        total - event.y * scale
    });
    let keyboard_delta = if keyboard.just_pressed(KeyCode::ArrowDown) {
        RULES_SCROLL_LINE
    } else if keyboard.just_pressed(KeyCode::ArrowUp) {
        -RULES_SCROLL_LINE
    } else if keyboard.just_pressed(KeyCode::PageDown) {
        scroll.1.size().y * 0.8
    } else if keyboard.just_pressed(KeyCode::PageUp) {
        -scroll.1.size().y * 0.8
    } else {
        0.0
    };
    let has_touch_move = touch_input
        .read()
        .any(|event| event.phase == TouchPhase::Moved);
    let touch_delta = if has_touch_move {
        touches.iter().map(|touch| -touch.delta().y).sum::<f32>()
    } else {
        0.0
    };
    scroll.0.y = clamped_scroll_position(
        scroll.0.y,
        wheel_delta + keyboard_delta + touch_delta,
        scroll.1.content_size().y,
        scroll.1.size().y,
        scroll.1.inverse_scale_factor,
    );
}

fn clamped_scroll_position(
    current: f32,
    delta: f32,
    content_size: f32,
    visible_size: f32,
    inverse_scale_factor: f32,
) -> f32 {
    let max_offset = (content_size - visible_size).max(0.0) * inverse_scale_factor;
    (current + delta).clamp(0.0, max_offset)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        board_layout::BoardDefinition,
        client::input::{HIT_RADIUS, navigate_vertex, nearest_vertex},
        game::board::VertexId,
        session::GameMode,
    };
    use bevy::math::Vec2;

    #[test]
    fn upward_touch_motion_scrolls_rules_down_within_bounds() {
        assert_eq!(clamped_scroll_position(0.0, 48.0, 600.0, 300.0, 1.0), 48.0);
        assert_eq!(
            clamped_scroll_position(280.0, 48.0, 600.0, 300.0, 1.0),
            300.0
        );
        assert_eq!(clamped_scroll_position(20.0, -48.0, 600.0, 300.0, 1.0), 0.0);
    }

    #[test]
    fn pointer_hit_testing_uses_the_nearest_vertex() {
        let definition = BoardDefinition::compact();
        let target = VertexId::new(12);
        let position = Vec2::from_array(definition.position(target).unwrap());

        assert_eq!(
            nearest_vertex(&definition, position, HIT_RADIUS),
            Some(target)
        );
        assert_eq!(
            nearest_vertex(&definition, Vec2::splat(100.0), HIT_RADIUS),
            None
        );
    }

    #[test]
    fn directional_navigation_moves_and_stops_at_the_boundary() {
        let definition = BoardDefinition::compact();
        let start = VertexId::new(0);
        let right = navigate_vertex(&definition, start, Vec2::X).unwrap();

        assert_ne!(right, start);
        assert!(definition.position(right).unwrap()[0] > definition.position(start).unwrap()[0]);

        let left = navigate_vertex(&definition, start, Vec2::NEG_X).unwrap();
        assert_eq!(left, start);
    }

    #[test]
    fn rules_can_be_opened_without_changing_the_game() {
        let session = GameSession::compact(GameMode::Local);
        let current_player = session.current_player();
        let mut ui = UiState::default();

        open_rules(&mut ui);

        assert_eq!(ui.modal, Some(ModalKind::Rules));
        assert_eq!(session.current_player(), current_player);
    }

    #[test]
    fn resignation_confirmation_is_not_opened_after_game_over() {
        let mut session = GameSession::compact(GameMode::Local);
        session.submit(SessionCommand::Pass).unwrap();
        session.submit(SessionCommand::Pass).unwrap();
        let mut ui = UiState::default();

        request_resign(&session, &mut ui);

        assert_eq!(ui.modal, None);
        assert!(ui.feedback_is_error);
        assert_eq!(ui.feedback_key, Some(error_key(SessionError::GameOver)));
    }

    #[test]
    fn main_menu_button_triggers_state_transition() {
        let mut app = App::new();
        app.add_plugins(bevy::state::app::StatesPlugin);
        app.init_state::<AppState>();
        app.insert_resource(SessionResource(GameSession::compact(GameMode::Local)));
        app.init_resource::<UiState>();

        // Set state to InGame first
        app.world_mut()
            .resource_mut::<NextState<AppState>>()
            .set(AppState::InGame);
        app.update();

        app.add_systems(Update, handle_buttons);
        app.world_mut()
            .spawn((Interaction::Pressed, ButtonAction::MainMenu));
        app.update();

        let next_state = app.world().resource::<NextState<AppState>>();
        assert!(matches!(next_state, NextState::Pending(AppState::MainMenu)));
    }

    #[test]
    fn result_modal_buttons_handle_close_and_restart() {
        let mut app = App::new();
        app.add_plugins(bevy::state::app::StatesPlugin);
        app.init_state::<AppState>();
        let mut session = GameSession::compact(GameMode::Local);
        session.submit(SessionCommand::Pass).unwrap();
        session.submit(SessionCommand::Pass).unwrap();
        app.insert_resource(SessionResource(session));
        app.init_resource::<UiState>();
        app.world_mut().resource_mut::<UiState>().modal = Some(ModalKind::Result);
        app.world_mut().resource_mut::<UiState>().result_modal_seen = true;

        app.add_systems(Update, handle_buttons);

        // Test CloseResult
        let close_btn = app
            .world_mut()
            .spawn((Interaction::Pressed, ButtonAction::CloseResult))
            .id();
        app.update();
        assert_eq!(app.world().resource::<UiState>().modal, None);

        // Test RestartDirect
        app.world_mut().resource_mut::<UiState>().modal = Some(ModalKind::Result);
        app.world_mut().entity_mut(close_btn).despawn();
        app.world_mut()
            .spawn((Interaction::Pressed, ButtonAction::RestartDirect));
        app.update();

        let ui = app.world().resource::<UiState>();
        assert_eq!(ui.modal, None);
        assert!(!ui.result_modal_seen);
        let session = app.world().resource::<SessionResource>();
        assert_eq!(session.0.status(), GameStatus::Playing);
    }
}
