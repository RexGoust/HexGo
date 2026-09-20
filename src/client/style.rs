use crate::{
    client::{
        FocusTarget, SessionResource, UiState, can_do_game_action, input::ButtonAction, ui::ACCENT,
    },
    game::state::GameStatus,
    session::GameMode,
};
use bevy::prelude::*;

pub(super) fn style_buttons(
    session: Res<SessionResource>,
    ui: Res<UiState>,
    mut buttons: Query<(
        &Interaction,
        &ButtonAction,
        &mut BackgroundColor,
        &mut BorderColor,
    )>,
) {
    for (interaction, action, mut background, mut border) in &mut buttons {
        let disabled = match action {
            ButtonAction::Pass | ButtonAction::Resign => {
                session.0.status() != GameStatus::Playing || !can_do_game_action(&session.0, &ui)
            }
            ButtonAction::Restart => matches!(session.0.mode(), GameMode::Network(_)),
            _ => false,
        };
        let focused = match action {
            ButtonAction::Pass => ui.focus == FocusTarget::Pass,
            ButtonAction::Resign => ui.focus == FocusTarget::Resign,
            ButtonAction::Restart => ui.focus == FocusTarget::Restart,
            ButtonAction::Rules => ui.focus == FocusTarget::Rules,
            ButtonAction::MainMenu => ui.focus == FocusTarget::MainMenu,
            ButtonAction::Confirm | ButtonAction::Cancel | ButtonAction::CloseRules => false,
        };
        background.0 = if disabled {
            Color::srgb(0.23, 0.17, 0.12)
        } else {
            match interaction {
                Interaction::Pressed => Color::srgb(0.61, 0.42, 0.17),
                Interaction::Hovered => Color::srgb(0.46, 0.33, 0.20),
                Interaction::None => Color::srgb(0.35, 0.25, 0.16),
            }
        };
        *border = BorderColor::all(if focused { ACCENT } else { Color::NONE });
    }
}
