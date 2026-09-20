use bevy::prelude::*;

use super::{
    styles::{ACCENT, BUTTON_BG, BUTTON_HOVER, BUTTON_PRESSED, BUTTON_SELECTED},
    types::{MenuAction, MenuScreen, MenuSetupConfig},
    view::{AiSetupScreenRoot, DifficultyButton, MainScreenRoot, SideButton},
};
use crate::{
    ai::AiConfig,
    client::{SessionResource, i18n::CurrentLanguage, state::AppState},
    session::{GameMode, GameSession},
};

/// Handles user clicks on main menu buttons.
pub fn handle_menu_actions(
    interactions: Query<(&Interaction, &MenuAction), Changed<Interaction>>,
    mut config: ResMut<MenuSetupConfig>,
    mut next_state: ResMut<NextState<AppState>>,
    mut session: ResMut<SessionResource>,
    mut ai_config: ResMut<AiConfig>,
    mut current_lang: ResMut<CurrentLanguage>,
) {
    for (interaction, action) in &interactions {
        if *interaction != Interaction::Pressed {
            continue;
        }

        match action {
            MenuAction::PlayLocal => {
                session.0 = GameSession::compact(GameMode::Local);
                next_state.set(AppState::InGame);
            }
            MenuAction::OpenAiSetup => {
                config.screen = MenuScreen::AiSetup;
            }
            MenuAction::BackToMain => {
                config.screen = MenuScreen::Main;
            }
            MenuAction::SelectSide(player) => {
                config.player_color = *player;
            }
            MenuAction::SelectDifficulty(difficulty) => {
                config.difficulty = *difficulty;
            }
            MenuAction::StartAiGame => {
                ai_config.iterations = config.difficulty.iterations();
                session.0 = GameSession::compact(GameMode::AI(config.player_color));
                next_state.set(AppState::InGame);
            }
            MenuAction::ToggleLanguage => {
                current_lang.0 = current_lang.0.toggle();
            }
        }
    }
}

/// Updates container visibility based on the active MenuScreen.
pub fn sync_menu_screen_visibility(
    config: Res<MenuSetupConfig>,
    mut main_screen: Query<&mut Node, (With<MainScreenRoot>, Without<AiSetupScreenRoot>)>,
    mut ai_screen: Query<&mut Node, (With<AiSetupScreenRoot>, Without<MainScreenRoot>)>,
) {
    for mut node in &mut main_screen {
        node.display = if config.screen == MenuScreen::Main {
            Display::Flex
        } else {
            Display::None
        };
    }

    for mut node in &mut ai_screen {
        node.display = if config.screen == MenuScreen::AiSetup {
            Display::Flex
        } else {
            Display::None
        };
    }
}

/// Synchronizes selection highlights (borders) on side and difficulty buttons.
pub fn sync_menu_selection_styles(
    config: Res<MenuSetupConfig>,
    mut buttons: Query<(
        &mut BorderColor,
        Option<&SideButton>,
        Option<&DifficultyButton>,
    )>,
) {
    for (mut border, side, diff) in &mut buttons {
        let is_selected = match (side, diff) {
            (Some(side_btn), _) => side_btn.0 == config.player_color,
            (_, Some(diff_btn)) => diff_btn.0 == config.difficulty,
            _ => continue,
        };

        *border = BorderColor::all(if is_selected { ACCENT } else { Color::NONE });
    }
}

type MenuButtonQueryItem<'a> = (
    &'a Interaction,
    &'a MenuAction,
    &'a mut BackgroundColor,
    Option<&'a SideButton>,
    Option<&'a DifficultyButton>,
);

/// Provides hover and pressed background styling for menu buttons.
pub fn style_menu_buttons(config: Res<MenuSetupConfig>, mut buttons: Query<MenuButtonQueryItem>) {
    for (interaction, action, mut background, side, diff) in &mut buttons {
        let is_selected = match (side, diff) {
            (Some(side_btn), _) => side_btn.0 == config.player_color,
            (_, Some(diff_btn)) => diff_btn.0 == config.difficulty,
            _ => false,
        };

        if is_selected {
            background.0 = match interaction {
                Interaction::Pressed => BUTTON_PRESSED,
                Interaction::Hovered => BUTTON_HOVER,
                Interaction::None => BUTTON_SELECTED,
            };
        } else {
            let base_color = match action {
                MenuAction::StartAiGame => BUTTON_SELECTED,
                _ => BUTTON_BG,
            };
            background.0 = match interaction {
                Interaction::Pressed => BUTTON_PRESSED,
                Interaction::Hovered => BUTTON_HOVER,
                Interaction::None => base_color,
            };
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        client::{i18n::Language, menu::types::AiDifficulty},
        game::player::Player,
    };

    fn setup_test_app() -> App {
        let mut app = App::new();
        app.add_plugins(bevy::state::app::StatesPlugin);
        app.init_state::<AppState>();
        app.init_resource::<MenuSetupConfig>();
        app.init_resource::<AiConfig>();
        app.init_resource::<CurrentLanguage>();
        app.insert_resource(SessionResource(GameSession::compact(GameMode::Local)));
        app
    }

    #[test]
    fn toggle_language_action_toggles_current_language() {
        let mut app = setup_test_app();
        app.insert_resource(CurrentLanguage(Language::ZhCn));
        app.add_systems(Update, handle_menu_actions);

        assert_eq!(app.world().resource::<CurrentLanguage>().0, Language::ZhCn);

        app.world_mut()
            .spawn((Interaction::Pressed, MenuAction::ToggleLanguage));
        app.update();

        assert_eq!(app.world().resource::<CurrentLanguage>().0, Language::EnUs);

        app.world_mut()
            .spawn((Interaction::Pressed, MenuAction::ToggleLanguage));
        app.update();

        assert_eq!(app.world().resource::<CurrentLanguage>().0, Language::ZhCn);
    }

    #[test]
    fn play_local_action_starts_local_game_session() {
        let mut app = setup_test_app();
        app.add_systems(Update, handle_menu_actions);

        app.world_mut()
            .spawn((Interaction::Pressed, MenuAction::PlayLocal));
        app.update();

        let session = app.world().resource::<SessionResource>();
        assert_eq!(session.0.mode(), GameMode::Local);

        let next_state = app.world().resource::<NextState<AppState>>();
        assert!(matches!(next_state, NextState::Pending(AppState::InGame)));
    }

    #[test]
    fn start_ai_game_action_sets_correct_difficulty_and_color() {
        let mut app = setup_test_app();
        app.add_systems(Update, handle_menu_actions);

        // Select White side and Hard difficulty
        app.world_mut()
            .resource_mut::<MenuSetupConfig>()
            .player_color = Player::White;
        app.world_mut().resource_mut::<MenuSetupConfig>().difficulty = AiDifficulty::Hard;

        app.world_mut()
            .spawn((Interaction::Pressed, MenuAction::StartAiGame));
        app.update();

        let session = app.world().resource::<SessionResource>();
        assert_eq!(session.0.mode(), GameMode::AI(Player::White));

        let ai_config = app.world().resource::<AiConfig>();
        assert_eq!(ai_config.iterations, 1000);

        let next_state = app.world().resource::<NextState<AppState>>();
        assert!(matches!(next_state, NextState::Pending(AppState::InGame)));
    }

    #[test]
    fn option_actions_mutate_menu_config() {
        let mut app = setup_test_app();
        app.add_systems(Update, handle_menu_actions);

        app.world_mut()
            .spawn((Interaction::Pressed, MenuAction::OpenAiSetup));
        app.world_mut().spawn((
            Interaction::Pressed,
            MenuAction::SelectDifficulty(AiDifficulty::Simple),
        ));
        app.world_mut()
            .spawn((Interaction::Pressed, MenuAction::SelectSide(Player::White)));
        app.update();

        let config = app.world().resource::<MenuSetupConfig>();
        assert_eq!(config.screen, MenuScreen::AiSetup);
        assert_eq!(config.difficulty, AiDifficulty::Simple);
        assert_eq!(config.player_color, Player::White);
    }

    #[test]
    fn all_menu_interaction_systems_can_initialize_without_query_conflicts() {
        let mut app = App::new();
        app.add_plugins(bevy::state::app::StatesPlugin);
        app.init_state::<AppState>();
        app.insert_resource(SessionResource(GameSession::compact(GameMode::Local)));
        app.init_resource::<AiConfig>();
        app.init_resource::<MenuSetupConfig>();
        app.init_resource::<CurrentLanguage>();
        app.add_systems(
            Update,
            (
                handle_menu_actions,
                sync_menu_screen_visibility,
                sync_menu_selection_styles,
                style_menu_buttons,
            ),
        );
        app.update();
    }
}
