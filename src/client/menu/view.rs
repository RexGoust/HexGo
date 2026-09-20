use bevy::prelude::*;

use super::{
    styles::{
        self, ACCENT, BUTTON_BG, BUTTON_SELECTED, CARD_BG, MENU_BG, TEXT_MUTED, TEXT_PRIMARY,
    },
    types::{AiDifficulty, MenuAction, MenuEntity},
};
use crate::game::player::Player;

/// Marker component for the primary main screen button container.
#[derive(Component, Debug, Clone, Copy, PartialEq, Eq)]
pub struct MainScreenRoot;

/// Marker component for the AI configuration screen container.
#[derive(Component, Debug, Clone, Copy, PartialEq, Eq)]
pub struct AiSetupScreenRoot;

/// Marker component for side selection toggle buttons.
#[derive(Component, Debug, Clone, Copy, PartialEq, Eq)]
pub struct SideButton(pub Player);

/// Marker component for difficulty selection toggle buttons.
#[derive(Component, Debug, Clone, Copy, PartialEq, Eq)]
pub struct DifficultyButton(pub AiDifficulty);

/// Spawns the entire Main Menu UI tree.
pub fn spawn_main_menu(mut commands: Commands, asset_server: Res<AssetServer>) {
    let font = styles::load_menu_font(&asset_server);

    commands
        .spawn((
            MenuEntity,
            Node {
                width: percent(100),
                height: percent(100),
                align_items: AlignItems::Center,
                justify_content: JustifyContent::Center,
                flex_direction: FlexDirection::Column,
                ..default()
            },
            BackgroundColor(MENU_BG),
        ))
        .with_children(|root| {
            root.spawn((
                Node {
                    width: percent(90),
                    max_width: px(440),
                    padding: UiRect::all(px(32)),
                    flex_direction: FlexDirection::Column,
                    align_items: AlignItems::Center,
                    row_gap: px(24),
                    border_radius: BorderRadius::all(px(14)),
                    border: UiRect::all(px(1)),
                    ..default()
                },
                BorderColor::all(BUTTON_BG),
                BackgroundColor(CARD_BG),
            ))
            .with_children(|card| {
                spawn_title_header(card, &font);
                spawn_main_screen(card, &font);
                spawn_ai_setup_screen(card, &font);
            });
        });
}

/// Destroys all menu entities when leaving AppState::MainMenu.
pub fn cleanup_menu(mut commands: Commands, query: Query<Entity, With<MenuEntity>>) {
    for entity in &query {
        commands.entity(entity).despawn();
    }
}

fn spawn_title_header(card: &mut ChildSpawnerCommands, font: &Handle<Font>) {
    card.spawn((Node {
        flex_direction: FlexDirection::Column,
        align_items: AlignItems::Center,
        row_gap: px(6),
        margin: UiRect::bottom(px(8)),
        ..default()
    },))
        .with_children(|header| {
            header.spawn(styles::menu_text_bundle("HEXGO", font, 38.0, TEXT_PRIMARY));
            header.spawn(styles::menu_text_bundle(
                "六角围棋 · 策略对战",
                font,
                15.0,
                TEXT_MUTED,
            ));
        });
}

fn spawn_main_screen(card: &mut ChildSpawnerCommands, font: &Handle<Font>) {
    card.spawn((
        MainScreenRoot,
        Node {
            width: percent(100),
            flex_direction: FlexDirection::Column,
            row_gap: px(16),
            ..default()
        },
    ))
    .with_children(|screen| {
        spawn_primary_button(
            screen,
            MenuAction::OpenAiSetup,
            "与 AI 对战",
            "挑战神经网络与 MCTS 对手",
            font,
        );
        spawn_primary_button(
            screen,
            MenuAction::PlayLocal,
            "本地对战",
            "双人同屏轮流落子",
            font,
        );
    });
}

fn spawn_primary_button(
    parent: &mut ChildSpawnerCommands,
    action: MenuAction,
    title: &str,
    subtitle: &str,
    font: &Handle<Font>,
) {
    parent
        .spawn((
            Button,
            action,
            Node {
                width: percent(100),
                height: px(68),
                padding: UiRect::axes(px(18), px(10)),
                flex_direction: FlexDirection::Column,
                align_items: AlignItems::Center,
                justify_content: JustifyContent::Center,
                row_gap: px(4),
                border: UiRect::all(px(2)),
                border_radius: BorderRadius::all(px(10)),
                ..default()
            },
            BorderColor::all(Color::NONE),
            BackgroundColor(BUTTON_BG),
        ))
        .with_children(|btn| {
            btn.spawn(styles::menu_text_bundle(title, font, 20.0, TEXT_PRIMARY));
            btn.spawn(styles::menu_text_bundle(subtitle, font, 13.0, TEXT_MUTED));
        });
}

fn spawn_ai_setup_screen(card: &mut ChildSpawnerCommands, font: &Handle<Font>) {
    card.spawn((
        AiSetupScreenRoot,
        Node {
            display: Display::None,
            width: percent(100),
            flex_direction: FlexDirection::Column,
            row_gap: px(20),
            ..default()
        },
    ))
    .with_children(|screen| {
        spawn_side_selector(screen, font);
        spawn_difficulty_selector(screen, font);
        spawn_ai_screen_actions(screen, font);
    });
}

fn spawn_side_selector(parent: &mut ChildSpawnerCommands, font: &Handle<Font>) {
    parent
        .spawn((Node {
            width: percent(100),
            flex_direction: FlexDirection::Column,
            row_gap: px(8),
            ..default()
        },))
        .with_children(|section| {
            section.spawn(styles::menu_text_bundle(
                "选择己方执子",
                font,
                14.0,
                TEXT_MUTED,
            ));
            section
                .spawn((Node {
                    width: percent(100),
                    flex_direction: FlexDirection::Row,
                    column_gap: px(12),
                    ..default()
                },))
                .with_children(|row| {
                    spawn_option_button(
                        row,
                        MenuAction::SelectSide(Player::Black),
                        "执黑 (先手)",
                        true,
                        font,
                        Some(SideButton(Player::Black)),
                        None,
                    );
                    spawn_option_button(
                        row,
                        MenuAction::SelectSide(Player::White),
                        "执白 (后手)",
                        false,
                        font,
                        Some(SideButton(Player::White)),
                        None,
                    );
                });
        });
}

fn spawn_difficulty_selector(parent: &mut ChildSpawnerCommands, font: &Handle<Font>) {
    parent
        .spawn((Node {
            width: percent(100),
            flex_direction: FlexDirection::Column,
            row_gap: px(8),
            ..default()
        },))
        .with_children(|section| {
            section.spawn(styles::menu_text_bundle(
                "AI 思考深度 (MCTS 迭代次数)",
                font,
                14.0,
                TEXT_MUTED,
            ));
            section
                .spawn((Node {
                    width: percent(100),
                    flex_direction: FlexDirection::Row,
                    column_gap: px(10),
                    ..default()
                },))
                .with_children(|row| {
                    spawn_option_button(
                        row,
                        MenuAction::SelectDifficulty(AiDifficulty::Simple),
                        AiDifficulty::Simple.label(),
                        false,
                        font,
                        None,
                        Some(DifficultyButton(AiDifficulty::Simple)),
                    );
                    spawn_option_button(
                        row,
                        MenuAction::SelectDifficulty(AiDifficulty::Normal),
                        AiDifficulty::Normal.label(),
                        true,
                        font,
                        None,
                        Some(DifficultyButton(AiDifficulty::Normal)),
                    );
                    spawn_option_button(
                        row,
                        MenuAction::SelectDifficulty(AiDifficulty::Hard),
                        AiDifficulty::Hard.label(),
                        false,
                        font,
                        None,
                        Some(DifficultyButton(AiDifficulty::Hard)),
                    );
                });
        });
}

fn spawn_option_button(
    parent: &mut ChildSpawnerCommands,
    action: MenuAction,
    label: &str,
    is_selected: bool,
    font: &Handle<Font>,
    side_marker: Option<SideButton>,
    diff_marker: Option<DifficultyButton>,
) {
    let mut entity = parent.spawn((
        Button,
        action,
        Node {
            flex_grow: 1.0,
            height: px(46),
            align_items: AlignItems::Center,
            justify_content: JustifyContent::Center,
            border: UiRect::all(px(2)),
            border_radius: BorderRadius::all(px(8)),
            ..default()
        },
        BorderColor::all(if is_selected { ACCENT } else { Color::NONE }),
        BackgroundColor(if is_selected {
            BUTTON_SELECTED
        } else {
            BUTTON_BG
        }),
    ));

    if let Some(side) = side_marker {
        entity.insert(side);
    }
    if let Some(diff) = diff_marker {
        entity.insert(diff);
    }

    entity.with_children(|btn| {
        btn.spawn(styles::menu_text_bundle(label, font, 15.0, TEXT_PRIMARY));
    });
}

fn spawn_ai_screen_actions(parent: &mut ChildSpawnerCommands, font: &Handle<Font>) {
    parent
        .spawn((Node {
            width: percent(100),
            flex_direction: FlexDirection::Column,
            row_gap: px(12),
            margin: UiRect::top(px(8)),
            ..default()
        },))
        .with_children(|actions| {
            actions
                .spawn((
                    Button,
                    MenuAction::StartAiGame,
                    Node {
                        width: percent(100),
                        height: px(52),
                        align_items: AlignItems::Center,
                        justify_content: JustifyContent::Center,
                        border: UiRect::all(px(2)),
                        border_radius: BorderRadius::all(px(8)),
                        ..default()
                    },
                    BorderColor::all(ACCENT),
                    BackgroundColor(BUTTON_SELECTED),
                ))
                .with_children(|btn| {
                    btn.spawn(styles::menu_text_bundle("开始对战", font, 18.0, ACCENT));
                });

            actions
                .spawn((
                    Button,
                    MenuAction::BackToMain,
                    Node {
                        width: percent(100),
                        height: px(44),
                        align_items: AlignItems::Center,
                        justify_content: JustifyContent::Center,
                        border: UiRect::all(px(1)),
                        border_radius: BorderRadius::all(px(8)),
                        ..default()
                    },
                    BorderColor::all(Color::NONE),
                    BackgroundColor(BUTTON_BG),
                ))
                .with_children(|btn| {
                    btn.spawn(styles::menu_text_bundle("返回", font, 16.0, TEXT_MUTED));
                });
        });
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn menu_cleanup_removes_all_menu_entities() {
        let mut app = App::new();
        app.add_plugins(bevy::state::app::StatesPlugin);

        let entity = app.world_mut().spawn(MenuEntity).id();
        assert!(app.world().get_entity(entity).is_ok());

        app.add_systems(Update, cleanup_menu);
        app.update();

        assert!(app.world().get_entity(entity).is_err());
    }
}
