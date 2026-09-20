use bevy::{prelude::*, window::PrimaryWindow};

use super::{
    styles::{
        self, ACCENT, BUTTON_BG, BUTTON_SELECTED, CARD_BG, MENU_BG, TEXT_MUTED, TEXT_PRIMARY,
    },
    types::{AiDifficulty, MenuAction, MenuEntity},
};
use crate::{
    client::i18n::{CurrentLanguage, I18nKey, I18nStore, Language},
    game::player::Player,
};

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

/// Marker component for the container holding difficulty selection buttons.
#[derive(Component, Debug, Clone, Copy, PartialEq, Eq)]
pub struct DifficultyContainer;

/// Marker component for the main menu card container.
#[derive(Component, Debug, Clone, Copy, PartialEq, Eq)]
pub struct MenuCard;

/// Spawns the entire Main Menu UI tree.
pub fn spawn_main_menu(
    mut commands: Commands,
    asset_server: Res<AssetServer>,
    store: Res<I18nStore>,
    lang: Res<CurrentLanguage>,
) {
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
                MenuCard,
                Node {
                    width: percent(90),
                    max_width: px(520),
                    padding: UiRect::axes(px(26), px(30)),
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
                spawn_title_header(card, &font, &store, lang.0);
                spawn_main_screen(card, &font, &store, lang.0);
                spawn_ai_setup_screen(card, &font, &store, lang.0);
            });
        });
}

/// Destroys all menu entities when leaving AppState::MainMenu.
pub fn cleanup_menu(mut commands: Commands, query: Query<Entity, With<MenuEntity>>) {
    for entity in &query {
        commands.entity(entity).despawn();
    }
}

fn spawn_title_header(
    card: &mut ChildSpawnerCommands,
    font: &Handle<Font>,
    store: &I18nStore,
    lang: Language,
) {
    card.spawn((Node {
        flex_direction: FlexDirection::Column,
        align_items: AlignItems::Center,
        row_gap: px(6),
        margin: UiRect::bottom(px(8)),
        ..default()
    },))
        .with_children(|header| {
            header.spawn(styles::menu_text_bundle("HEXGO", font, 38.0, TEXT_PRIMARY));
            header.spawn((
                I18nKey("menu.subtitle"),
                styles::menu_text_bundle(store.t(lang, "menu.subtitle"), font, 15.0, TEXT_MUTED),
            ));
            spawn_language_button(header, font, store, lang);
        });
}

fn spawn_language_button(
    parent: &mut ChildSpawnerCommands,
    font: &Handle<Font>,
    store: &I18nStore,
    lang: Language,
) {
    parent
        .spawn((
            Button,
            MenuAction::ToggleLanguage,
            Node {
                height: px(32),
                padding: UiRect::axes(px(14), px(4)),
                align_items: AlignItems::Center,
                justify_content: JustifyContent::Center,
                border: UiRect::all(px(1)),
                border_radius: BorderRadius::all(px(16)),
                margin: UiRect::top(px(4)),
                ..default()
            },
            BorderColor::all(BUTTON_BG),
            BackgroundColor(Color::srgba(0.0, 0.0, 0.0, 0.2)),
        ))
        .with_children(|btn| {
            btn.spawn((
                I18nKey("menu.lang_switch"),
                styles::menu_text_bundle(store.t(lang, "menu.lang_switch"), font, 13.0, TEXT_MUTED),
            ));
        });
}

fn spawn_main_screen(
    card: &mut ChildSpawnerCommands,
    font: &Handle<Font>,
    store: &I18nStore,
    lang: Language,
) {
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
            "menu.vs_ai",
            "menu.vs_ai_desc",
            font,
            store,
            lang,
        );
        spawn_primary_button(
            screen,
            MenuAction::PlayLocal,
            "menu.local",
            "menu.local_desc",
            font,
            store,
            lang,
        );
    });
}

fn spawn_primary_button(
    parent: &mut ChildSpawnerCommands,
    action: MenuAction,
    title_key: &'static str,
    subtitle_key: &'static str,
    font: &Handle<Font>,
    store: &I18nStore,
    lang: Language,
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
            btn.spawn((
                I18nKey(title_key),
                styles::menu_text_bundle(store.t(lang, title_key), font, 20.0, TEXT_PRIMARY),
            ));
            btn.spawn((
                I18nKey(subtitle_key),
                styles::menu_text_bundle(store.t(lang, subtitle_key), font, 13.0, TEXT_MUTED),
            ));
        });
}

fn spawn_ai_setup_screen(
    card: &mut ChildSpawnerCommands,
    font: &Handle<Font>,
    store: &I18nStore,
    lang: Language,
) {
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
        spawn_side_selector(screen, font, store, lang);
        spawn_difficulty_selector(screen, font, store, lang);
        spawn_ai_screen_actions(screen, font, store, lang);
    });
}

enum OptionMarker {
    Side(Player),
    Difficulty(AiDifficulty),
}

fn spawn_side_selector(
    parent: &mut ChildSpawnerCommands,
    font: &Handle<Font>,
    store: &I18nStore,
    lang: Language,
) {
    parent
        .spawn((Node {
            width: percent(100),
            flex_direction: FlexDirection::Column,
            row_gap: px(8),
            ..default()
        },))
        .with_children(|section| {
            section.spawn((
                I18nKey("menu.choose_side"),
                styles::menu_text_bundle(store.t(lang, "menu.choose_side"), font, 14.0, TEXT_MUTED),
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
                        "menu.side_black",
                        store.t(lang, "menu.side_black"),
                        true,
                        font,
                        OptionMarker::Side(Player::Black),
                    );
                    spawn_option_button(
                        row,
                        MenuAction::SelectSide(Player::White),
                        "menu.side_white",
                        store.t(lang, "menu.side_white"),
                        false,
                        font,
                        OptionMarker::Side(Player::White),
                    );
                });
        });
}

fn spawn_difficulty_selector(
    parent: &mut ChildSpawnerCommands,
    font: &Handle<Font>,
    store: &I18nStore,
    lang: Language,
) {
    parent
        .spawn((Node {
            width: percent(100),
            flex_direction: FlexDirection::Column,
            row_gap: px(8),
            ..default()
        },))
        .with_children(|section| {
            section.spawn((
                I18nKey("menu.mcts_depth"),
                styles::menu_text_bundle(store.t(lang, "menu.mcts_depth"), font, 14.0, TEXT_MUTED),
            ));
            section
                .spawn((
                    DifficultyContainer,
                    Node {
                        width: percent(100),
                        flex_direction: FlexDirection::Row,
                        column_gap: px(10),
                        ..default()
                    },
                ))
                .with_children(|row| {
                    spawn_option_button(
                        row,
                        MenuAction::SelectDifficulty(AiDifficulty::Simple),
                        AiDifficulty::Simple.key(),
                        store.t(lang, AiDifficulty::Simple.key()),
                        false,
                        font,
                        OptionMarker::Difficulty(AiDifficulty::Simple),
                    );
                    spawn_option_button(
                        row,
                        MenuAction::SelectDifficulty(AiDifficulty::Normal),
                        AiDifficulty::Normal.key(),
                        store.t(lang, AiDifficulty::Normal.key()),
                        true,
                        font,
                        OptionMarker::Difficulty(AiDifficulty::Normal),
                    );
                    spawn_option_button(
                        row,
                        MenuAction::SelectDifficulty(AiDifficulty::Hard),
                        AiDifficulty::Hard.key(),
                        store.t(lang, AiDifficulty::Hard.key()),
                        false,
                        font,
                        OptionMarker::Difficulty(AiDifficulty::Hard),
                    );
                });
        });
}

fn spawn_option_button(
    parent: &mut ChildSpawnerCommands,
    action: MenuAction,
    key: &'static str,
    label: &str,
    is_selected: bool,
    font: &Handle<Font>,
    marker: OptionMarker,
) {
    let mut entity = parent.spawn((
        Button,
        action,
        Node {
            flex_grow: 1.0,
            height: px(46),
            padding: UiRect::axes(px(10), px(6)),
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

    match marker {
        OptionMarker::Side(side) => {
            entity.insert(SideButton(side));
        }
        OptionMarker::Difficulty(diff) => {
            entity.insert(DifficultyButton(diff));
        }
    }

    entity.with_children(|btn| {
        btn.spawn((
            I18nKey(key),
            styles::menu_text_bundle(label, font, 15.0, TEXT_PRIMARY),
        ));
    });
}

fn spawn_ai_screen_actions(
    parent: &mut ChildSpawnerCommands,
    font: &Handle<Font>,
    store: &I18nStore,
    lang: Language,
) {
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
                    btn.spawn((
                        I18nKey("menu.start_match"),
                        styles::menu_text_bundle(
                            store.t(lang, "menu.start_match"),
                            font,
                            18.0,
                            ACCENT,
                        ),
                    ));
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
                    btn.spawn((
                        I18nKey("menu.back"),
                        styles::menu_text_bundle(
                            store.t(lang, "menu.back"),
                            font,
                            16.0,
                            TEXT_MUTED,
                        ),
                    ));
                });
        });
}

type MenuCardFilter = (
    With<MenuCard>,
    Without<DifficultyContainer>,
    Without<DifficultyButton>,
);

/// Adapts main menu layout (such as difficulty button layout) between portrait/narrow and wide viewports.
pub fn layout_menu(
    window: Query<&Window, With<PrimaryWindow>>,
    mut container_query: Query<&mut Node, With<DifficultyContainer>>,
    mut buttons_query: Query<&mut Node, (With<DifficultyButton>, Without<DifficultyContainer>)>,
    mut card_query: Query<&mut Node, MenuCardFilter>,
) {
    let Some(win) = window.iter().next() else {
        return;
    };

    let is_portrait = win.width() < win.height() || win.width() < 520.0;
    for mut container in &mut container_query {
        if is_portrait {
            container.flex_direction = FlexDirection::Column;
            container.row_gap = px(8);
            container.column_gap = px(0);
        } else {
            container.flex_direction = FlexDirection::Row;
            container.row_gap = px(0);
            container.column_gap = px(10);
        }
    }

    for mut btn_node in &mut buttons_query {
        if is_portrait {
            btn_node.width = percent(100);
            btn_node.height = px(42);
            btn_node.flex_grow = 0.0;
        } else {
            btn_node.width = Val::Auto;
            btn_node.height = px(46);
            btn_node.flex_grow = 1.0;
        }
    }

    let is_compact_height = win.height() < 500.0;
    for mut card_node in &mut card_query {
        if is_compact_height {
            card_node.padding = UiRect::axes(px(20), px(12));
            card_node.row_gap = px(12);
            card_node.max_height = percent(96);
            card_node.overflow = Overflow::scroll_y();
        } else {
            card_node.padding = UiRect::axes(px(26), px(28));
            card_node.row_gap = px(20);
            card_node.max_height = percent(94);
            card_node.overflow = Overflow::scroll_y();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use bevy::window::PrimaryWindow;

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

    #[test]
    fn difficulty_selector_switches_to_column_on_portrait_screens() {
        let mut app = App::new();
        let window = app
            .world_mut()
            .spawn((Window::default(), PrimaryWindow))
            .id();
        let container = app
            .world_mut()
            .spawn((DifficultyContainer, Node::default()))
            .id();
        let btn = app
            .world_mut()
            .spawn((DifficultyButton(AiDifficulty::Normal), Node::default()))
            .id();
        let card = app.world_mut().spawn((MenuCard, Node::default())).id();

        app.add_systems(Update, layout_menu);

        // Portrait mobile screen (e.g. 390x844): difficulty buttons must be stacked in column to prevent overflow
        app.world_mut()
            .get_mut::<Window>(window)
            .unwrap()
            .resolution
            .set(390.0, 844.0);
        app.update();

        let node = app.world().get::<Node>(container).unwrap();
        assert_eq!(node.flex_direction, FlexDirection::Column);
        assert_eq!(node.row_gap, px(8));
        assert_eq!(node.column_gap, px(0));

        let btn_node = app.world().get::<Node>(btn).unwrap();
        assert_eq!(btn_node.width, percent(100));
        assert_eq!(btn_node.height, px(42));
        assert_eq!(btn_node.flex_grow, 0.0);

        // Landscape desktop screen (1280x800): difficulty buttons arrange in row
        app.world_mut()
            .get_mut::<Window>(window)
            .unwrap()
            .resolution
            .set(1280.0, 800.0);
        app.update();

        let node = app.world().get::<Node>(container).unwrap();
        assert_eq!(node.flex_direction, FlexDirection::Row);
        assert_eq!(node.row_gap, px(0));
        assert_eq!(node.column_gap, px(10));

        let btn_node = app.world().get::<Node>(btn).unwrap();
        assert_eq!(btn_node.width, Val::Auto);
        assert_eq!(btn_node.height, px(46));
        assert_eq!(btn_node.flex_grow, 1.0);

        // Landscape mobile screen (800x360): card uses compact padding and scroll
        app.world_mut()
            .get_mut::<Window>(window)
            .unwrap()
            .resolution
            .set(800.0, 360.0);
        app.update();

        let card_node = app.world().get::<Node>(card).unwrap();
        assert_eq!(card_node.max_height, percent(96));
        assert_eq!(card_node.overflow, Overflow::scroll_y());
    }
}
