use bevy::prelude::*;

use crate::client::input::ButtonAction;
use crate::client::state::InGameEntity;

pub mod rules_summary;

const TEXT_COLOR: Color = Color::srgb(1.0, 0.96, 0.89);
pub const MUTED_TEXT: Color = Color::srgb(0.82, 0.72, 0.55);
pub const SIDEBAR_WIDTH: f32 = 300.0;
const PANEL_BACKGROUND: Color = Color::srgb(0.17, 0.13, 0.09);
const BUTTON_BACKGROUND: Color = Color::srgb(0.35, 0.25, 0.16);
pub const ACCENT: Color = Color::srgb(0.84, 0.61, 0.23);
// pub const WARNING: Color = Color::srgb(0.78, 0.30, 0.21);
pub const ERROR: Color = Color::srgb(0.89, 0.38, 0.31);

#[derive(Debug, Clone, Copy, Component)]
pub enum ResponsiveElement {
    ControlPanel,
    DesktopOnly,
    StatusGroup,
    ActionGroup,
    ActionButton,
}

#[derive(Component)]
pub(super) struct GameModeText;

#[derive(Component)]
pub(super) struct CurrentPlayerText;

#[derive(Component)]
pub(super) struct PassCountText;

#[derive(Component)]
pub(super) struct FeedbackText;

#[derive(Component)]
pub(super) struct ResultPanel;

#[derive(Component)]
pub(super) struct ResultText;

#[derive(Component)]
pub(super) struct ModalOverlay;

#[derive(Component)]
pub(super) struct ModalText;

#[derive(Component)]
pub(super) struct RulesOverlay;

#[derive(Component)]
pub(super) struct RulesScroll;

#[derive(Debug, Clone, Copy, Component)]
pub enum AdaptiveContent {
    Feedback,
    Result,
    ModalDialog,
}

fn load_cjk_font(asset_server: &Res<AssetServer>) -> Handle<Font> {
    asset_server.load("fonts/SourceHanSansSC-Regular.otf")
}

fn text_bundle(text: &str, font: &Handle<Font>, size: f32, color: Color) -> impl Bundle {
    (
        Text::new(text),
        TextFont {
            font: font.clone().into(),
            font_size: FontSize::Px(size),
            ..default()
        },
        TextLayout::new(Justify::Left, LineBreak::NoWrap),
        TextColor(color),
    )
}

fn action_button(action: ButtonAction, label: &str, font: &Handle<Font>) -> impl Bundle {
    (
        Button,
        action,
        Node {
            width: percent(100),
            flex_grow: 1.0,
            height: px(48),
            padding: UiRect::horizontal(px(16)),
            align_items: AlignItems::Center,
            justify_content: JustifyContent::Center,
            border: UiRect::all(px(2)),
            border_radius: BorderRadius::all(px(8)),
            ..default()
        },
        BorderColor::all(Color::NONE),
        BackgroundColor(BUTTON_BACKGROUND),
        children![text_bundle(label, font, 17.0, TEXT_COLOR)],
    )
}

fn rules_close_button(font: &Handle<Font>) -> impl Bundle {
    (
        Button,
        ButtonAction::CloseRules,
        Node {
            width: percent(100),
            height: px(48),
            flex_shrink: 0.0,
            padding: UiRect::horizontal(px(16)),
            align_items: AlignItems::Center,
            justify_content: JustifyContent::Center,
            border: UiRect::all(px(2)),
            border_radius: BorderRadius::all(px(8)),
            ..default()
        },
        BorderColor::all(Color::NONE),
        BackgroundColor(Color::srgb(0.35, 0.25, 0.16)),
        children![text_bundle("关闭", font, 17.0, TEXT_COLOR)],
    )
}

fn rules_text_node() -> Node {
    Node {
        width: percent(100),
        flex_shrink: 0.0,
        ..default()
    }
}

fn spawn_sidebar(commands: &mut Commands, font: &Handle<Font>) {
    commands
        .spawn((
            InGameEntity,
            ResponsiveElement::ControlPanel,
            Node {
                position_type: PositionType::Absolute,
                right: px(0),
                top: px(0),
                width: px(SIDEBAR_WIDTH),
                height: percent(100),
                padding: UiRect::all(px(28)),
                flex_direction: FlexDirection::Column,
                row_gap: px(16),
                ..default()
            },
            BackgroundColor(PANEL_BACKGROUND),
        ))
        .with_children(|panel| {
            panel.spawn((
                ResponsiveElement::DesktopOnly,
                text_bundle("HEXGO", font, 31.0, TEXT_COLOR),
            ));
            panel.spawn((
                GameModeText,
                ResponsiveElement::DesktopOnly,
                text_bundle("本地双人对局", font, 16.0, MUTED_TEXT),
            ));
            panel.spawn((
                ResponsiveElement::DesktopOnly,
                Node {
                    height: px(16),
                    ..default()
                },
            ));
            panel
                .spawn((
                    ResponsiveElement::StatusGroup,
                    Node {
                        flex_direction: FlexDirection::Column,
                        row_gap: px(16),
                        ..default()
                    },
                ))
                .with_children(|status| {
                    status.spawn((
                        CurrentPlayerText,
                        text_bundle("当前执子：黑方", font, 23.0, TEXT_COLOR),
                    ));
                    status.spawn((
                        PassCountText,
                        text_bundle("连续停着：0 / 2", font, 17.0, MUTED_TEXT),
                    ));
                });
            panel.spawn((
                FeedbackText,
                AdaptiveContent::Feedback,
                text_bundle("请选择一个交点落子", font, 16.0, MUTED_TEXT),
                Node {
                    min_height: px(52),
                    margin: UiRect::vertical(px(8)),
                    ..default()
                },
            ));
            panel
                .spawn((
                    ResponsiveElement::ActionGroup,
                    Node {
                        width: percent(100),
                        flex_direction: FlexDirection::Column,
                        row_gap: px(16),
                        ..default()
                    },
                ))
                .with_children(|actions| {
                    actions.spawn((
                        ResponsiveElement::ActionButton,
                        action_button(ButtonAction::Pass, "停着", font),
                    ));
                    actions.spawn((
                        ResponsiveElement::ActionButton,
                        action_button(ButtonAction::Resign, "认输", font),
                    ));
                    actions.spawn((
                        ResponsiveElement::ActionButton,
                        action_button(ButtonAction::Restart, "重新开始", font),
                    ));
                    actions.spawn((
                        ResponsiveElement::ActionButton,
                        action_button(ButtonAction::Rules, "游戏规则", font),
                    ));
                });
            panel
                .spawn((
                    ResultPanel,
                    AdaptiveContent::Result,
                    Node {
                        display: Display::None,
                        width: percent(100),
                        margin: UiRect::top(px(14)),
                        padding: UiRect::all(px(16)),
                        border: UiRect::all(px(1)),
                        border_radius: BorderRadius::all(px(8)),
                        ..default()
                    },
                    BorderColor::all(ACCENT),
                    BackgroundColor(Color::srgb(0.23, 0.17, 0.11)),
                ))
                .with_children(|result| {
                    result.spawn((
                        ResultText,
                        text_bundle("", font, 16.0, TEXT_COLOR),
                        Node {
                            width: percent(100),
                            ..default()
                        },
                    ));
                });
            panel.spawn((
                ResponsiveElement::DesktopOnly,
                Node {
                    flex_grow: 1.0,
                    ..default()
                },
            ));
            panel.spawn((
                ResponsiveElement::DesktopOnly,
                text_bundle(
                    "Tab 切换区域 · 方向键选择 · Enter 确认",
                    font,
                    13.0,
                    MUTED_TEXT,
                ),
            ));
        });
}

fn spawn_modal(commands: &mut Commands, font: &Handle<Font>) {
    commands
        .spawn((
            InGameEntity,
            ModalOverlay,
            Node {
                display: Display::None,
                position_type: PositionType::Absolute,
                left: px(0),
                top: px(0),
                width: percent(100),
                height: percent(100),
                align_items: AlignItems::Center,
                justify_content: JustifyContent::Center,
                ..default()
            },
            GlobalZIndex(100),
            BackgroundColor(Color::srgba(0.08, 0.055, 0.03, 0.76)),
        ))
        .with_children(|overlay| {
            overlay
                .spawn((
                    AdaptiveContent::ModalDialog,
                    Node {
                        width: percent(90),
                        max_width: px(410),
                        padding: UiRect::all(px(28)),
                        flex_direction: FlexDirection::Column,
                        row_gap: px(18),
                        border_radius: BorderRadius::all(px(12)),
                        ..default()
                    },
                    BackgroundColor(PANEL_BACKGROUND),
                ))
                .with_children(|dialog| {
                    dialog.spawn((ModalText, text_bundle("", font, 21.0, TEXT_COLOR)));
                    dialog
                        .spawn((Node {
                            flex_direction: FlexDirection::Row,
                            column_gap: px(12),
                            ..default()
                        },))
                        .with_children(|buttons| {
                            buttons.spawn(action_button(ButtonAction::Confirm, "确认", font));
                            buttons.spawn(action_button(ButtonAction::Cancel, "取消", font));
                        });
                });
        });
}

fn spawn_rules_modal(commands: &mut Commands, font: &Handle<Font>) {
    commands
        .spawn((
            InGameEntity,
            RulesOverlay,
            Node {
                display: Display::None,
                position_type: PositionType::Absolute,
                left: px(0),
                top: px(0),
                width: percent(100),
                height: percent(100),
                padding: UiRect::all(px(20)),
                align_items: AlignItems::Center,
                justify_content: JustifyContent::Center,
                ..default()
            },
            GlobalZIndex(100),
            BackgroundColor(Color::srgba(0.08, 0.055, 0.03, 0.82)),
        ))
        .with_children(|overlay| {
            overlay
                .spawn((
                    Node {
                        width: percent(100),
                        max_width: px(640),
                        height: percent(88),
                        max_height: px(680),
                        padding: UiRect::all(px(24)),
                        flex_direction: FlexDirection::Column,
                        row_gap: px(16),
                        border_radius: BorderRadius::all(px(12)),
                        ..default()
                    },
                    BackgroundColor(PANEL_BACKGROUND),
                ))
                .with_children(|dialog| {
                    dialog.spawn(text_bundle("游戏规则", font, 27.0, TEXT_COLOR));
                    let mut scroll = dialog.spawn((
                        RulesScroll,
                        ScrollPosition::default(),
                        Interaction::default(),
                        Pickable {
                            is_hoverable: false,
                            should_block_lower: true,
                        },
                        Node {
                            width: percent(100),
                            flex_grow: 1.0,
                            flex_direction: FlexDirection::Column,
                            overflow: Overflow::scroll_y(),
                            padding: UiRect::right(px(10)),
                            ..default()
                        },
                    ));
                    scroll.with_children(|content| {
                        content
                            .spawn(text_bundle(rules_summary::SUMMARY, font, 16.0, TEXT_COLOR))
                            .insert(TextLayout::new(Justify::Left, LineBreak::AnyCharacter))
                            .insert(Pickable::IGNORE)
                            .insert(rules_text_node());
                    });
                    dialog.spawn(rules_close_button(font));
                });
        });
}

pub fn setup_ui(mut commands: Commands, asset_server: Res<AssetServer>) {
    let font = load_cjk_font(&asset_server);

    spawn_sidebar(&mut commands, &font);
    spawn_modal(&mut commands, &font);
    spawn_rules_modal(&mut commands, &font);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rules_text_keeps_its_full_height_inside_the_scroll_view() {
        let node = rules_text_node();

        assert_eq!(node.width, percent(100));
        assert_eq!(node.flex_shrink, 0.0);
    }
}
