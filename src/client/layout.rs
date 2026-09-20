use bevy::{prelude::*, ui::GridPlacement, window::PrimaryWindow};

use crate::client::{
    SessionResource,
    board::*,
    ui::{AdaptiveContent, ResponsiveElement, SIDEBAR_WIDTH},
};

pub const COMPACT_SIDEBAR_WIDTH: f32 = 280.0;
pub const MOBILE_BUTTON_WIDTH: f32 = 154.0;
pub const MOBILE_BUTTON_HEIGHT: f32 = 46.0;
pub const COMPACT_BUTTON_HEIGHT: f32 = 36.0;
pub const MOBILE_PANEL_CONTENT_MAX_WIDTH: f32 = 340.0;

const MOBILE_PANEL_HEIGHT: f32 = 288.0;
const MOBILE_PANEL_MAX_HEIGHT_RATIO: f32 = 0.55;
const BOARD_PADDING: f32 = 72.0;
const MOBILE_BOARD_PADDING: f32 = 28.0;

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ResponsiveLayout {
    pub panel_on_bottom: bool,
    pub is_compact_landscape: bool,
    pub sidebar_width: f32,
    pub board_size: Vec2,
    pub board_center: Vec2,
}

fn mobile_panel_height(window_height: f32) -> f32 {
    MOBILE_PANEL_HEIGHT.min(window_height * MOBILE_PANEL_MAX_HEIGHT_RATIO)
}

pub fn responsive_layout(window_size: Vec2) -> ResponsiveLayout {
    let panel_on_bottom = window_size.x < window_size.y;
    if panel_on_bottom {
        let panel_height = mobile_panel_height(window_size.y);
        ResponsiveLayout {
            panel_on_bottom,
            is_compact_landscape: false,
            sidebar_width: 0.0,
            board_size: Vec2::new(
                (window_size.x - MOBILE_BOARD_PADDING * 2.0).max(1.0),
                (window_size.y - panel_height - MOBILE_BOARD_PADDING * 2.0).max(1.0),
            ),
            board_center: Vec2::new(0.0, panel_height * 0.5),
        }
    } else {
        let is_compact_landscape = window_size.y < 600.0 || window_size.x < 900.0;
        let sidebar_width = if is_compact_landscape {
            COMPACT_SIDEBAR_WIDTH
        } else {
            SIDEBAR_WIDTH
        };
        let board_padding = if is_compact_landscape {
            MOBILE_BOARD_PADDING
        } else {
            BOARD_PADDING
        };
        ResponsiveLayout {
            panel_on_bottom,
            is_compact_landscape,
            sidebar_width,
            board_size: Vec2::new(
                (window_size.x - sidebar_width - board_padding * 2.0).max(1.0),
                (window_size.y - board_padding * 2.0).max(1.0),
            ),
            board_center: Vec2::new(-sidebar_width * 0.5, 0.0),
        }
    }
}

pub fn layout_control_panel(
    window: Single<&Window, With<PrimaryWindow>>,
    mut elements: Query<(&ResponsiveElement, &mut Node)>,
) {
    let layout = responsive_layout(Vec2::new(window.width(), window.height()));
    for (element, mut node) in &mut elements {
        match (element, layout.panel_on_bottom, layout.is_compact_landscape) {
            (ResponsiveElement::ControlPanel, true, _) => {
                node.left = px(0);
                node.right = Val::Auto;
                node.top = Val::Auto;
                node.bottom = px(0);
                node.width = percent(100);
                node.height = px(mobile_panel_height(window.height()));
                node.padding = UiRect::new(px(14), px(14), px(10), px(10));
                node.row_gap = px(6);
                node.overflow = Overflow::visible();
            }
            (ResponsiveElement::ControlPanel, false, true) => {
                node.left = Val::Auto;
                node.right = px(0);
                node.top = px(0);
                node.bottom = Val::Auto;
                node.width = px(layout.sidebar_width);
                node.height = percent(100);
                node.padding = UiRect::new(px(12), px(12), px(10), px(10));
                node.row_gap = px(8);
                node.overflow = Overflow::scroll_y();
            }
            (ResponsiveElement::ControlPanel, false, false) => {
                node.left = Val::Auto;
                node.right = px(0);
                node.top = px(0);
                node.bottom = Val::Auto;
                node.width = px(layout.sidebar_width);
                node.height = percent(100);
                node.padding = UiRect::all(px(28));
                node.row_gap = px(16);
                node.overflow = Overflow::visible();
            }
            (ResponsiveElement::DesktopOnly, true, _)
            | (ResponsiveElement::DesktopOnly, false, true) => {
                node.display = Display::None;
            }
            (ResponsiveElement::DesktopOnly, false, false) => {
                node.display = Display::Flex;
            }
            (ResponsiveElement::StatusGroup, true, _) => {
                node.flex_direction = FlexDirection::Column;
                node.flex_wrap = FlexWrap::NoWrap;
                node.column_gap = px(0);
                node.justify_content = JustifyContent::Center;
                node.align_items = AlignItems::Center;
                node.row_gap = px(4);
                node.width = percent(100);
                node.max_width = px(MOBILE_PANEL_CONTENT_MAX_WIDTH);
                node.align_self = AlignSelf::Center;
            }
            (ResponsiveElement::StatusGroup, false, true) => {
                node.flex_direction = FlexDirection::Column;
                node.flex_wrap = FlexWrap::NoWrap;
                node.column_gap = px(0);
                node.justify_content = JustifyContent::FlexStart;
                node.align_items = AlignItems::FlexStart;
                node.row_gap = px(2);
                node.width = percent(100);
                node.max_width = Val::Auto;
                node.align_self = AlignSelf::Auto;
            }
            (ResponsiveElement::StatusGroup, false, false) => {
                node.flex_direction = FlexDirection::Column;
                node.flex_wrap = FlexWrap::NoWrap;
                node.column_gap = px(0);
                node.justify_content = JustifyContent::FlexStart;
                node.align_items = AlignItems::FlexStart;
                node.row_gap = px(16);
                node.width = percent(100);
                node.max_width = Val::Auto;
                node.align_self = AlignSelf::Auto;
            }
            (ResponsiveElement::ActionGroup, true, _) => {
                node.display = Display::Flex;
                node.flex_direction = FlexDirection::Row;
                node.flex_wrap = FlexWrap::Wrap;
                node.justify_content = JustifyContent::Center;
                node.align_items = AlignItems::Center;
                node.align_self = AlignSelf::Center;
                node.column_gap = px(8);
                node.row_gap = px(6);
                node.width = percent(100);
                node.max_width = px(MOBILE_PANEL_CONTENT_MAX_WIDTH);
                node.height = Val::Auto;
            }
            (ResponsiveElement::ActionGroup, false, true) => {
                node.display = Display::Flex;
                node.flex_direction = FlexDirection::Row;
                node.flex_wrap = FlexWrap::Wrap;
                node.justify_content = JustifyContent::Center;
                node.align_items = AlignItems::Center;
                node.align_self = AlignSelf::Center;
                node.column_gap = px(6);
                node.row_gap = px(6);
                node.width = percent(100);
                node.max_width = Val::Auto;
                node.height = Val::Auto;
            }
            (ResponsiveElement::ActionGroup, false, false) => {
                node.display = Display::Flex;
                node.flex_direction = FlexDirection::Column;
                node.flex_wrap = FlexWrap::NoWrap;
                node.justify_content = JustifyContent::FlexStart;
                node.align_items = AlignItems::Stretch;
                node.align_self = AlignSelf::Auto;
                node.column_gap = px(0);
                node.row_gap = px(16);
                node.width = percent(100);
                node.max_width = Val::Auto;
                node.height = Val::Auto;
            }
            (ResponsiveElement::ActionButton, true, _) => {
                layout_action_button(&mut node, ActionButtonLayout::Portrait);
            }
            (ResponsiveElement::ActionButton, false, true) => {
                layout_action_button(&mut node, ActionButtonLayout::CompactLandscape);
            }
            (ResponsiveElement::ActionButton, false, false) => {
                layout_action_button(&mut node, ActionButtonLayout::Desktop);
            }
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ActionButtonLayout {
    Portrait,
    CompactLandscape,
    Desktop,
}

pub fn layout_action_button(node: &mut Node, style: ActionButtonLayout) {
    match style {
        ActionButtonLayout::Portrait => {
            node.width = px(MOBILE_BUTTON_WIDTH);
            node.height = px(MOBILE_BUTTON_HEIGHT);
            node.min_width = px(MOBILE_BUTTON_WIDTH);
            node.max_width = px(MOBILE_BUTTON_WIDTH);
            node.flex_basis = Val::Auto;
            node.flex_grow = 0.0;
            node.flex_shrink = 0.0;
            node.padding = UiRect::horizontal(px(10));
            node.grid_column = GridPlacement::default();
        }
        ActionButtonLayout::CompactLandscape => {
            node.width = Val::Auto;
            node.height = px(COMPACT_BUTTON_HEIGHT);
            node.min_width = px(116.0);
            node.max_width = percent(100);
            node.flex_basis = px(116.0);
            node.flex_grow = 1.0;
            node.flex_shrink = 0.0;
            node.padding = UiRect::horizontal(px(6));
            node.grid_column = GridPlacement::default();
        }
        ActionButtonLayout::Desktop => {
            node.width = percent(100);
            node.height = px(48);
            node.min_width = Val::Auto;
            node.max_width = Val::Auto;
            node.flex_basis = Val::Auto;
            node.flex_grow = 0.0;
            node.flex_shrink = 0.0;
            node.padding = UiRect::horizontal(px(16));
            node.grid_column = GridPlacement::default();
        }
    }
}

pub fn layout_mobile_content(
    window: Single<&Window, With<PrimaryWindow>>,
    mut content: Query<(&AdaptiveContent, &mut Node, Option<&mut TextLayout>)>,
) {
    let resp_layout = responsive_layout(Vec2::new(window.width(), window.height()));
    let is_compact = resp_layout.panel_on_bottom || resp_layout.is_compact_landscape;
    for (kind, mut node, mut text_layout) in &mut content {
        match (kind, is_compact) {
            (AdaptiveContent::Feedback, true) => {
                node.min_height = px(22);
                node.margin = UiRect::ZERO;
                node.width = Val::Auto;
                node.max_width = if resp_layout.panel_on_bottom {
                    px(MOBILE_PANEL_CONTENT_MAX_WIDTH)
                } else {
                    Val::Auto
                };
                node.align_self = if resp_layout.panel_on_bottom {
                    AlignSelf::Center
                } else {
                    AlignSelf::FlexStart
                };
                if let Some(ref mut text_l) = text_layout {
                    text_l.justify = if resp_layout.panel_on_bottom {
                        Justify::Center
                    } else {
                        Justify::Left
                    };
                    text_l.linebreak = LineBreak::WordBoundary;
                }
            }
            (AdaptiveContent::Feedback, false) => {
                node.min_height = px(52);
                node.margin = UiRect::vertical(px(8));
                node.width = percent(100);
                node.max_width = Val::Auto;
                node.align_self = AlignSelf::Auto;
                if let Some(ref mut layout) = text_layout {
                    layout.justify = Justify::Left;
                    layout.linebreak = LineBreak::WordBoundary;
                }
            }
            (AdaptiveContent::Result, true) => {
                node.margin = UiRect::top(px(4));
                node.padding = UiRect::all(px(8));
            }
            (AdaptiveContent::Result, false) => {
                node.margin = UiRect::top(px(14));
                node.padding = UiRect::all(px(16));
            }
            (AdaptiveContent::ModalDialog, true) => {
                node.padding = UiRect::axes(px(20), px(16));
                node.max_height = percent(94);
                node.overflow = Overflow::scroll_y();
            }
            (AdaptiveContent::ModalDialog, false) => {
                node.padding = UiRect::all(px(28));
                node.max_height = percent(92);
                node.overflow = Overflow::scroll_y();
            }
            (AdaptiveContent::ResultButtonGroup, _) => {
                let is_narrow = window.width() < 500.0 || resp_layout.panel_on_bottom;
                node.flex_direction = FlexDirection::Row;
                if is_narrow {
                    node.flex_wrap = FlexWrap::Wrap;
                    node.column_gap = px(8);
                    node.row_gap = px(8);
                } else {
                    node.flex_wrap = FlexWrap::NoWrap;
                    node.column_gap = px(8);
                    node.row_gap = px(0);
                }
            }
            (AdaptiveContent::ResultButton, _) => {
                let is_narrow = window.width() < 500.0 || resp_layout.panel_on_bottom;
                if is_narrow {
                    node.height = px(42);
                    node.min_width = px(110.0);
                    node.max_width = percent(100);
                    node.flex_basis = px(110.0);
                    node.flex_grow = 1.0;
                    node.padding = UiRect::horizontal(px(6));
                } else {
                    node.height = px(46);
                    node.min_width = Val::Auto;
                    node.max_width = Val::Auto;
                    node.flex_basis = px(0);
                    node.flex_grow = 1.0;
                    node.padding = UiRect::horizontal(px(8));
                }
            }
        }
    }
}

pub fn fit_board_to_window(
    window: Single<&Window, With<PrimaryWindow>>,
    session: Res<SessionResource>,
    mut root: Single<&mut Transform, With<BoardRoot>>,
) {
    let layout = responsive_layout(Vec2::new(window.width(), window.height()));
    let (min, max) = session.0.definition().bounds();
    let extent = Vec2::new(max[0] - min[0], max[1] - min[1]);
    let scale = (layout.board_size.x / extent.x)
        .min(layout.board_size.y / extent.y)
        .max(0.01);
    root.scale = Vec3::splat(scale);
    root.translation = layout.board_center.extend(0.0);
}

pub fn screen_position_is_on_board(window_size: Vec2, position: Vec2) -> bool {
    let layout = responsive_layout(window_size);
    if layout.panel_on_bottom {
        position.y < window_size.y - mobile_panel_height(window_size.y)
    } else {
        position.x < window_size.x - layout.sidebar_width
    }
}

pub(super) fn is_mobile_layout(window_size: Vec2) -> bool {
    let layout = responsive_layout(window_size);
    layout.panel_on_bottom || layout.is_compact_landscape
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn responsive_layout_keeps_wide_controls_beside_the_board() {
        let layout = responsive_layout(Vec2::new(1280.0, 800.0));

        assert!(!layout.panel_on_bottom);
        assert!(!layout.is_compact_landscape);
        assert_eq!(layout.board_center, Vec2::new(-SIDEBAR_WIDTH * 0.5, 0.0));
        assert_eq!(
            layout.board_size,
            Vec2::new(
                1280.0 - SIDEBAR_WIDTH - BOARD_PADDING * 2.0,
                800.0 - BOARD_PADDING * 2.0,
            )
        );
    }

    #[test]
    fn responsive_layout_reserves_a_bottom_panel_for_portrait_screens() {
        let window_size = Vec2::new(480.0, 800.0);
        let layout = responsive_layout(window_size);

        assert!(layout.panel_on_bottom);
        assert!(!layout.is_compact_landscape);
        assert_eq!(
            layout.board_center,
            Vec2::new(0.0, MOBILE_PANEL_HEIGHT * 0.5)
        );
        assert!(screen_position_is_on_board(
            window_size,
            Vec2::new(240.0, 200.0)
        ));
        assert!(!screen_position_is_on_board(
            window_size,
            Vec2::new(240.0, 700.0)
        ));
    }

    #[test]
    fn responsive_layout_keeps_controls_beside_board_on_mobile_landscape() {
        let window_size = Vec2::new(800.0, 360.0);
        let layout = responsive_layout(window_size);

        assert!(!layout.panel_on_bottom);
        assert!(layout.is_compact_landscape);
        assert_eq!(layout.sidebar_width, COMPACT_SIDEBAR_WIDTH);
        assert_eq!(
            layout.board_center,
            Vec2::new(-COMPACT_SIDEBAR_WIDTH * 0.5, 0.0)
        );
        assert_eq!(
            layout.board_size,
            Vec2::new(
                800.0 - COMPACT_SIDEBAR_WIDTH - MOBILE_BOARD_PADDING * 2.0,
                360.0 - MOBILE_BOARD_PADDING * 2.0,
            )
        );
        assert!(screen_position_is_on_board(
            window_size,
            Vec2::new(300.0, 180.0)
        ));
        assert!(!screen_position_is_on_board(
            window_size,
            Vec2::new(700.0, 180.0)
        ));
    }

    #[test]
    fn controls_follow_viewport_changes_without_restarting() {
        let mut app = App::new();
        let window = app
            .world_mut()
            .spawn((Window::default(), PrimaryWindow))
            .id();
        let panel = app
            .world_mut()
            .spawn((ResponsiveElement::ControlPanel, Node::default()))
            .id();
        let status = app
            .world_mut()
            .spawn((ResponsiveElement::StatusGroup, Node::default()))
            .id();
        let actions = app
            .world_mut()
            .spawn((ResponsiveElement::ActionGroup, Node::default()))
            .id();
        let action_btn = app
            .world_mut()
            .spawn((ResponsiveElement::ActionButton, Node::default()))
            .id();
        app.add_systems(Update, layout_control_panel);

        for (width, height, is_portrait, is_compact_landscape) in [
            (1280.0, 800.0, false, false),
            (360.0, 640.0, true, false),
            (800.0, 360.0, false, true),
            (390.0, 844.0, true, false),
            (1280.0, 800.0, false, false),
        ] {
            app.world_mut()
                .get_mut::<Window>(window)
                .unwrap()
                .resolution
                .set(width, height);
            app.update();
            let node = app.world().get::<Node>(panel).unwrap();
            assert_eq!(
                node.width,
                if is_portrait {
                    percent(100)
                } else if is_compact_landscape {
                    px(COMPACT_SIDEBAR_WIDTH)
                } else {
                    px(SIDEBAR_WIDTH)
                }
            );
            let node = app.world().get::<Node>(status).unwrap();
            assert_eq!(
                node.align_items,
                if is_portrait {
                    AlignItems::Center
                } else {
                    AlignItems::FlexStart
                }
            );
            assert_eq!(
                node.row_gap,
                if is_portrait {
                    px(4)
                } else if is_compact_landscape {
                    px(2)
                } else {
                    px(16)
                }
            );
            let node = app.world().get::<Node>(actions).unwrap();
            assert_eq!(
                node.justify_content,
                if is_portrait || is_compact_landscape {
                    JustifyContent::Center
                } else {
                    JustifyContent::FlexStart
                }
            );
            let node = app.world().get::<Node>(action_btn).unwrap();
            assert_eq!(
                node.height,
                if is_portrait {
                    px(MOBILE_BUTTON_HEIGHT)
                } else if is_compact_landscape {
                    px(COMPACT_BUTTON_HEIGHT)
                } else {
                    px(48)
                }
            );
            let layout = responsive_layout(Vec2::new(width, height));
            assert!(layout.board_size.x > 0.0 && layout.board_size.y > 0.0);
        }
    }

    #[test]
    fn mobile_action_buttons_share_the_available_row_width() {
        let mut node = Node::default();

        layout_action_button(&mut node, ActionButtonLayout::Portrait);
        assert_eq!(node.width, px(MOBILE_BUTTON_WIDTH));
        assert_eq!(node.height, px(MOBILE_BUTTON_HEIGHT));
        assert_eq!(node.min_width, px(MOBILE_BUTTON_WIDTH));
        assert_eq!(node.max_width, px(MOBILE_BUTTON_WIDTH));
        assert_eq!(node.flex_grow, 0.0);

        layout_action_button(&mut node, ActionButtonLayout::CompactLandscape);
        assert_eq!(node.width, Val::Auto);
        assert_eq!(node.height, px(COMPACT_BUTTON_HEIGHT));
        assert_eq!(node.min_width, px(116.0));
        assert_eq!(node.max_width, percent(100));
        assert_eq!(node.flex_basis, px(116.0));
        assert_eq!(node.flex_grow, 1.0);

        layout_action_button(&mut node, ActionButtonLayout::Desktop);
        assert_eq!(node.width, percent(100));
        assert_eq!(node.height, px(48));
    }

    #[test]
    fn feedback_text_is_centered_on_mobile() {
        let mut app = App::new();
        let window = app
            .world_mut()
            .spawn((Window::default(), PrimaryWindow))
            .id();
        let feedback = app
            .world_mut()
            .spawn((
                AdaptiveContent::Feedback,
                Node::default(),
                TextLayout::new(Justify::Left, LineBreak::NoWrap),
            ))
            .id();
        app.add_systems(Update, layout_mobile_content);

        // Mobile portrait
        app.world_mut()
            .get_mut::<Window>(window)
            .unwrap()
            .resolution
            .set(360.0, 640.0);
        app.update();
        let node = app.world().get::<Node>(feedback).unwrap();
        let layout = app.world().get::<TextLayout>(feedback).unwrap();
        assert_eq!(node.width, Val::Auto);
        assert_eq!(node.align_self, AlignSelf::Center);
        assert_eq!(layout.justify, Justify::Center);
        assert_eq!(layout.linebreak, LineBreak::WordBoundary);

        // Mobile landscape
        app.world_mut()
            .get_mut::<Window>(window)
            .unwrap()
            .resolution
            .set(800.0, 360.0);
        app.update();
        let node = app.world().get::<Node>(feedback).unwrap();
        let layout = app.world().get::<TextLayout>(feedback).unwrap();
        assert_eq!(node.width, Val::Auto);
        assert_eq!(node.align_self, AlignSelf::FlexStart);
        assert_eq!(layout.justify, Justify::Left);
        assert_eq!(layout.linebreak, LineBreak::WordBoundary);

        // Desktop
        app.world_mut()
            .get_mut::<Window>(window)
            .unwrap()
            .resolution
            .set(1280.0, 800.0);
        app.update();
        let node = app.world().get::<Node>(feedback).unwrap();
        let layout = app.world().get::<TextLayout>(feedback).unwrap();
        assert_eq!(node.width, percent(100));
        assert_eq!(node.align_self, AlignSelf::Auto);
        assert_eq!(layout.justify, Justify::Left);
        assert_eq!(layout.linebreak, LineBreak::WordBoundary);
    }

    #[test]
    fn result_modal_buttons_wrap_on_narrow_screens() {
        let mut app = App::new();
        let window = app
            .world_mut()
            .spawn((Window::default(), PrimaryWindow))
            .id();
        let button_group = app
            .world_mut()
            .spawn((AdaptiveContent::ResultButtonGroup, Node::default()))
            .id();
        let button = app
            .world_mut()
            .spawn((AdaptiveContent::ResultButton, Node::default()))
            .id();
        app.add_systems(Update, layout_mobile_content);

        // Mobile portrait (360x640)
        app.world_mut()
            .get_mut::<Window>(window)
            .unwrap()
            .resolution
            .set(360.0, 640.0);
        app.update();
        let group_node = app.world().get::<Node>(button_group).unwrap();
        let btn_node = app.world().get::<Node>(button).unwrap();
        assert_eq!(group_node.flex_wrap, FlexWrap::Wrap);
        assert_eq!(btn_node.min_width, px(110.0));
        assert_eq!(btn_node.height, px(42.0));

        // Desktop (1280x800)
        app.world_mut()
            .get_mut::<Window>(window)
            .unwrap()
            .resolution
            .set(1280.0, 800.0);
        app.update();
        let group_node = app.world().get::<Node>(button_group).unwrap();
        let btn_node = app.world().get::<Node>(button).unwrap();
        assert_eq!(group_node.flex_wrap, FlexWrap::NoWrap);
        assert_eq!(btn_node.min_width, Val::Auto);
        assert_eq!(btn_node.height, px(46.0));
    }
}
