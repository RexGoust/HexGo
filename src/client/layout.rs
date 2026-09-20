use bevy::{prelude::*, ui::GridPlacement, window::PrimaryWindow};

use crate::client::{
    SessionResource,
    board::*,
    input::ButtonAction,
    ui::{AdaptiveContent, ResponsiveElement, SIDEBAR_WIDTH},
};

const MOBILE_PANEL_HEIGHT: f32 = 288.0;
const MOBILE_PANEL_MAX_HEIGHT_RATIO: f32 = 0.55;
const MOBILE_BREAKPOINT: f32 = 800.0;
const BOARD_PADDING: f32 = 72.0;
const MOBILE_BOARD_PADDING: f32 = 28.0;

#[derive(Debug, Clone, Copy, PartialEq)]
struct ResponsiveLayout {
    panel_on_bottom: bool,
    board_size: Vec2,
    board_center: Vec2,
}

fn mobile_panel_height(window_height: f32) -> f32 {
    MOBILE_PANEL_HEIGHT.min(window_height * MOBILE_PANEL_MAX_HEIGHT_RATIO)
}

fn responsive_layout(window_size: Vec2) -> ResponsiveLayout {
    let panel_on_bottom = window_size.x < MOBILE_BREAKPOINT || window_size.x < window_size.y;
    if panel_on_bottom {
        let panel_height = mobile_panel_height(window_size.y);
        ResponsiveLayout {
            panel_on_bottom,
            board_size: Vec2::new(
                (window_size.x - MOBILE_BOARD_PADDING * 2.0).max(1.0),
                (window_size.y - panel_height - MOBILE_BOARD_PADDING * 2.0).max(1.0),
            ),
            board_center: Vec2::new(0.0, panel_height * 0.5),
        }
    } else {
        ResponsiveLayout {
            panel_on_bottom,
            board_size: Vec2::new(
                (window_size.x - SIDEBAR_WIDTH - BOARD_PADDING * 2.0).max(1.0),
                (window_size.y - BOARD_PADDING * 2.0).max(1.0),
            ),
            board_center: Vec2::new(-SIDEBAR_WIDTH * 0.5, 0.0),
        }
    }
}

pub fn layout_control_panel(
    window: Single<&Window, With<PrimaryWindow>>,
    mut elements: Query<(&ResponsiveElement, &mut Node, Option<&ButtonAction>)>,
) {
    let layout = responsive_layout(Vec2::new(window.width(), window.height()));
    for (element, mut node, action) in &mut elements {
        match (element, layout.panel_on_bottom) {
            (ResponsiveElement::ControlPanel, true) => {
                node.left = px(0);
                node.right = Val::Auto;
                node.top = Val::Auto;
                node.bottom = px(0);
                node.width = percent(100);
                node.height = px(mobile_panel_height(window.height()));
                node.padding = UiRect::all(px(14));
                node.row_gap = px(8);
            }
            (ResponsiveElement::ControlPanel, false) => {
                node.left = Val::Auto;
                node.right = px(0);
                node.top = px(0);
                node.bottom = Val::Auto;
                node.width = px(SIDEBAR_WIDTH);
                node.height = percent(100);
                node.padding = UiRect::all(px(28));
                node.row_gap = px(16);
            }
            (ResponsiveElement::DesktopOnly, is_mobile) => {
                node.display = if is_mobile {
                    Display::None
                } else {
                    Display::Flex
                };
            }
            (ResponsiveElement::StatusGroup, true) => {
                node.flex_direction = FlexDirection::Row;
                node.flex_wrap = FlexWrap::Wrap;
                node.column_gap = px(8);
                node.justify_content = JustifyContent::SpaceBetween;
                node.row_gap = px(0);
            }
            (ResponsiveElement::StatusGroup, false) => {
                node.flex_direction = FlexDirection::Column;
                node.flex_wrap = FlexWrap::NoWrap;
                node.column_gap = px(0);
                node.justify_content = JustifyContent::FlexStart;
                node.row_gap = px(16);
            }
            (ResponsiveElement::ActionGroup, true) => {
                node.display = Display::Grid;
                node.height = px(88);
                node.grid_template_columns = RepeatedGridTrack::flex(6, 1.0);
                node.grid_template_rows = RepeatedGridTrack::px(2, 40.0);
                node.column_gap = px(8);
                node.row_gap = px(8);
            }
            (ResponsiveElement::ActionGroup, false) => {
                node.display = Display::Flex;
                node.height = Val::Auto;
                node.grid_template_columns.clear();
                node.grid_template_rows.clear();
                node.flex_direction = FlexDirection::Column;
                node.column_gap = px(0);
                node.row_gap = px(16);
            }
            (ResponsiveElement::ActionButton, true) => {
                layout_action_button(&mut node, action, true);
            }
            (ResponsiveElement::ActionButton, false) => {
                layout_action_button(&mut node, action, false);
            }
        }
    }
}

pub fn layout_action_button(node: &mut Node, action: Option<&ButtonAction>, is_mobile: bool) {
    if is_mobile {
        node.width = percent(100);
        node.min_width = px(0);
        node.flex_basis = Val::Auto;
        node.flex_grow = 0.0;
        node.grid_column = match action {
            Some(ButtonAction::Pass | ButtonAction::Resign | ButtonAction::Rules) => {
                GridPlacement::span(2)
            }
            Some(ButtonAction::Restart | ButtonAction::MainMenu) => GridPlacement::span(3),
            _ => GridPlacement::default(),
        };
    } else {
        node.width = percent(100);
        node.min_width = Val::Auto;
        node.flex_basis = Val::Auto;
        node.flex_grow = 0.0;
        node.grid_column = GridPlacement::default();
    }
}

pub fn layout_mobile_content(
    window: Single<&Window, With<PrimaryWindow>>,
    mut content: Query<(&AdaptiveContent, &mut Node)>,
) {
    let is_mobile = responsive_layout(Vec2::new(window.width(), window.height())).panel_on_bottom;
    for (kind, mut node) in &mut content {
        match (kind, is_mobile) {
            (AdaptiveContent::Feedback, true) => {
                node.min_height = px(32);
                node.margin = UiRect::ZERO;
            }
            (AdaptiveContent::Feedback, false) => {
                node.min_height = px(52);
                node.margin = UiRect::vertical(px(8));
            }
            (AdaptiveContent::Result, true) => {
                node.margin = UiRect::top(px(4));
                node.padding = UiRect::all(px(8));
            }
            (AdaptiveContent::Result, false) => {
                node.margin = UiRect::top(px(14));
                node.padding = UiRect::all(px(16));
            }
            (AdaptiveContent::ModalDialog, true) => node.padding = UiRect::all(px(20)),
            (AdaptiveContent::ModalDialog, false) => node.padding = UiRect::all(px(28)),
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
        position.x < window_size.x - SIDEBAR_WIDTH
    }
}

pub(super) fn is_mobile_layout(window_size: Vec2) -> bool {
    responsive_layout(window_size).panel_on_bottom
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn responsive_layout_keeps_wide_controls_beside_the_board() {
        let layout = responsive_layout(Vec2::new(1280.0, 800.0));

        assert!(!layout.panel_on_bottom);
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
        app.add_systems(Update, layout_control_panel);

        for (width, height, mobile) in [
            (1280.0, 800.0, false),
            (360.0, 640.0, true),
            (390.0, 844.0, true),
            (1280.0, 800.0, false),
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
                if mobile {
                    percent(100)
                } else {
                    px(SIDEBAR_WIDTH)
                }
            );
            let node = app.world().get::<Node>(status).unwrap();
            assert_eq!(
                node.flex_wrap,
                if mobile {
                    FlexWrap::Wrap
                } else {
                    FlexWrap::NoWrap
                }
            );
            let layout = responsive_layout(Vec2::new(width, height));
            assert!(layout.board_size.x > 0.0 && layout.board_size.y > 0.0);
        }
    }

    #[test]
    fn mobile_action_buttons_share_the_available_row_width() {
        let mut node = Node::default();

        layout_action_button(&mut node, Some(&ButtonAction::Pass), true);

        assert_eq!(node.width, percent(100));
        assert_eq!(node.min_width, px(0));
        assert_eq!(node.flex_basis, Val::Auto);
        assert_eq!(node.flex_grow, 0.0);
        assert_eq!(node.grid_column, GridPlacement::span(2));

        layout_action_button(&mut node, Some(&ButtonAction::MainMenu), true);
        assert_eq!(node.grid_column, GridPlacement::span(3));
    }
}
