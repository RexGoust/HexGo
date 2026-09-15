use super::ui::{CurrentPlayerText, GameModeText, PassCountText, ResultPanel, ResultText};
use crate::{
    client::{
        FocusTarget, ModalKind, SessionResource, UiState,
        board::{Marker, PreviewStone, Stone},
        can_do_game_action, layout,
        materials::StoneMaterials,
        ui::{ERROR, FeedbackText, MUTED_TEXT, ModalOverlay, ModalText, RulesOverlay, RulesScroll},
    },
    game::{
        GameResult,
        player::Player,
        state::{GameStatus, VertexState},
    },
    session::{GameMode, GameSession},
};
use bevy::{prelude::*, window::PrimaryWindow};

pub(super) fn player_role_tag(player: Player, mode: GameMode) -> &'static str {
    match mode {
        GameMode::Local => "",
        GameMode::AI(my_player) => {
            if player == my_player {
                "（己方）"
            } else {
                "（AI）"
            }
        }
        GameMode::Network(my_player) => {
            if player == my_player {
                "（己方）"
            } else {
                "（对方）"
            }
        }
        GameMode::SelfPlay => "（AI）",
    }
}

pub(super) fn game_mode_description(mode: GameMode) -> &'static str {
    match mode {
        GameMode::Local => "本地双人对局",
        GameMode::AI(Player::Black) => "人机对战（己方执黑）",
        GameMode::AI(Player::White) => "人机对战（己方执白）",
        GameMode::Network(Player::Black) => "网络对战（己方执黑）",
        GameMode::Network(Player::White) => "网络对战（己方执白）",
        GameMode::SelfPlay => "AI 对弈模式",
    }
}

pub(super) fn sync_game_mode(
    session: Res<SessionResource>,
    mut mode_text: Single<&mut Text, With<GameModeText>>,
) {
    let value = game_mode_description(session.0.mode());
    if mode_text.0 != value {
        mode_text.0 = value.into();
    }
}

pub(super) fn sync_stones(
    session: Res<SessionResource>,
    materials: Res<StoneMaterials>,
    mut stones: Query<(&Stone, &mut Visibility, &mut MeshMaterial2d<ColorMaterial>)>,
) {
    for (stone, mut visibility, mut material) in &mut stones {
        match session.0.vertex_state(stone.0) {
            Some(VertexState::Occupied(Player::Black)) => {
                *visibility = Visibility::Visible;
                material.0 = materials.black.clone();
            }
            Some(VertexState::Occupied(Player::White)) => {
                *visibility = Visibility::Visible;
                material.0 = materials.white.clone();
            }
            _ => *visibility = Visibility::Hidden,
        }
    }
}

pub(super) fn sync_preview(
    session: Res<SessionResource>,
    ui: Res<UiState>,
    materials: Res<StoneMaterials>,
    mut preview: Single<
        (
            &mut Transform,
            &mut Visibility,
            &mut MeshMaterial2d<ColorMaterial>,
        ),
        With<PreviewStone>,
    >,
) {
    let preview_vertex = ui.hovered.or((ui.focus == FocusTarget::Board)
        .then_some(ui.focused_vertex)
        .flatten());
    if ui.modal.is_none()
        && session.0.status() == GameStatus::Playing
        && can_do_game_action(&session.0, &ui)
        && let Some(vertex) = preview_vertex
        && session.0.vertex_state(vertex) == Some(VertexState::Empty)
    {
        let position = Vec2::from_array(session.0.definition().position(vertex).unwrap());
        preview.0.translation = position.extend(3.0);
        *preview.1 = Visibility::Visible;
        preview.2.0 = match session.0.current_player() {
            Player::Black => materials.preview_black.clone(),
            Player::White => materials.preview_white.clone(),
        };
    } else {
        *preview.1 = Visibility::Hidden;
    }
}

pub(super) fn sync_focus_marker(
    session: Res<SessionResource>,
    ui: Res<UiState>,
    mut markers: Query<(&Marker, &mut Transform, &mut Visibility)>,
) {
    for (marker, mut transform, mut visibility) in &mut markers {
        if !matches!(marker, Marker::Focus) {
            continue;
        }

        if ui.focus == FocusTarget::Board
            && ui.modal.is_none()
            && let Some(vertex) = ui.focused_vertex
        {
            let position = Vec2::from_array(session.0.definition().position(vertex).unwrap());
            transform.translation = position.extend(1.0);
            *visibility = Visibility::Visible;
        } else {
            *visibility = Visibility::Hidden;
        }
    }
}

pub(super) fn sync_last_move_marker(
    session: Res<SessionResource>,
    mut markers: Query<(&Marker, &mut Transform, &mut Visibility)>,
) {
    for (marker, mut transform, mut visibility) in &mut markers {
        if !matches!(marker, Marker::LastMove) {
            continue;
        }

        if let Some(vertex) = session.0.last_move() {
            let position = Vec2::from_array(session.0.definition().position(vertex).unwrap());
            transform.translation = position.extend(4.0);
            *visibility = Visibility::Visible;
        } else {
            *visibility = Visibility::Hidden;
        }
    }
}

pub(super) fn sync_current_player(
    session: Res<SessionResource>,
    mut current_player: Single<&mut Text, With<CurrentPlayerText>>,
) {
    let value = match session.0.status() {
        GameStatus::Playing => {
            let current = session.0.current_player();
            let role = player_role_tag(current, session.0.mode());
            format!("当前执子：{}{}", player_name(current), role)
        }
        GameStatus::Finished(_) => "对局已结束".into(),
    };
    if current_player.0 != value {
        current_player.0 = value;
    }
}

pub(super) fn sync_pass_count(
    session: Res<SessionResource>,
    mut pass_count: Single<&mut Text, With<PassCountText>>,
) {
    let value = format!("连续停着：{} / 2", session.0.consecutive_passes());
    if pass_count.0 != value {
        pass_count.0 = value;
    }
}

pub(super) fn sync_result(
    window: Single<&Window, With<PrimaryWindow>>,
    session: Res<SessionResource>,
    mut result_panel: Single<&mut Node, With<ResultPanel>>,
    mut result_text: Single<&mut Text, With<ResultText>>,
) {
    if let Some(result) = session.0.result() {
        result_panel.display = Display::Flex;
        let is_mobile = layout::is_mobile_layout(Vec2::new(window.width(), window.height()));
        let value = if is_mobile {
            compact_result_summary(&session.0, result)
        } else {
            result_summary(&session.0, result)
        };
        if result_text.0 != value {
            result_text.0 = value;
        }
    } else {
        result_panel.display = Display::None;
    }
}

fn compact_result_summary(session: &GameSession, result: GameResult) -> String {
    let mode = session.mode();
    match result {
        GameResult::WinByResignation { winner } => {
            let role = player_role_tag(winner, mode);
            format!("对局结果：{}{}因对方认输获胜", player_name(winner), role)
        }
        GameResult::WinByScore { winner, margin } => {
            let score = session.score_breakdown();
            let winner_role = player_role_tag(winner, mode);
            format!(
                "对局结果：{}{}胜 {:.1} 目\n黑方 {:.1}（棋 {} / 地 {}）\n白方 {:.1}（棋 {} / 地 {} / 贴 {:.1}）",
                player_name(winner),
                winner_role,
                margin,
                score.black_total,
                score.black_stones,
                score.black_territory,
                score.white_total,
                score.white_stones,
                score.white_territory,
                score.komi,
            )
        }
        GameResult::Draw => {
            let score = session.score_breakdown();
            format!(
                "对局结果：和棋\n黑方总分：{:.1} · 白方总分：{:.1}",
                score.black_total, score.white_total
            )
        }
    }
}

pub(super) fn default_feedback_for_mode(session: &GameSession) -> String {
    if session.status() != GameStatus::Playing {
        return "对局已结束".into();
    }
    match session.mode() {
        GameMode::Local => "请选择一个交点落子".into(),
        GameMode::AI(my_player) => {
            if session.current_player() == my_player {
                "轮到己方，请选择交点落子".into()
            } else {
                "AI 正在思考中...".into()
            }
        }
        GameMode::Network(my_player) => {
            if session.current_player() == my_player {
                "轮到己方，请选择交点落子".into()
            } else {
                "等待对方落子...".into()
            }
        }
        GameMode::SelfPlay => "AI 自动对弈中...".into(),
    }
}

pub(super) fn sync_feedback(
    session: Res<SessionResource>,
    ui: Res<UiState>,
    mut feedback: Single<(&mut Text, &mut TextColor), With<FeedbackText>>,
) {
    let is_waiting_opponent = session.0.status() == GameStatus::Playing
        && !ui.feedback_is_error
        && ui.modal.is_none()
        && match session.0.mode() {
            GameMode::AI(my_player) => session.0.current_player() != my_player,
            GameMode::Network(my_player) => session.0.current_player() != my_player,
            GameMode::SelfPlay => true,
            GameMode::Local => false,
        };

    let value = if ui.feedback.is_empty() || is_waiting_opponent {
        default_feedback_for_mode(&session.0)
    } else {
        ui.feedback.clone()
    };
    if feedback.0.0 != value {
        feedback.0.0 = value;
    }
    feedback.1.0 = if ui.feedback_is_error {
        ERROR
    } else {
        MUTED_TEXT
    };
}

fn player_name(player: Player) -> &'static str {
    match player {
        Player::Black => "黑方",
        Player::White => "白方",
    }
}

fn result_summary(session: &GameSession, result: GameResult) -> String {
    let mode = session.mode();
    match result {
        GameResult::WinByResignation { winner } => {
            let role = player_role_tag(winner, mode);
            format!("对局结果\n{}{}因对方认输获胜", player_name(winner), role)
        }
        GameResult::WinByScore { winner, margin } => {
            let score = session.score_breakdown();
            let winner_role = player_role_tag(winner, mode);
            let black_role = player_role_tag(Player::Black, mode);
            let white_role = player_role_tag(Player::White, mode);
            format!(
                "对局结果\n{}{}胜 {:.1} 目\n\n黑方{}\n棋子：{}\n领地：{}\n总分：{:.1}\n\n白方{}\n棋子：{}\n领地：{}\n贴目：{:.1}\n总分：{:.1}",
                player_name(winner),
                winner_role,
                margin,
                black_role,
                score.black_stones,
                score.black_territory,
                score.black_total,
                white_role,
                score.white_stones,
                score.white_territory,
                score.komi,
                score.white_total,
            )
        }
        GameResult::Draw => {
            let score = session.score_breakdown();
            format!(
                "对局结果\n和棋\n\n黑方总分：{:.1}\n白方总分：{:.1}",
                score.black_total, score.white_total
            )
        }
    }
}

pub(super) fn sync_modal(
    ui: Res<UiState>,
    mut overlay: Single<&mut Node, With<ModalOverlay>>,
    mut text: Single<&mut Text, With<ModalText>>,
) {
    match ui.modal {
        Some(ModalKind::Resign) => {
            overlay.display = Display::Flex;
            let value = "确认认输？\n当前对局将立即结束。";
            if text.0 != value {
                text.0 = value.into();
            }
        }
        Some(ModalKind::Restart) => {
            overlay.display = Display::Flex;
            let value = "确认重新开始？\n当前棋盘进度将被清空。";
            if text.0 != value {
                text.0 = value.into();
            }
        }
        Some(ModalKind::Rules) => overlay.display = Display::None,
        None => overlay.display = Display::None,
    }
}

pub(super) fn sync_rules_modal(
    ui: Res<UiState>,
    mut overlay: Single<&mut Node, With<RulesOverlay>>,
    mut scroll: Single<&mut ScrollPosition, With<RulesScroll>>,
) {
    let is_open = ui.modal == Some(ModalKind::Rules);
    let was_open = overlay.display == Display::Flex;
    overlay.display = if is_open {
        Display::Flex
    } else {
        Display::None
    };
    if is_open && !was_open {
        scroll.0 = Vec2::ZERO;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::session::{GameMode, SessionCommand};

    #[test]
    fn unrelated_ui_changes_do_not_reset_an_open_rules_scroll() {
        let mut app = App::new();
        app.init_resource::<UiState>()
            .add_systems(Update, sync_rules_modal);
        app.world_mut().spawn((
            RulesOverlay,
            Node {
                display: Display::None,
                ..default()
            },
        ));
        app.world_mut()
            .spawn((RulesScroll, ScrollPosition(Vec2::new(0.0, 120.0))));

        app.world_mut().resource_mut::<UiState>().modal = Some(ModalKind::Rules);
        app.update();
        {
            let world = app.world_mut();
            let mut query = world.query_filtered::<&mut ScrollPosition, With<RulesScroll>>();
            let mut scroll = query.single_mut(world).unwrap();
            assert_eq!(scroll.0, Vec2::ZERO);
            scroll.y = 64.0;
        }

        app.world_mut().resource_mut::<UiState>().hovered = None;
        app.update();
        {
            let world = app.world_mut();
            let mut query = world.query_filtered::<&ScrollPosition, With<RulesScroll>>();
            let scroll = query.single(world).unwrap();
            assert_eq!(scroll.y, 64.0);
        }
    }

    #[test]
    fn score_result_lines_fit_the_sidebar_card() {
        let modes = [
            GameMode::Local,
            GameMode::AI(Player::Black),
            GameMode::AI(Player::White),
            GameMode::SelfPlay,
        ];

        for mode in modes {
            let mut session = GameSession::compact(mode);
            session.submit(SessionCommand::Pass).unwrap();
            session.submit(SessionCommand::Pass).unwrap();
            let summary = result_summary(&session, session.result().unwrap());

            assert!(
                summary.lines().all(|line| line.chars().count() <= 14),
                "Summary line exceeded 14 characters for mode {:?}:\n{}",
                mode,
                summary
            );

            let compact_summary = compact_result_summary(&session, session.result().unwrap());
            assert!(
                compact_summary.lines().count() <= 3,
                "Compact summary exceeded 3 lines for mode {:?}",
                mode
            );

            let mut resign_session = GameSession::compact(mode);
            resign_session.submit(SessionCommand::Resign).unwrap();
            let resign_summary = result_summary(&resign_session, resign_session.result().unwrap());
            assert!(
                resign_summary
                    .lines()
                    .all(|line| line.chars().count() <= 14),
                "Resign summary line exceeded 14 characters for mode {:?}:\n{}",
                mode,
                resign_summary
            );
            let compact_resign =
                compact_result_summary(&resign_session, resign_session.result().unwrap());
            assert!(compact_resign.lines().count() <= 3);
        }
    }

    #[test]
    fn role_tags_and_descriptions_match_each_mode() {
        assert_eq!(game_mode_description(GameMode::Local), "本地双人对局");
        assert_eq!(
            game_mode_description(GameMode::AI(Player::Black)),
            "人机对战（己方执黑）"
        );
        assert_eq!(
            game_mode_description(GameMode::AI(Player::White)),
            "人机对战（己方执白）"
        );
        assert_eq!(
            game_mode_description(GameMode::Network(Player::Black)),
            "网络对战（己方执黑）"
        );
        assert_eq!(
            game_mode_description(GameMode::Network(Player::White)),
            "网络对战（己方执白）"
        );
        assert_eq!(game_mode_description(GameMode::SelfPlay), "AI 对弈模式");

        assert_eq!(player_role_tag(Player::Black, GameMode::Local), "");
        assert_eq!(player_role_tag(Player::White, GameMode::Local), "");

        assert_eq!(
            player_role_tag(Player::Black, GameMode::AI(Player::Black)),
            "（己方）"
        );
        assert_eq!(
            player_role_tag(Player::White, GameMode::AI(Player::Black)),
            "（AI）"
        );
        assert_eq!(
            player_role_tag(Player::Black, GameMode::AI(Player::White)),
            "（AI）"
        );
        assert_eq!(
            player_role_tag(Player::White, GameMode::AI(Player::White)),
            "（己方）"
        );

        assert_eq!(
            player_role_tag(Player::Black, GameMode::Network(Player::Black)),
            "（己方）"
        );
        assert_eq!(
            player_role_tag(Player::White, GameMode::Network(Player::Black)),
            "（对方）"
        );
        assert_eq!(player_role_tag(Player::Black, GameMode::SelfPlay), "（AI）");
        assert_eq!(player_role_tag(Player::White, GameMode::SelfPlay), "（AI）");
    }

    #[test]
    fn default_feedback_reflects_turn_and_mode() {
        let local = GameSession::compact(GameMode::Local);
        assert_eq!(default_feedback_for_mode(&local), "请选择一个交点落子");

        let ai_black = GameSession::compact(GameMode::AI(Player::Black));
        assert_eq!(
            default_feedback_for_mode(&ai_black),
            "轮到己方，请选择交点落子"
        );

        let ai_white = GameSession::compact(GameMode::AI(Player::White));
        assert_eq!(default_feedback_for_mode(&ai_white), "AI 正在思考中...");

        let self_play = GameSession::compact(GameMode::SelfPlay);
        assert_eq!(default_feedback_for_mode(&self_play), "AI 自动对弈中...");

        let mut finished = GameSession::compact(GameMode::Local);
        finished.submit(SessionCommand::Pass).unwrap();
        finished.submit(SessionCommand::Pass).unwrap();
        assert_eq!(default_feedback_for_mode(&finished), "对局已结束");
    }

    #[test]
    fn sync_current_player_shows_correct_role_indicator() {
        let mut app = App::new();
        app.insert_resource(SessionResource(GameSession::compact(GameMode::AI(
            Player::Black,
        ))))
        .add_systems(Update, sync_current_player);

        app.world_mut().spawn((CurrentPlayerText, Text::new("")));
        app.update();

        let world = app.world_mut();
        let mut query = world.query_filtered::<&Text, With<CurrentPlayerText>>();
        let text = query.single(world).unwrap();
        assert_eq!(text.0, "当前执子：黑方（己方）");
    }

    #[test]
    fn sync_game_mode_updates_mode_text() {
        let mut app = App::new();
        app.insert_resource(SessionResource(GameSession::compact(GameMode::AI(
            Player::Black,
        ))))
        .add_systems(Update, sync_game_mode);

        app.world_mut().spawn((GameModeText, Text::new("")));
        app.update();

        let world = app.world_mut();
        let mut query = world.query_filtered::<&Text, With<GameModeText>>();
        let text = query.single(world).unwrap();
        assert_eq!(text.0, "人机对战（己方执黑）");
    }
}
