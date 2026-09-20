use super::ui::{
    CurrentPlayerText, GameModeText, PassCountText, ResultModalDetails, ResultModalOverlay,
    ResultModalTitle, ResultPanel, ResultText,
};
use crate::client::i18n::{CurrentLanguage, I18nStore, Language};
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

pub(super) fn player_role_tag_key(player: Player, mode: GameMode) -> Option<&'static str> {
    match mode {
        GameMode::Local => None,
        GameMode::AI(my_player) => {
            if player == my_player {
                Some("role.you")
            } else {
                Some("role.ai")
            }
        }
        GameMode::Network(my_player) => {
            if player == my_player {
                Some("role.you")
            } else {
                Some("role.opponent")
            }
        }
        GameMode::SelfPlay => Some("role.ai"),
    }
}

pub(super) fn player_role_tag(
    player: Player,
    mode: GameMode,
    store: &I18nStore,
    lang: Language,
) -> &str {
    player_role_tag_key(player, mode)
        .map(|key| store.t(lang, key))
        .unwrap_or("")
}

pub(super) fn game_mode_key(mode: GameMode) -> &'static str {
    match mode {
        GameMode::Local => "game.mode_local",
        GameMode::AI(Player::Black) => "game.mode_ai_black",
        GameMode::AI(Player::White) => "game.mode_ai_white",
        GameMode::Network(Player::Black) => "game.mode_network_black",
        GameMode::Network(Player::White) => "game.mode_network_white",
        GameMode::SelfPlay => "game.mode_self_play",
    }
}

pub(super) fn game_mode_description(mode: GameMode, store: &I18nStore, lang: Language) -> &str {
    store.t(lang, game_mode_key(mode))
}

pub(super) fn sync_game_mode(
    session: Res<SessionResource>,
    store: Res<I18nStore>,
    lang: Res<CurrentLanguage>,
    mut mode_text: Single<&mut Text, With<GameModeText>>,
) {
    let value = game_mode_description(session.0.mode(), &store, lang.0);
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
    store: Res<I18nStore>,
    lang: Res<CurrentLanguage>,
    mut current_player: Single<&mut Text, With<CurrentPlayerText>>,
) {
    let value = match session.0.status() {
        GameStatus::Playing => {
            let current = session.0.current_player();
            let role = player_role_tag(current, session.0.mode(), &store, lang.0);
            let p_name = player_name(current, &store, lang.0);
            store.format(lang.0, "game.turn", &[("player", p_name), ("role", role)])
        }
        GameStatus::Finished(_) => store.t(lang.0, "game.over").into(),
    };
    if current_player.0 != value {
        current_player.0 = value;
    }
}

pub(super) fn sync_pass_count(
    session: Res<SessionResource>,
    store: Res<I18nStore>,
    lang: Res<CurrentLanguage>,
    mut pass_count: Single<&mut Text, With<PassCountText>>,
) {
    let count_str = session.0.consecutive_passes().to_string();
    let value = store.format(lang.0, "game.passes", &[("count", &count_str)]);
    if pass_count.0 != value {
        pass_count.0 = value;
    }
}

pub(super) fn sync_result(
    window: Single<&Window, With<PrimaryWindow>>,
    session: Res<SessionResource>,
    store: Res<I18nStore>,
    lang: Res<CurrentLanguage>,
    mut result_panel: Single<&mut Node, With<ResultPanel>>,
    mut result_text: Single<&mut Text, With<ResultText>>,
) {
    if let Some(result) = session.0.result() {
        let is_mobile = layout::is_mobile_layout(Vec2::new(window.width(), window.height()));
        if is_mobile {
            result_panel.display = Display::None;
        } else {
            result_panel.display = Display::Flex;
            let value = result_summary(&session.0, result, &store, lang.0);
            if result_text.0 != value {
                result_text.0 = value;
            }
        }
    } else {
        result_panel.display = Display::None;
    }
}

pub(super) fn default_feedback_key_for_mode(session: &GameSession) -> &'static str {
    if session.status() != GameStatus::Playing {
        return "game.over";
    }
    match session.mode() {
        GameMode::Local => "feedback.select_point",
        GameMode::AI(my_player) => {
            if session.current_player() == my_player {
                "feedback.your_turn"
            } else {
                "feedback.ai_thinking"
            }
        }
        GameMode::Network(my_player) => {
            if session.current_player() == my_player {
                "feedback.your_turn"
            } else {
                "feedback.waiting_opponent"
            }
        }
        GameMode::SelfPlay => "feedback.ai_playing",
    }
}

pub(super) fn default_feedback_for_mode(
    session: &GameSession,
    store: &I18nStore,
    lang: Language,
) -> String {
    store.t(lang, default_feedback_key_for_mode(session)).into()
}

pub(super) fn sync_feedback(
    session: Res<SessionResource>,
    ui: Res<UiState>,
    store: Res<I18nStore>,
    lang: Res<CurrentLanguage>,
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

    let value = if is_waiting_opponent {
        default_feedback_for_mode(&session.0, &store, lang.0)
    } else if let Some(key) = ui.feedback_key {
        store.t(lang.0, key).into()
    } else {
        default_feedback_for_mode(&session.0, &store, lang.0)
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

pub(super) fn player_name_key(player: Player) -> &'static str {
    match player {
        Player::Black => "player.black",
        Player::White => "player.white",
    }
}

pub(super) fn player_name(player: Player, store: &I18nStore, lang: Language) -> &str {
    store.t(lang, player_name_key(player))
}

pub(super) fn result_summary(
    session: &GameSession,
    result: GameResult,
    store: &I18nStore,
    lang: Language,
) -> String {
    let mode = session.mode();
    match result {
        GameResult::WinByResignation { winner } => {
            let role = player_role_tag(winner, mode, store, lang);
            let p_name = player_name(winner, store, lang);
            store.format(
                lang,
                "result.resignation_sidebar",
                &[("player", p_name), ("role", role)],
            )
        }
        GameResult::WinByScore { winner, margin } => {
            let score = session.score_breakdown();
            let winner_role = player_role_tag(winner, mode, store, lang);
            let black_role = player_role_tag(Player::Black, mode, store, lang);
            let white_role = player_role_tag(Player::White, mode, store, lang);
            let margin_str = format!("{:.1}", margin);
            let b_stones = score.black_stones.to_string();
            let b_territory = score.black_territory.to_string();
            let b_total = format!("{:.1}", score.black_total);
            let w_stones = score.white_stones.to_string();
            let w_territory = score.white_territory.to_string();
            let komi = format!("{:.1}", score.komi);
            let w_total = format!("{:.1}", score.white_total);
            let p_name = player_name(winner, store, lang);

            store.format(
                lang,
                "result.score_details_sidebar",
                &[
                    ("player", p_name),
                    ("role", winner_role),
                    ("margin", &margin_str),
                    ("black_role", black_role),
                    ("black_stones", &b_stones),
                    ("black_territory", &b_territory),
                    ("black_total", &b_total),
                    ("white_role", white_role),
                    ("white_stones", &w_stones),
                    ("white_territory", &w_territory),
                    ("komi", &komi),
                    ("white_total", &w_total),
                ],
            )
        }
        GameResult::Draw => {
            let score = session.score_breakdown();
            let b_total = format!("{:.1}", score.black_total);
            let w_total = format!("{:.1}", score.white_total);
            store.format(
                lang,
                "result.draw_sidebar",
                &[("black_total", &b_total), ("white_total", &w_total)],
            )
        }
    }
}

pub(super) fn sync_modal(
    ui: Res<UiState>,
    store: Res<I18nStore>,
    lang: Res<CurrentLanguage>,
    mut overlay: Single<&mut Node, With<ModalOverlay>>,
    mut text: Single<&mut Text, With<ModalText>>,
) {
    match ui.modal {
        Some(ModalKind::Resign) => {
            overlay.display = Display::Flex;
            let value = store.t(lang.0, "modal.resign_prompt");
            if text.0 != value {
                text.0 = value.into();
            }
        }
        Some(ModalKind::Restart) => {
            overlay.display = Display::Flex;
            let value = store.t(lang.0, "modal.restart_prompt");
            if text.0 != value {
                text.0 = value.into();
            }
        }
        Some(ModalKind::Rules) | Some(ModalKind::Result) => overlay.display = Display::None,
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

pub(super) fn sync_result_modal(
    session: Res<SessionResource>,
    mut ui: ResMut<UiState>,
    store: Res<I18nStore>,
    lang: Res<CurrentLanguage>,
    mut overlay: Single<&mut Node, With<ResultModalOverlay>>,
    mut title: Single<&mut Text, (With<ResultModalTitle>, Without<ResultModalDetails>)>,
    mut details: Single<&mut Text, (With<ResultModalDetails>, Without<ResultModalTitle>)>,
) {
    if session.0.result().is_some() && !ui.result_modal_seen && ui.modal.is_none() {
        ui.modal = Some(ModalKind::Result);
        ui.result_modal_seen = true;
    }

    let is_open = ui.modal == Some(ModalKind::Result);
    overlay.display = if is_open {
        Display::Flex
    } else {
        Display::None
    };

    if is_open && let Some(result) = session.0.result() {
        let mode = session.0.mode();
        let (title_text, details_text) =
            format_modal_result(&session.0, result, mode, &store, lang.0);
        if title.0 != title_text {
            title.0 = title_text;
        }
        if details.0 != details_text {
            details.0 = details_text;
        }
    }
}

pub(super) fn format_modal_result(
    session: &GameSession,
    result: GameResult,
    mode: GameMode,
    store: &I18nStore,
    lang: Language,
) -> (String, String) {
    match result {
        GameResult::WinByResignation { winner } => {
            let role = player_role_tag(winner, mode, store, lang);
            let p_name = player_name(winner, store, lang);
            let title = store.format(
                lang,
                "result.resignation_winner",
                &[("player", p_name), ("role", role)],
            );
            let details = store.t(lang, "result.resignation_details").to_string();
            (title, details)
        }
        GameResult::WinByScore { winner, margin } => {
            let score = session.score_breakdown();
            let winner_role = player_role_tag(winner, mode, store, lang);
            let black_role = player_role_tag(Player::Black, mode, store, lang);
            let white_role = player_role_tag(Player::White, mode, store, lang);
            let margin_str = format!("{:.1}", margin);
            let b_stones = score.black_stones.to_string();
            let b_territory = score.black_territory.to_string();
            let b_total = format!("{:.1}", score.black_total);
            let w_stones = score.white_stones.to_string();
            let w_territory = score.white_territory.to_string();
            let komi = format!("{:.1}", score.komi);
            let w_total = format!("{:.1}", score.white_total);
            let p_name = player_name(winner, store, lang);

            let title = store.format(
                lang,
                "result.score_winner",
                &[
                    ("player", p_name),
                    ("role", winner_role),
                    ("margin", &margin_str),
                ],
            );
            let details = store.format(
                lang,
                "result.score_details_modal",
                &[
                    ("black_role", black_role),
                    ("black_stones", &b_stones),
                    ("black_territory", &b_territory),
                    ("black_total", &b_total),
                    ("white_role", white_role),
                    ("white_stones", &w_stones),
                    ("white_territory", &w_territory),
                    ("komi", &komi),
                    ("white_total", &w_total),
                ],
            );
            (title, details)
        }
        GameResult::Draw => {
            let score = session.score_breakdown();
            let b_total = format!("{:.1}", score.black_total);
            let w_total = format!("{:.1}", score.white_total);
            let title = store.t(lang, "result.draw_title").to_string();
            let details = store.format(
                lang,
                "result.draw_details",
                &[("black_total", &b_total), ("white_total", &w_total)],
            );
            (title, details)
        }
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
        let store = I18nStore::default();
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
            let summary =
                result_summary(&session, session.result().unwrap(), &store, Language::ZhCn);

            assert!(
                summary.lines().all(|line| line.chars().count() <= 14),
                "Summary line exceeded 14 characters for mode {:?}:\n{}",
                mode,
                summary
            );

            let (modal_title, modal_details) = format_modal_result(
                &session,
                session.result().unwrap(),
                mode,
                &store,
                Language::ZhCn,
            );
            assert!(!modal_title.is_empty());
            assert!(!modal_details.is_empty());

            let mut resign_session = GameSession::compact(mode);
            resign_session.submit(SessionCommand::Resign).unwrap();
            let resign_summary = result_summary(
                &resign_session,
                resign_session.result().unwrap(),
                &store,
                Language::ZhCn,
            );
            assert!(
                resign_summary
                    .lines()
                    .all(|line| line.chars().count() <= 14),
                "Resign summary line exceeded 14 characters for mode {:?}:\n{}",
                mode,
                resign_summary
            );
            let (resign_title, resign_details) = format_modal_result(
                &resign_session,
                resign_session.result().unwrap(),
                mode,
                &store,
                Language::ZhCn,
            );
            assert!(!resign_title.is_empty());
            assert!(!resign_details.is_empty());
        }
    }

    #[test]
    fn role_tags_and_descriptions_match_each_mode() {
        let store = I18nStore::default();
        assert_eq!(
            game_mode_description(GameMode::Local, &store, Language::ZhCn),
            store.t(Language::ZhCn, "game.mode_local")
        );
        assert_eq!(
            game_mode_description(GameMode::AI(Player::Black), &store, Language::ZhCn),
            store.t(Language::ZhCn, "game.mode_ai_black")
        );
        assert_eq!(
            game_mode_description(GameMode::AI(Player::White), &store, Language::ZhCn),
            store.t(Language::ZhCn, "game.mode_ai_white")
        );
        assert_eq!(
            game_mode_description(GameMode::Network(Player::Black), &store, Language::ZhCn),
            store.t(Language::ZhCn, "game.mode_network_black")
        );
        assert_eq!(
            game_mode_description(GameMode::Network(Player::White), &store, Language::ZhCn),
            store.t(Language::ZhCn, "game.mode_network_white")
        );
        assert_eq!(
            game_mode_description(GameMode::SelfPlay, &store, Language::ZhCn),
            store.t(Language::ZhCn, "game.mode_self_play")
        );

        assert_eq!(
            game_mode_description(GameMode::Local, &store, Language::EnUs),
            "Local 2-Player"
        );
        assert_eq!(
            game_mode_description(GameMode::AI(Player::Black), &store, Language::EnUs),
            "VS AI (Playing Black)"
        );

        assert_eq!(
            player_role_tag(Player::Black, GameMode::Local, &store, Language::ZhCn),
            ""
        );
        assert_eq!(
            player_role_tag(Player::White, GameMode::Local, &store, Language::ZhCn),
            ""
        );

        assert_eq!(
            player_role_tag(
                Player::Black,
                GameMode::AI(Player::Black),
                &store,
                Language::ZhCn
            ),
            store.t(Language::ZhCn, "role.you")
        );
        assert_eq!(
            player_role_tag(
                Player::White,
                GameMode::AI(Player::Black),
                &store,
                Language::ZhCn
            ),
            store.t(Language::ZhCn, "role.ai")
        );
        assert_eq!(
            player_role_tag(
                Player::Black,
                GameMode::AI(Player::White),
                &store,
                Language::ZhCn
            ),
            store.t(Language::ZhCn, "role.ai")
        );
        assert_eq!(
            player_role_tag(
                Player::White,
                GameMode::AI(Player::White),
                &store,
                Language::ZhCn
            ),
            store.t(Language::ZhCn, "role.you")
        );

        assert_eq!(
            player_role_tag(
                Player::Black,
                GameMode::Network(Player::Black),
                &store,
                Language::ZhCn
            ),
            store.t(Language::ZhCn, "role.you")
        );
        assert_eq!(
            player_role_tag(
                Player::White,
                GameMode::Network(Player::Black),
                &store,
                Language::ZhCn
            ),
            store.t(Language::ZhCn, "role.opponent")
        );
        assert_eq!(
            player_role_tag(Player::Black, GameMode::SelfPlay, &store, Language::ZhCn),
            store.t(Language::ZhCn, "role.ai")
        );
        assert_eq!(
            player_role_tag(Player::White, GameMode::SelfPlay, &store, Language::ZhCn),
            store.t(Language::ZhCn, "role.ai")
        );

        // English role tags
        assert_eq!(
            player_role_tag(
                Player::Black,
                GameMode::AI(Player::Black),
                &store,
                Language::EnUs
            ),
            " (You)"
        );
        assert_eq!(
            player_role_tag(
                Player::White,
                GameMode::AI(Player::Black),
                &store,
                Language::EnUs
            ),
            " (AI)"
        );
    }

    #[test]
    fn default_feedback_reflects_turn_and_mode() {
        let store = I18nStore::default();
        let local = GameSession::compact(GameMode::Local);
        assert_eq!(
            default_feedback_for_mode(&local, &store, Language::ZhCn),
            store.t(Language::ZhCn, "feedback.select_point")
        );
        assert_eq!(
            default_feedback_for_mode(&local, &store, Language::EnUs),
            store.t(Language::EnUs, "feedback.select_point")
        );

        let ai_black = GameSession::compact(GameMode::AI(Player::Black));
        assert_eq!(
            default_feedback_for_mode(&ai_black, &store, Language::ZhCn),
            store.t(Language::ZhCn, "feedback.your_turn")
        );
        assert_eq!(
            default_feedback_for_mode(&ai_black, &store, Language::EnUs),
            store.t(Language::EnUs, "feedback.your_turn")
        );

        let ai_white = GameSession::compact(GameMode::AI(Player::White));
        assert_eq!(
            default_feedback_for_mode(&ai_white, &store, Language::ZhCn),
            store.t(Language::ZhCn, "feedback.ai_thinking")
        );
        assert_eq!(
            default_feedback_for_mode(&ai_white, &store, Language::EnUs),
            store.t(Language::EnUs, "feedback.ai_thinking")
        );

        let self_play = GameSession::compact(GameMode::SelfPlay);
        assert_eq!(
            default_feedback_for_mode(&self_play, &store, Language::ZhCn),
            store.t(Language::ZhCn, "feedback.ai_playing")
        );

        let mut finished = GameSession::compact(GameMode::Local);
        finished.submit(SessionCommand::Pass).unwrap();
        finished.submit(SessionCommand::Pass).unwrap();
        assert_eq!(
            default_feedback_for_mode(&finished, &store, Language::ZhCn),
            store.t(Language::ZhCn, "game.over")
        );
        assert_eq!(
            default_feedback_for_mode(&finished, &store, Language::EnUs),
            store.t(Language::EnUs, "game.over")
        );
    }

    #[test]
    fn sync_current_player_shows_correct_role_indicator() {
        let mut app = App::new();
        app.insert_resource(SessionResource(GameSession::compact(GameMode::AI(
            Player::Black,
        ))))
        .init_resource::<I18nStore>()
        .insert_resource(CurrentLanguage(Language::ZhCn))
        .add_systems(Update, sync_current_player);

        app.world_mut().spawn((CurrentPlayerText, Text::new("")));
        app.update();

        let store = I18nStore::default();
        let expected = store.format(
            Language::ZhCn,
            "game.turn",
            &[
                ("player", store.t(Language::ZhCn, "player.black")),
                ("role", store.t(Language::ZhCn, "role.you")),
            ],
        );
        let world = app.world_mut();
        let mut query = world.query_filtered::<&Text, With<CurrentPlayerText>>();
        let text = query.single(world).unwrap();
        assert_eq!(text.0, expected);
    }

    #[test]
    fn sync_current_player_shows_correct_role_indicator_in_english() {
        let mut app = App::new();
        app.insert_resource(SessionResource(GameSession::compact(GameMode::AI(
            Player::Black,
        ))))
        .init_resource::<I18nStore>()
        .insert_resource(CurrentLanguage(Language::EnUs))
        .add_systems(Update, sync_current_player);

        app.world_mut().spawn((CurrentPlayerText, Text::new("")));
        app.update();

        let store = I18nStore::default();
        let expected = store.format(
            Language::EnUs,
            "game.turn",
            &[
                ("player", store.t(Language::EnUs, "player.black")),
                ("role", store.t(Language::EnUs, "role.you")),
            ],
        );
        let world = app.world_mut();
        let mut query = world.query_filtered::<&Text, With<CurrentPlayerText>>();
        let text = query.single(world).unwrap();
        assert_eq!(text.0, expected);
    }

    #[test]
    fn sync_game_mode_updates_mode_text() {
        let mut app = App::new();
        app.insert_resource(SessionResource(GameSession::compact(GameMode::AI(
            Player::Black,
        ))))
        .init_resource::<I18nStore>()
        .insert_resource(CurrentLanguage(Language::ZhCn))
        .add_systems(Update, sync_game_mode);

        app.world_mut().spawn((GameModeText, Text::new("")));
        app.update();

        let store = I18nStore::default();
        let expected = store.t(Language::ZhCn, "game.mode_ai_black");
        let world = app.world_mut();
        let mut query = world.query_filtered::<&Text, With<GameModeText>>();
        let text = query.single(world).unwrap();
        assert_eq!(text.0, expected);
    }

    #[test]
    fn sync_game_mode_updates_mode_text_in_english() {
        let mut app = App::new();
        app.insert_resource(SessionResource(GameSession::compact(GameMode::AI(
            Player::Black,
        ))))
        .init_resource::<I18nStore>()
        .insert_resource(CurrentLanguage(Language::EnUs))
        .add_systems(Update, sync_game_mode);

        app.world_mut().spawn((GameModeText, Text::new("")));
        app.update();

        let store = I18nStore::default();
        let expected = store.t(Language::EnUs, "game.mode_ai_black");
        let world = app.world_mut();
        let mut query = world.query_filtered::<&Text, With<GameModeText>>();
        let text = query.single(world).unwrap();
        assert_eq!(text.0, expected);
    }

    #[test]
    fn result_triggers_result_modal_and_updates_content() {
        let mut app = App::new();
        let mut session = GameSession::compact(GameMode::Local);
        session.submit(SessionCommand::Pass).unwrap();
        session.submit(SessionCommand::Pass).unwrap();
        assert!(session.result().is_some());

        let store = I18nStore::default();
        let (expected_title, expected_details) = format_modal_result(
            &session,
            session.result().unwrap(),
            session.mode(),
            &store,
            Language::ZhCn,
        );

        app.insert_resource(SessionResource(session))
            .init_resource::<UiState>()
            .init_resource::<I18nStore>()
            .insert_resource(CurrentLanguage(Language::ZhCn))
            .add_systems(Update, sync_result_modal);

        app.world_mut().spawn((
            ResultModalOverlay,
            Node {
                display: Display::None,
                ..default()
            },
        ));
        app.world_mut().spawn((ResultModalTitle, Text::new("")));
        app.world_mut().spawn((ResultModalDetails, Text::new("")));

        app.update();

        let ui = app.world().resource::<UiState>();
        assert_eq!(ui.modal, Some(ModalKind::Result));
        assert!(ui.result_modal_seen);

        let world = app.world_mut();
        let mut overlay_q = world.query_filtered::<&Node, With<ResultModalOverlay>>();
        let overlay = overlay_q.single(world).unwrap();
        assert_eq!(overlay.display, Display::Flex);

        let mut title_q = world.query_filtered::<&Text, With<ResultModalTitle>>();
        let title = title_q.single(world).unwrap();
        assert_eq!(title.0, expected_title);

        let mut details_q = world.query_filtered::<&Text, With<ResultModalDetails>>();
        let details = details_q.single(world).unwrap();
        assert_eq!(details.0, expected_details);
    }

    #[test]
    fn closing_result_modal_does_not_retrigger_next_frame() {
        let mut app = App::new();
        let mut session = GameSession::compact(GameMode::Local);
        session.submit(SessionCommand::Pass).unwrap();
        session.submit(SessionCommand::Pass).unwrap();

        app.insert_resource(SessionResource(session))
            .init_resource::<UiState>()
            .init_resource::<I18nStore>()
            .insert_resource(CurrentLanguage(Language::ZhCn))
            .add_systems(Update, sync_result_modal);

        app.world_mut().spawn((
            ResultModalOverlay,
            Node {
                display: Display::None,
                ..default()
            },
        ));
        app.world_mut().spawn((ResultModalTitle, Text::new("")));
        app.world_mut().spawn((ResultModalDetails, Text::new("")));

        // First update triggers modal
        app.update();
        assert_eq!(
            app.world().resource::<UiState>().modal,
            Some(ModalKind::Result)
        );

        // User closes the modal to inspect board
        app.world_mut().resource_mut::<UiState>().modal = None;
        app.update();

        // Modal should remain closed and not re-trigger
        assert_eq!(app.world().resource::<UiState>().modal, None);
        let world = app.world_mut();
        let mut overlay_q = world.query_filtered::<&Node, With<ResultModalOverlay>>();
        let overlay = overlay_q.single(world).unwrap();
        assert_eq!(overlay.display, Display::None);
    }
}
