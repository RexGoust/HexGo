#![allow(dead_code)]

use bevy::prelude::*;

use crate::game::player::Player;

/// Search iterations and preset difficulties for AI opponent.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum AiDifficulty {
    /// 50 iterations: fast, responsive.
    Simple,
    /// 200 iterations: moderate depth.
    #[default]
    Normal,
    /// 1000 iterations: deep tactical search.
    Hard,
}

impl AiDifficulty {
    /// Returns the number of MCTS iterations for this difficulty.
    pub const fn iterations(self) -> usize {
        match self {
            Self::Simple => 50,
            Self::Normal => 200,
            Self::Hard => 1000,
        }
    }

    /// User-facing label in Chinese.
    pub const fn label(self) -> &'static str {
        match self {
            Self::Simple => "简单 (50步)",
            Self::Normal => "普通 (200步)",
            Self::Hard => "困难 (1000步)",
        }
    }
}

/// Active view in the main menu hierarchy.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum MenuScreen {
    /// Main landing screen with primary entry points.
    #[default]
    Main,
    /// AI setup screen with player side and difficulty options.
    AiSetup,
}

/// Runtime configuration state maintained while in MainMenu.
#[derive(Resource, Debug, Clone, Copy, PartialEq, Eq)]
pub struct MenuSetupConfig {
    pub screen: MenuScreen,
    pub player_color: Player,
    pub difficulty: AiDifficulty,
}

impl Default for MenuSetupConfig {
    fn default() -> Self {
        Self {
            screen: MenuScreen::Main,
            player_color: Player::Black,
            difficulty: AiDifficulty::Normal,
        }
    }
}

/// Action triggered by interacting with a menu button.
#[derive(Component, Debug, Clone, Copy, PartialEq, Eq)]
pub enum MenuAction {
    /// Start a local 2-player game immediately.
    PlayLocal,
    /// Navigate to AI match setup screen.
    OpenAiSetup,
    /// Navigate back to main screen.
    BackToMain,
    /// Select player side (Black or White).
    SelectSide(Player),
    /// Select AI difficulty (Simple, Normal, Hard).
    SelectDifficulty(AiDifficulty),
    /// Start AI match with currently selected configuration.
    StartAiGame,
}

/// Marker component attached to root main menu entities for teardown on state exit.
#[derive(Component, Debug, Clone, Copy, PartialEq, Eq)]
pub struct MenuEntity;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn difficulty_iterations_match_requirements() {
        assert_eq!(AiDifficulty::Simple.iterations(), 50);
        assert_eq!(AiDifficulty::Normal.iterations(), 200);
        assert_eq!(AiDifficulty::Hard.iterations(), 1000);
    }

    #[test]
    fn default_menu_config_is_sane() {
        let config = MenuSetupConfig::default();
        assert_eq!(config.screen, MenuScreen::Main);
        assert_eq!(config.player_color, Player::Black);
        assert_eq!(config.difficulty, AiDifficulty::Normal);
    }

    #[test]
    fn difficulty_labels_are_non_empty() {
        let difficulties = [
            AiDifficulty::Simple,
            AiDifficulty::Normal,
            AiDifficulty::Hard,
        ];
        for diff in difficulties {
            assert!(!diff.label().is_empty());
        }
    }
}
