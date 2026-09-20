#![allow(dead_code)]

use bevy::prelude::*;

pub mod styles;
pub mod types;
pub mod view;

use crate::client::state::AppState;

#[allow(unused_imports)]
pub use types::{AiDifficulty, MenuAction, MenuEntity, MenuScreen, MenuSetupConfig};

/// Plugin that manages the HexGo Main Menu lifecycle, state, and UI.
pub struct MenuPlugin;

impl Plugin for MenuPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<MenuSetupConfig>()
            .add_systems(OnEnter(AppState::MainMenu), view::spawn_main_menu)
            .add_systems(OnExit(AppState::MainMenu), view::cleanup_menu);
    }
}
