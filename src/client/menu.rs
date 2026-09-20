#![allow(dead_code)]

use bevy::prelude::*;

pub mod styles;
pub mod types;

#[allow(unused_imports)]
pub use types::{AiDifficulty, MenuAction, MenuEntity, MenuScreen, MenuSetupConfig};

/// Plugin that manages the HexGo Main Menu lifecycle, state, and UI.
pub struct MenuPlugin;

impl Plugin for MenuPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<MenuSetupConfig>();
    }
}
