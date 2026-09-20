use bevy::prelude::*;

pub const MENU_BG: Color = Color::srgb(0.12, 0.09, 0.06);
pub const CARD_BG: Color = Color::srgb(0.17, 0.13, 0.09);
pub const BUTTON_BG: Color = Color::srgb(0.35, 0.25, 0.16);
pub const BUTTON_HOVER: Color = Color::srgb(0.46, 0.33, 0.20);
pub const BUTTON_PRESSED: Color = Color::srgb(0.61, 0.42, 0.17);
pub const BUTTON_SELECTED: Color = Color::srgb(0.50, 0.36, 0.20);
pub const ACCENT: Color = Color::srgb(0.84, 0.61, 0.23);
pub const TEXT_PRIMARY: Color = Color::srgb(1.0, 0.96, 0.89);
pub const TEXT_MUTED: Color = Color::srgb(0.82, 0.72, 0.55);

pub fn load_menu_font(asset_server: &AssetServer) -> Handle<Font> {
    asset_server.load("fonts/SourceHanSansSC-Regular.otf")
}

pub fn menu_text_bundle(text: &str, font: &Handle<Font>, size: f32, color: Color) -> impl Bundle {
    (
        Text::new(text),
        TextFont {
            font: font.clone().into(),
            font_size: FontSize::Px(size),
            ..default()
        },
        TextLayout::new(Justify::Center, LineBreak::NoWrap),
        TextColor(color),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn color_palette_values_are_distinct() {
        assert_ne!(MENU_BG, TEXT_PRIMARY);
        assert_ne!(BUTTON_BG, ACCENT);
    }
}
