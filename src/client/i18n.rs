use bevy::prelude::*;
use std::collections::HashMap;

const ZH_JSON: &str = include_str!("../../assets/locales/zh-CN.json");
const EN_JSON: &str = include_str!("../../assets/locales/en-US.json");

/// Supported languages in HexGo.
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, Hash, Default, serde::Serialize, serde::Deserialize,
)]
pub enum Language {
    #[default]
    ZhCn,
    EnUs,
}

impl Language {
    /// Toggles between Chinese and English.
    pub const fn toggle(self) -> Self {
        match self {
            Self::ZhCn => Self::EnUs,
            Self::EnUs => Self::ZhCn,
        }
    }

    /// Standard BCP 47 locale code.
    #[allow(dead_code)]
    pub const fn code(self) -> &'static str {
        match self {
            Self::ZhCn => "zh-CN",
            Self::EnUs => "en-US",
        }
    }
}

/// Resource holding the currently selected language.
#[derive(Resource, Debug, Clone, Copy, PartialEq, Eq)]
pub struct CurrentLanguage(pub Language);

impl Default for CurrentLanguage {
    fn default() -> Self {
        Self(detect_system_language().unwrap_or(Language::ZhCn))
    }
}

/// Attempts to detect system language from environment variables.
fn detect_system_language() -> Option<Language> {
    if let Ok(lang) = std::env::var("LANG").or_else(|_| std::env::var("LC_ALL")) {
        let lower = lang.to_lowercase();
        if lower.starts_with("zh") {
            return Some(Language::ZhCn);
        }
        if lower.starts_with("en") {
            return Some(Language::EnUs);
        }
    }
    None
}

/// Embedded catalog containing translation mappings for all supported languages.
#[derive(Resource, Clone, Debug)]
pub struct I18nStore {
    zh: HashMap<String, String>,
    en: HashMap<String, String>,
}

impl Default for I18nStore {
    fn default() -> Self {
        Self {
            zh: serde_json::from_str(ZH_JSON).expect("Failed to parse assets/locales/zh-CN.json"),
            en: serde_json::from_str(EN_JSON).expect("Failed to parse assets/locales/en-US.json"),
        }
    }
}

impl I18nStore {
    /// Translates a key for the given language, falling back to the key if not found.
    pub fn t<'a>(&'a self, lang: Language, key: &'a str) -> &'a str {
        let map = match lang {
            Language::ZhCn => &self.zh,
            Language::EnUs => &self.en,
        };
        map.get(key).map(|s| s.as_str()).unwrap_or(key)
    }

    /// Translates and substitutes named `{name}` placeholders in the template string.
    pub fn format(&self, lang: Language, key: &str, vars: &[(&str, &str)]) -> String {
        let mut text = self.t(lang, key).to_string();
        for (name, val) in vars {
            let placeholder = format!("{{{}}}", name);
            text = text.replace(&placeholder, val);
        }
        text
    }
}

/// Marker component for static text entities that update automatically on language change.
#[derive(Component, Debug, Clone, Copy, PartialEq, Eq)]
pub struct I18nKey(pub &'static str);

/// System that synchronizes all static text elements when CurrentLanguage changes.
pub fn sync_i18n_static_texts(
    lang: Res<CurrentLanguage>,
    store: Res<I18nStore>,
    mut query: Query<(&I18nKey, &mut Text)>,
) {
    if !lang.is_changed() {
        return;
    }
    for (key, mut text) in &mut query {
        let val = store.t(lang.0, key.0);
        if text.0 != val {
            text.0 = val.into();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn embedded_locales_are_valid_and_consistent() {
        let store = I18nStore::default();
        assert!(!store.zh.is_empty());
        assert!(!store.en.is_empty());

        // Ensure key parity between zh-CN and en-US
        for key in store.zh.keys() {
            assert!(
                store.en.contains_key(key),
                "Key {:?} is missing from en-US.json",
                key
            );
        }
        for key in store.en.keys() {
            assert!(
                store.zh.contains_key(key),
                "Key {:?} is missing from zh-CN.json",
                key
            );
        }
    }

    #[test]
    fn format_substitutes_placeholders_correctly() {
        let store = I18nStore::default();
        let formatted = store.format(
            Language::ZhCn,
            "game.turn",
            &[("player", "黑方"), ("role", "（己方）")],
        );
        assert_eq!(formatted, "当前执子：黑方（己方）");

        let formatted_en = store.format(
            Language::EnUs,
            "game.turn",
            &[("player", "Black"), ("role", " (You)")],
        );
        assert_eq!(formatted_en, "Turn: Black (You)");
    }

    #[test]
    fn language_toggle_alternates() {
        assert_eq!(Language::ZhCn.toggle(), Language::EnUs);
        assert_eq!(Language::EnUs.toggle(), Language::ZhCn);
    }
}
