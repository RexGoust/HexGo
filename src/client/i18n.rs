use bevy::prelude::*;
use std::collections::HashMap;

const ZH_JSON: &str = include_str!("../../assets/locales/zh-CN.json");
const EN_JSON: &str = include_str!("../../assets/locales/en-US.json");

/// Supported languages in HexGo.
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, Hash, Default, serde::Serialize, serde::Deserialize,
)]
pub enum Language {
    ZhCn,
    #[default]
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
        Self(detect_default_language())
    }
}

/// Checks whether a locale string represents Mainland China.
///
/// Returns `true` if the locale includes the region code "CN" (or "156"), or specifies
/// Simplified Chinese script without a conflicting non-Mainland region.
/// Returns `false` for other regions (such as Taiwan `zh-TW`, Hong Kong `zh-HK`,
/// Singapore `zh-SG`, USA `en-US`, etc.) or empty/unparseable values.
pub fn is_mainland_china_locale(locale: &str) -> bool {
    let trimmed = locale.trim();
    if trimmed.is_empty() {
        return false;
    }

    // Strip character encoding (e.g. ".UTF-8", ".GBK") and modifier (e.g. "@euro", "@pinyin")
    let base = trimmed.split(['.', '@']).next().unwrap_or("").trim();
    if base.is_empty() {
        return false;
    }

    // Split into subtags (e.g. "zh_CN" -> ["zh", "CN"], "zh-Hans-CN" -> ["zh", "Hans", "CN"])
    let subtags: Vec<&str> = base.split(['-', '_']).filter(|s| !s.is_empty()).collect();
    if subtags.is_empty() {
        return false;
    }

    // If any subsequent subtag matches ISO 3166-1 alpha-2 "CN" or UN M.49 numeric "156"
    for &subtag in subtags.iter().skip(1) {
        if subtag.eq_ignore_ascii_case("cn") || subtag == "156" {
            return true;
        }
    }

    // If the entire locale string is just "cn" or "156"
    if subtags.len() == 1 && (subtags[0].eq_ignore_ascii_case("cn") || subtags[0] == "156") {
        return true;
    }

    // Simplified Chinese script ("zh-Hans" / "zh_Hans") without an explicit non-Mainland region
    if subtags.len() == 2
        && subtags[0].eq_ignore_ascii_case("zh")
        && subtags[1].eq_ignore_ascii_case("hans")
    {
        return true;
    }

    false
}

/// Resolves the default language given an optional system locale and an environment lookup function.
///
/// If Mainland China is detected either from the system locale or relevant environment variables,
/// returns [`Language::ZhCn`]. Otherwise, returns [`Language::EnUs`].
pub fn resolve_default_language<'a>(
    system_locale: Option<&str>,
    lookup_env: impl Fn(&str) -> Option<&'a str>,
) -> Language {
    if let Some(locale) = system_locale
        && is_mainland_china_locale(locale)
    {
        return Language::ZhCn;
    }

    for var_name in ["LC_ALL", "LC_MESSAGES", "LANG"] {
        if let Some(val) = lookup_env(var_name)
            && is_mainland_china_locale(val)
        {
            return Language::ZhCn;
        }
    }

    if let Some(val) = lookup_env("LANGUAGE") {
        for part in val.split(':') {
            if is_mainland_china_locale(part) {
                return Language::ZhCn;
            }
        }
    }

    Language::EnUs
}

/// Detects the default language based on system locale and region.
///
/// If the detected region is Mainland China, defaults to [`Language::ZhCn`].
/// For all other regions (or if locale cannot be detected), defaults to [`Language::EnUs`].
pub fn detect_default_language() -> Language {
    let env_store: HashMap<String, String> = ["LC_ALL", "LC_MESSAGES", "LANG", "LANGUAGE"]
        .into_iter()
        .filter_map(|k| std::env::var(k).ok().map(|v| (k.to_string(), v)))
        .collect();

    let sys_loc = sys_locale::get_locale();
    resolve_default_language(sys_loc.as_deref(), |k| env_store.get(k).map(|s| s.as_str()))
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
            &[("player", "P1"), ("role", "R1")],
        );
        assert!(formatted.contains("P1") && formatted.contains("R1"));

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

    #[test]
    fn language_default_is_en_us() {
        assert_eq!(Language::default(), Language::EnUs);
    }

    #[test]
    fn mainland_china_locales_are_detected() {
        // Explicit CN region codes (POSIX and BCP 47 formats)
        assert!(is_mainland_china_locale("zh-CN"));
        assert!(is_mainland_china_locale("zh_CN"));
        assert!(is_mainland_china_locale("zh_CN.UTF-8"));
        assert!(is_mainland_china_locale("zh_CN.GBK"));
        assert!(is_mainland_china_locale("zh_CN.gb18030"));
        assert!(is_mainland_china_locale("zh_CN.utf8"));
        assert!(is_mainland_china_locale("zh-Hans-CN"));
        assert!(is_mainland_china_locale("zh-Hans_CN"));
        assert!(is_mainland_china_locale("zh-Hans-CN.UTF-8"));
        assert!(is_mainland_china_locale("zh-CN@pinyin"));
        assert!(is_mainland_china_locale("cmn-Hans-CN"));
        assert!(is_mainland_china_locale("cmn-CN"));
        assert!(is_mainland_china_locale("en-CN"));
        assert!(is_mainland_china_locale("en_CN.UTF-8"));
        assert!(is_mainland_china_locale("zh-156"));
        assert!(is_mainland_china_locale("cn"));
        assert!(is_mainland_china_locale("CN"));

        // Simplified Chinese script without a conflicting region
        assert!(is_mainland_china_locale("zh-Hans"));
        assert!(is_mainland_china_locale("zh_Hans"));
    }

    #[test]
    fn non_mainland_locales_are_rejected() {
        // Other Chinese-speaking regions
        assert!(!is_mainland_china_locale("zh-TW"));
        assert!(!is_mainland_china_locale("zh_TW"));
        assert!(!is_mainland_china_locale("zh_TW.UTF-8"));
        assert!(!is_mainland_china_locale("zh-HK"));
        assert!(!is_mainland_china_locale("zh_HK"));
        assert!(!is_mainland_china_locale("zh_HK.UTF-8"));
        assert!(!is_mainland_china_locale("zh-MO"));
        assert!(!is_mainland_china_locale("zh_MO"));
        assert!(!is_mainland_china_locale("zh-SG"));
        assert!(!is_mainland_china_locale("zh_SG"));
        assert!(!is_mainland_china_locale("zh-Hans-SG"));
        assert!(!is_mainland_china_locale("zh-Hant"));
        assert!(!is_mainland_china_locale("zh-Hant-TW"));
        assert!(!is_mainland_china_locale("zh-Hant-HK"));

        // Other languages and regions
        assert!(!is_mainland_china_locale("en-US"));
        assert!(!is_mainland_china_locale("en-GB"));
        assert!(!is_mainland_china_locale("ja-JP"));
        assert!(!is_mainland_china_locale("ko-KR"));
        assert!(!is_mainland_china_locale("fr-FR"));
        assert!(!is_mainland_china_locale("de-DE"));
        assert!(!is_mainland_china_locale("ru-RU"));

        // Generic / empty
        assert!(!is_mainland_china_locale(""));
        assert!(!is_mainland_china_locale("   "));
        assert!(!is_mainland_china_locale("C"));
        assert!(!is_mainland_china_locale("POSIX"));
    }

    #[test]
    fn resolve_default_language_prioritizes_system_locale_and_falls_back_to_env() {
        let no_env = |_key: &str| None;

        // System locale is Mainland China -> ZhCn
        assert_eq!(
            resolve_default_language(Some("zh-CN"), no_env),
            Language::ZhCn
        );
        assert_eq!(
            resolve_default_language(Some("zh_CN.UTF-8"), no_env),
            Language::ZhCn
        );

        // System locale is other region -> EnUs
        assert_eq!(
            resolve_default_language(Some("en-US"), no_env),
            Language::EnUs
        );
        assert_eq!(
            resolve_default_language(Some("zh-TW"), no_env),
            Language::EnUs
        );
        assert_eq!(
            resolve_default_language(Some("ja-JP"), no_env),
            Language::EnUs
        );

        // No system locale, fallback to env vars
        assert_eq!(
            resolve_default_language(None, |k| if k == "LANG" {
                Some("zh_CN.UTF-8")
            } else {
                None
            }),
            Language::ZhCn
        );
        assert_eq!(
            resolve_default_language(None, |k| if k == "LC_ALL" {
                Some("zh-Hans-CN")
            } else {
                None
            }),
            Language::ZhCn
        );
        assert_eq!(
            resolve_default_language(None, |k| if k == "LANGUAGE" {
                Some("zh_CN:en_US")
            } else {
                None
            }),
            Language::ZhCn
        );

        // No system locale, env vars are non-China or empty -> EnUs
        assert_eq!(
            resolve_default_language(None, |k| if k == "LANG" {
                Some("en_US.UTF-8")
            } else {
                None
            }),
            Language::EnUs
        );
        assert_eq!(
            resolve_default_language(None, |k| if k == "LANG" {
                Some("zh_TW.UTF-8")
            } else {
                None
            }),
            Language::EnUs
        );
        assert_eq!(resolve_default_language(None, no_env), Language::EnUs);
    }
}
