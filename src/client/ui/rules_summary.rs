use crate::client::i18n::Language;

#[allow(dead_code)]
pub const SUMMARY_ZH: &str = "基本规则\n\
\n\
1. 黑方先行，双方轮流在棋盘交点落子。只有棋盘连线直接相连的交点才算相邻。\n\
\n\
2. 相连的同色棋子组成棋串。棋串相邻的空交点称为气；没有气的棋串会被提掉。\n\
\n\
3. 落子时先提掉相邻且无气的敌方棋串，再检查己方棋串。落子后己方仍无气属于自杀，禁止落子。\n\
\n\
4. 全局同形禁着：落子后的棋盘不得与本局此前出现过的任何棋盘局面相同。\n\
\n\
5. 每回合也可以停着或认输。停着永远合法；连续两次停着后，对局立即结束。\n\
\n\
6. 对局结束时采用面积计分：棋子数加己方围住的空点数；白方另加贴目。双方都接触的空域不计分。\n\
\n\
7. 结束时仍在棋盘上的棋子一律视为活棋。若有应被提掉的棋子，请继续落子，不要停着。";

#[allow(dead_code)]
pub const SUMMARY_EN: &str = "Basic Rules\n\
\n\
1. Black moves first, then players alternate placing a stone on a vertex. Two vertices are adjacent only when connected by a board edge.\n\
\n\
2. Connected stones of the same color form a group. Adjacent empty vertices are liberties; a group with no liberties is captured and removed.\n\
\n\
3. Captures are resolved before checking the placed stone. If the placed stone's group still has no liberties, the move is suicide and forbidden.\n\
\n\
4. Positional Superko: A move may not recreate any board position previously reached in the same game.\n\
\n\
5. A player may pass or resign on their turn. Passing is always legal. Two consecutive passes end the game immediately.\n\
\n\
6. The game is scored using area scoring: stones plus surrounded empty territory, with komi added for White. Neutral regions touching both players do not score.\n\
\n\
7. All stones remaining on the board at game end are treated as alive. If opponent stones should be captured, play them out rather than passing.";

/// Canonical Chinese summary preserved for backward compatibility.
#[allow(dead_code)]
pub const SUMMARY: &str = SUMMARY_ZH;

/// Returns the rules summary in the requested language.
#[allow(dead_code)]
pub fn summary(lang: Language) -> &'static str {
    match lang {
        Language::ZhCn => SUMMARY_ZH,
        Language::EnUs => SUMMARY_EN,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn summaries_contain_key_rules_concepts() {
        assert!(summary(Language::ZhCn).contains("全局同形禁着"));
        assert!(summary(Language::ZhCn).contains("连续两次停着"));
        assert!(summary(Language::EnUs).contains("Positional Superko"));
        assert!(summary(Language::EnUs).contains("Two consecutive passes"));
    }
}
