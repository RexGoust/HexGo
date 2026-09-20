**English** | [简体中文](MANUAL_SMOKE_TEST.zh-CN.md)

> English is normative. If the translations differ, this document takes precedence.

# Local Game Manual Smoke Test

Run `cargo run`, then complete this checklist before submitting a change that affects the local game UI.

## Launch and Layout

1. Confirm that the compact hexagonal grid appears on the left and the status panel appears on the right.
2. Confirm that all board edges are visible and connected, without duplicate or missing line segments.
3. Resize the window to its minimum dimensions and back. The board must remain centered, proportional, and fully visible.
4. Confirm that Simplified Chinese labels render legibly. The application searches for Source Han Sans, Noto Sans CJK, WenQuanYi Zen Hei, Microsoft YaHei, or PingFang and logs an English warning before falling back to Bevy's default font.

## Pointer Play

1. Hover over several empty intersections. A translucent stone matching the current player must follow the nearest intersection.
2. Click an empty intersection. A stone must appear, the current player must change, and the last move must receive an orange marker.
3. Click the occupied intersection. The board and current player must remain unchanged, and the panel must show an occupied-point error.
4. Play a short capture sequence. Captured stones must disappear immediately after the engine accepts the move.
5. Attempt suicide and positional-superko moves when suitable positions are available. Each rejected move must leave the board and turn unchanged and show the matching error.

## Narrow and Touch Layout

For the web build, run `trunk serve` and open the displayed URL. Check desktop (1280×800) and phone (360×640 and 390×844) viewports, resize without reloading, and rotate the phone. The canvas must follow the browser viewport, including changes to the mobile browser toolbar, without stretching the board or offsetting pointer input. Status labels must wrap on narrow screens.

Before testing the game itself, disable the browser cache and throttle the connection. Confirm that the web loading screen reports the current downloaded and total sizes, a stable transfer speed, an estimated remaining time, and progress that reaches 100%. After the download, it must report that the game is starting and disappear only when initialization succeeds. Reload with the WASM cached and confirm that the loading screen still transitions cleanly without stale values. Test offline mode and confirm that a readable failure message and a focused Reload button remain available. Repeat these checks at each desktop and phone viewport listed above, including with reduced-motion emulation enabled.

1. Resize the window to portrait orientation (e.g. 360×640 or 390×844). Confirm that in-game controls move below the board, the action buttons wrap neatly, and desktop-only headings and keyboard help are hidden. In the AI setup screen on portrait screens, confirm that difficulty options stack vertically without horizontal overflow. When rotated to landscape (e.g. 800×360 or 640×360), confirm that the controls are placed beside the board on the right as a compact sidebar, keeping the board large and fully visible.
2. Confirm that the board remains centered and fully visible without overlapping controls in both portrait and landscape orientations.
3. On a touch-capable device, tap several empty intersections and confirm that each tap places exactly one stone at the nearest intersection.
4. Tap the control panel actions and confirm that the tap does not also place a stone on the board.
5. Open Game Rules on a touch-capable device, swipe the rule text both upward and downward, and confirm that the content scrolls in the matching direction and remains within its bounds. Close the dialog and confirm that the game state is unchanged.
6. End a game and confirm that the result modal displays cleanly. Return to a wide landscape desktop window and confirm that the detailed result card returns.

## Keyboard Play

1. Press `Tab` until the board is focused. Move through intersections with all four arrow keys and place a stone with `Enter` or `Space`.
2. Use `Tab` and `Shift+Tab` to visit Pass, Resign, Restart, and Game Rules. The focused action must have a visible accent border.
3. Activate each action using only the keyboard. Press `Escape` in a confirmation dialog and confirm that no state changes.

## Completion and Restart

1. Select Pass once. The pass count must become `1 / 2` and the player must change.
2. Place a stone. The pass count must return to `0 / 2`.
3. Pass twice consecutively. Further play must be disabled and the result panel must show the winner, margin, stones, territory, komi, and totals.
4. Restart, cancel the dialog, and confirm the position remains. Restart again and confirm; the empty board, Black turn, and zero pass count must return.
5. Start another game, choose Resign, cancel once, then confirm. The result must name the opponent as the winner without showing a scored result.
