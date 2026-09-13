**English** | [简体中文](RULES.zh-CN.md)

> English is the normative version. If the two versions differ, this document takes precedence.

# HexGo v0.1 Rules

HexGo is a Go-like game played on the **line intersections of a regular hexagonal tiling**.

Its most important difference from ordinary Go is:

- An interior intersection in ordinary Go normally has **4 adjacent points**.
- An interior intersection in HexGo normally has **3 adjacent points**.
- Stone connections, liberties, and captures are determined only by lines that actually exist on the board.

---

## 1. Board

The HexGo board consists of a set of **vertices** and the **edges** that connect them.

Each vertex is in one of three states:

- Empty
- Black stone
- White stone

Stones may be placed only on vertices, not in the centers of hexagonal cells.

A normal interior vertex is connected to three adjacent vertices, so it has at most **3 neighbors**.

A boundary vertex may have only:

- 2 adjacent vertices
- 1 adjacent vertex

Two vertices are adjacent only when a direct edge between them exists on the board.

For example:

```text
    ○
   /
○─●
   \
    ○
```

The central `●` has three adjacent vertices.

Vertices that are geometrically close but separated across the interior of a hexagon are not adjacent.

---

## 2. Players

The game is played by two players:

- Black
- White

By default:

**Black moves first.**

The players then alternate turns.

---

## 3. Actions on a Turn

On a player's turn, that player may choose to:

1. Place one stone of their color on a legal empty vertex;
2. Pass;
3. Resign.

---

## 4. Groups

Two stones of the same color are connected when they occupy adjacent vertices.

All same-colored stones reachable from one another through a sequence of adjacent same-colored stones form a **group**.

For example:

```text
●─●
   \
    ●
```

The three Black stones are connected by board edges and therefore belong to one group.

Groups are the basic units used to determine liberties and captures.

---

## 5. Liberties

A **liberty** of a group is any empty vertex directly adjacent to that group.

An empty vertex is counted as only **one liberty**, even if it is adjacent to more than one stone in the group.

For example, an isolated interior stone:

```text
    ·
   /
·─●
   \
    ·
```

has three liberties.

Therefore:

> In HexGo, a normal isolated interior stone has at most 3 initial liberties.

---

## 6. Captures

When a group has **0 liberties**, it is captured immediately.

All stones in the captured group are removed from the board, and their vertices become empty again.

For example:

```text
    ○
   /
○─●
   \
    ○
```

The Black stone has no liberties and is therefore captured.

Regardless of how many stones a group contains, it remains on the board as long as the group has at least one liberty.

---

## 7. Move Resolution Order

After a player places a stone, resolve the move in the following order:

### Step 1: Place the stone

Place the player's stone on the chosen empty vertex.

### Step 2: Check opposing groups

Check every opposing group adjacent to the newly placed stone.

Remove every such group that has no liberties.

Multiple opposing groups may be captured by the same move.

### Step 3: Check the player's group

After all opposing captures are complete, check the group containing the newly placed stone.

If that group still has no liberties, the move is suicide and is illegal.

---

## 8. Suicide Is Forbidden

A player may not make a move that leaves their own group without liberties after captures have been resolved.

For example:

```text
    ○
   /
○─·
   \
    ○
```

If Black plays at `·`:

```text
    ○
   /
○─●
   \
    ○
```

and the move does not capture any White stones, the newly placed Black stone immediately has no liberties.

This move is illegal.

However, a move is legal if it temporarily fills the player's last liberty but simultaneously captures surrounding opposing stones and thereby gains a liberty.

Suicide must therefore be checked **after all opposing captures have been completed**.

---

## 9. Positional Superko

HexGo uses **positional superko**.

After a stone is placed and all captures are resolved:

> The move is illegal if the resulting board position is identical to any board position that occurred earlier in the game.

Two board positions are identical when:

- Every Black stone occupies the same vertex;
- Every White stone occupies the same vertex;
- Every empty vertex is the same.

The player who created the position is not considered.

This prevents not only an immediate recapture in a traditional ko, but also indefinite repetition through more complex cycles.

---

## 10. Pass Is Exempt from Superko

Passing does not change the board.

Therefore:

> Passing is always legal and is not subject to the positional superko check.

Otherwise, consecutive passes could not end the game normally.

Any normal stone placement resets the consecutive-pass count.

---

## 11. End of the Game

The game ends in any of the following cases.

### Case 1: Two consecutive passes

If one player passes and the other player then also passes:

**The game ends immediately.**

### Case 2: Resignation

A player may resign on their turn.

The opponent wins immediately after a resignation, and the board is not scored.

### Case 3: No legal moves

If a move leaves the opponent with no legal moves on the board:

**The game ends immediately.**

The player who made that move wins immediately.

---

## 12. Dead Stones

By default, HexGo does not use subjective adjudication of whether stones are alive or dead.

If a player believes that an opposing group is dead, the player should capture it before the game ends.

Therefore:

> Every stone that remains on the board after two consecutive passes is treated as alive and participates in scoring.

If a player believes the board still contains dead stones that must be resolved, that player should continue playing instead of passing.

This rule avoids requiring the software to decide life and death or to resolve disagreement between the players.

---

## 13. Empty Regions

After the game ends, partition all connected empty vertices into **empty regions**.

For each region, inspect every stone adjacent to it.

### Black territory

An empty region belongs to Black if it:

- Is adjacent to at least one Black stone; and
- Is not adjacent to any White stone.

### White territory

An empty region belongs to White if it:

- Is adjacent to at least one White stone; and
- Is not adjacent to any Black stone.

### Neutral region

An empty region is neutral and belongs to neither player if it:

- Is adjacent to both Black and White stones.

An empty region adjacent to no stones is also neutral.

The board boundary itself is neither Black nor White, so a region touching the board boundary may still be territory.

---

## 14. Scoring

HexGo uses **area scoring** by default.

Black's score is:

> Number of Black stones + number of empty vertices controlled by Black

White's score is:

> Number of White stones + number of empty vertices controlled by White + komi

That is:

```text
BlackScore =
    BlackStones
  + BlackTerritory

WhiteScore =
    WhiteStones
  + WhiteTerritory
  + Komi
```

Neutral regions do not score for either player.

---

## 15. Captures Are Not Scored Separately

Opposing stones captured during the game do not provide additional points.

Captures are valuable because they:

- Remove opposing stones;
- Free occupied vertices;
- Change control of regions at the end of the game.

The final score therefore does not require a record of how many stones each player captured.

---

## 16. Komi

Because HexGo's three-neighbor topology differs greatly from ordinary Go, conventional values such as 6.5 or 7.5 cannot be assumed to suit HexGo.

The rules therefore define a configurable parameter:

```text
Komi
```

White receives `Komi` additional points in the final score.

The recommended value during development and testing is:

```text
Komi = 0.5
```

The primary purpose of 0.5 komi is to prevent draws. It does not imply that Black and White are statistically balanced.

An appropriate value for competitive play should be determined later using substantial data from:

- AI self-play;
- Human games;
- Statistics across different board sizes.

---

## 17. Result

If:

```text
BlackScore > WhiteScore
```

Black wins.

If:

```text
WhiteScore > BlackScore
```

White wins.

If a competition uses integer komi and both scores are equal, the game is a draw.

A draw cannot occur when the default half-point komi is used.

---

## 18. Boundary Rules

The board boundary provides no additional liberties.

A stone's liberties come only from adjacent empty vertices that actually exist on the board.

A boundary stone may therefore have fewer initial liberties than an interior stone.

For example, consider a boundary vertex with only two connections:

```text
●
 \
  ·
```

If the vertex has only two valid adjacent vertices, it can have at most two liberties.

Therefore:

> The edge of the board naturally has a stronger enclosing effect.

This is part of the topology of the HexGo board, not a special rule.

---

## 19. Adjacency Is Defined Only by Edges

Every use of the following concepts:

- Adjacency
- Connection
- Group
- Liberty
- Capture
- Empty region

is determined only by edges in the board graph.

The entire ruleset can be formalized as a graph:

```text
G = (V, E)
```

where:

- `V` is the set of vertices on which stones may be placed;
- `E` is the set of edges connecting vertices.

If:

```text
(u, v) ∈ E
```

then `u` and `v` are adjacent.

Otherwise, they are not adjacent, regardless of how close they appear on screen.

---

## 20. Illegal Moves

A stone placement is illegal when:

1. The target vertex is already occupied;
2. The target position is not part of the board;
3. The player's group still has no liberties after all opposing captures are completed;
4. The resulting board position has occurred previously in the game.

An illegal move does not change the game state and does not consume the player's turn.

---

## 21. Complete Move-Legality Procedure

A stone placement can be summarized as:

```text
Choose an empty vertex
          ↓
Place the player's stone
          ↓
Find adjacent opposing groups
          ↓
Capture every opposing group with 0 liberties
          ↓
Check the player's group
          ↓
Does it have 0 liberties?
 ├─ Yes → Illegal
 └─ No
      ↓
Check the position history
      ↓
Does the board repeat a previous position?
 ├─ Yes → Illegal
 └─ No  → The move succeeds
```

---

## 22. Eyes

An **eye** is not a separate rule. It is a strategic concept that emerges naturally from liberties and connections.

An empty region surrounded by a player's stones can protect those groups if the opponent cannot legally play inside it.

Because a normal HexGo vertex has only three neighbors:

> HexGo eye shapes and life-and-death patterns differ substantially from those in four-neighbor Go.

Many established life-and-death conclusions from ordinary Go cannot be applied directly to HexGo.

---

## 23. Seki

Black and White groups may form a position similar to **seki** in ordinary Go when they restrain one another and neither player can capture the other without causing their own stones to die.

At the end of the game:

- Stones involved in seki remain on the board and count toward their owners' stone area;
- Shared empty vertices adjacent to both Black and White are neutral;
- Those neutral vertices do not count as either player's territory.

---

## 24. Standard Rules Summary

The core rules of HexGo can be summarized as follows:

> Black and White alternate placing stones on the vertices of a honeycomb board.<br>
> Stones connect through board edges, and each normal interior vertex has only three neighbors.<br>
> Connected stones of the same color form a group, and adjacent empty vertices are liberties.<br>
> A group with no liberties is captured immediately.<br>
> A move that leaves the player's group without liberties is forbidden.<br>
> A move that recreates any previous board position is forbidden.<br>
> Players may pass; two consecutive passes end the game.<br>
> The final score counts a player's stones and the empty vertices controlled by that player.

---

## 25. Key Differences from Ordinary Go

| Rule | Ordinary Go | HexGo |
|---|---:|---:|
| Basic board structure | Intersections of a square grid | Vertices of a hexagonal tiling |
| Normal interior neighbors | 4 | 3 |
| Maximum initial liberties of one stone | 4 | 3 |
| Group connection | Four directions | Three directions |
| Captures | Capture at zero liberties | Capture at zero liberties |
| Suicide | Forbidden | Forbidden |
| Ko | Cycles are forbidden | Positional superko |
| End condition | Both players pass | Both players pass |
| Default scoring | Depends on the ruleset | Area scoring |
| Life-and-death topology | Four-neighbor | Three-neighbor |

---

## One-Sentence Definition

**HexGo is Go played on the vertex graph of a honeycomb lattice, where every interior vertex has three neighbors.**
