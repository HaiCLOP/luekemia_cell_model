# Design notes: following xohari.me

Studied 2026-11-10 from https://www.xohari.me (computed styles, CSS variables and screenshots, English and Assamese views).
The web app follows this visual language. Where xohari conflicts with our design no-go list, the no-go list wins and the substitution is noted.

## Palette (from xohari `:root` variables and computed colours)

| Token | Value | Use on xohari | Use here |
|---|---|---|---|
| `--bg` | `#faf2eb` | warm cream page background | page background |
| `--paper` | `#fffdf9` | workflow paper | panels, upload area |
| `--ink` | `#121410` | all body and headline text | body and headlines |
| `--muted` | `#4a6142` | secondary text (green-grey) | captions, helper text |
| `--olive` | `#1a7a3a` | links, brand, italic headline line, primary buttons | primary actions, italic accent line, "normal" decision |
| `--green` | `#1ea854` | big step numerals | secondary accent |
| `--lime` | `#b5f800` | stat "ticket" blocks, CTA pill | highlight blocks (sparingly) |
| `--line` | `#e0f0d0` | hairline borders | borders, dividers |
| `--pill` | `#d4f5c0` | chips | chips |
| footer | `#1a7a3a` block on white, white text | footer | footer |
| (added) eosin | `#a83f27` (present on xohari as `rgb(168,63,39)`) | rare accent | "leukemic" decision colour |
| (added) amber | `#b7791f` | n/a | "refer to expert" decision colour |

Decision colours are paired with text labels and icons-free shapes, never colour alone.

## Typography

- Display: **Cormorant Garamond** (500 to 700, with italics). Headlines are two lines: a roman line in ink and an
  **italic second line in olive green**, letter-spacing about -0.035em, tight leading (about 0.98 to 1.14).
- Body on xohari: **Inter Tight**. Our no-go list excludes Inter (and Inter Tight is the Inter family), so body text uses
  **Hanken Grotesk**, a similar tight grotesk, at the same sizes: 16 px base, 13.5 px small paragraphs, 600 weight for
  links and buttons, slight negative tracking (-0.02em) on labels.
- Numbers: big step numerals set in the sans at heavy weight in greens (xohari's 1 2 3 4 workflow).

## Layout and spacing

- Content max widths seen: 1160 to 1240 px for sections, 780 px for prose, 520 to 560 px for forms.
- Generous vertical rhythm: sections separated by 96 to 140 px on desktop.
- Corner radius `--radius: 18px` for panels; buttons and chips are full pills (999px).
- Nav height `--nav: 54px`.

## Header

- A **floating pill navigation bar**: white at 95% opacity, centred, max about 800 px wide, rounded bottom corners
  (radius 0 0 16px 16px) hanging from the top edge, soft shadow. Left: leaf glyph + wordmark in olive 600.
  Middle: plain text links. Right: a language pill (outline) and a filled lime pill CTA plus an outline pill.
- Here: wordmark + links (Analyse, Model card, Method) + an outline pill linking to the GitHub repo.

## Footer

- A large **olive-green block** inset from the page edges with rounded corners, white text, three columns with small
  pill labels above each column ("Explore", "Associated with", "Helpdesk"), big medium-weight link text, and a giant
  cropped wordmark at the bottom edge.
- A playful footnote line in small type ("*phone calls give us the jitters...").
- Here: columns "Explore", "Built by", "Data", the giant cropped wordmark, and the disclaimer line.

## Components seen

- Primary button: filled olive pill, white text, 600 weight. Secondary: outline pill with olive border and text.
- Stat "tickets": lime blocks with punched semicircle notches on the sides, label top-left, big number bottom-left.
- Category marquee: pale pill strip with dot bullets.
- Numbered workflow: giant numerals with a short heading and one sentence.

## Tone of copy

- Short, direct, confident sentences. Headlines are a plain statement plus an italic payoff
  ("Say it. Direct to / *Campus Redressal.*"). Helper text is one practical sentence.
- Light human touch in small print, never in headings.
- Here: plain statements about what the model does and does not do, with the disclaimer always visible.

## Conflicts with our no-go list

| xohari uses | Rule | Resolution |
|---|---|---|
| Inter Tight body font | no Inter or Geist | Hanken Grotesk |
| frosted translucent CTA panel over a photo | no glass cards | solid paper panel with hairline border |
| four stat tickets in a row | (three cards in a row banned; four is not, but avoid the pattern) | stats inline as a single ticket strip with three or four figures inside one block |
| hover lift and animated clouds and marquees | no hover animations | static; colour change on focus/hover only, no motion |
| illustrated people | not available, no stock art | one hand-drawn SVG cell motif (nucleus and cytoplasm) in the site greens |
| emoji / sparkle | banned | none |
| em dashes in copy | banned | none in copy (use commas, colons, full stops) |
