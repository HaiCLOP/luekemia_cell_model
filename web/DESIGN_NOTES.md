# Design notes: the haematology bench

The site is styled after the physical world the model comes from: a microscope, a glass slide, a stained smear and a
lab report. Early direction took inspiration from the warmth of xohari.me (Dibrugarh University's campus redressal site):
a soft serif with expressive italics, generous space and plain-spoken copy. Everything else is our own.

## Palette (from the Giemsa stain)

| Token | Value | Use |
|---|---|---|
| `--paper` | `#f6f1e7` | bench paper background |
| `--card` | `#fffcf6` | slide labels, report cards, trays |
| `--ink` | `#1d2340` | hematoxylin indigo: text, disclaimer bar, footer |
| `--indigo` | `#2b3a78` | primary buttons and links |
| `--eosin` | `#d4573a` | accent, italic headline line, "leukemic" |
| `--sage` | `#4f7a5a` | "normal" |
| `--ochre` | `#a86b12` | "refer to expert" |
| `--rule` / `--rule-strong` | `#d9cfbd` / `#bfb19a` | hairlines and dashed dividers |

Decision colours always come with a text label (the stamp, the scale key), never colour alone.

## Type

- Display: **Fraunces** (optical size and softness axes). Two-line headlines: a plain statement, then an italic payoff in eosin.
- Body: **IBM Plex Sans**.
- Labels and numbers: **IBM Plex Mono**, uppercase with wide tracking for specimen-style labels; tabular figures.

## Signature components

- **Slide-label brand:** a rectangular glass-slide label with a vertical specimen code ("CT-01") and the name.
- **Microscope field:** the hero shows a real C-NMC cell in a circular field of view with a reticle and a 10 µm scale bar.
- **Specimen tray:** the upload panel, with a dashed drop well and an offset paper shadow.
- **Lab-report strip:** key figures in a ruled four-column strip with mono numbers.
- **Protocol list:** steps as ruled rows with "STEP 01" mono numbering.
- **Report card + stamp:** results show a static, slightly rotated double-ruled stamp (Normal / Leukemic / Refer to expert),
  a ruled probability scale with the band zones and the score marked, and the specimen beside its heatmap in round fields.
- **Footer:** indigo block, one line of display copy, mono fine print.

## Rules we hold to

No purple gradients, Lucide or sparkle icons, three-card rows, bento grids, glass cards, glowing orbs, coloured card
stripes, Inter or Geist, emojis in headings, em dashes in copy, three-tier pricing, fake testimonials, dot-grid backgrounds,
hover animations, default titles or favicons, or placeholder text. `npm run lint` checks these automatically.
