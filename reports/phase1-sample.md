# Phase 1 sample report

Documents in: **50,000**, kept: **49,693** (99.4%).
Characters in: 237,047,786, kept: 235,981,802.

## Lengths

| measure | p5 | p25 | p50 | p75 | p95 |
|---|---|---|---|---|---|
| raw characters | 745.00 | 1,604.00 | 2,907.00 | 5,070.00 | 13,654.05 |
| clean characters | 751.00 | 1,608.00 | 2,912.00 | 5,077.00 | 13,669.60 |
| GPT-2 tokens (raw, from source) | 162.00 | 347.00 | 624.00 | 1,096.00 | 2,985.00 |

## Quality signals (raw)

| measure | p5 | p25 | p50 | p75 | p95 |
|---|---|---|---|---|---|
| mean_word_length | 4.51 | 4.88 | 5.14 | 5.42 | 5.92 |
| alphabetic_word_fraction | 0.93 | 0.97 | 0.98 | 0.99 | 1.00 |
| duplicate_line_fraction | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| top_bigram_char_fraction | 0.01 | 0.01 | 0.02 | 0.03 | 0.06 |

## What each filter did

| step | count |
|---|---|
| input | 50,000 |
| output | 49,693 |
| passed_quality | 49,704 |
| redacted:email | 1,482 |
| redacted:ipv4 | 310 |
| redacted:phone | 1,969 |
| rejected_quality | 296 |
| rule:bullet_lines | 80 |
| rule:ellipsis_lines | 132 |
| rule:mean_word_length | 2 |
| rule:non_alphabetic_words | 54 |
| rule:repeated_bigram | 5 |
| rule:symbol_to_word_ratio | 2 |
| rule:too_few_words | 25 |
| unique_exact | 49,697 |

## Top 15 source domains (raw)

| domain | documents |
|---|---|
| en.wikipedia.org | 332 |
| reference.com | 152 |
| britannica.com | 134 |
| enotes.com | 125 |
| teacherspayteachers.com | 92 |
| slideplayer.com | 77 |
| phys.org | 70 |
| bookrags.com | 65 |
| sciencedaily.com | 65 |
| eurekalert.org | 63 |
| newscientist.com | 62 |
| wordnik.com | 60 |
| redorbit.com | 57 |
| simple.wikipedia.org | 57 |
| en.wikisource.org | 54 |
