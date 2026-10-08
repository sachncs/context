## Strategies & Insights
[str-00001] helpful=0 harmful=0 :: Always tag the token as a BIO tag, never as a free-form string. Each token in the input gets exactly one output label, in the same order as the input tokens.
[str-00002] helpful=0 harmful=0 :: 'O' marks a token that is NOT part of any named entity. Use 'O' liberally for punctuation, determiners, and common verbs.
[str-00003] helpful=0 harmful=0 :: XBRL person names are tagged PER; place names are LOC; company / fund names are ORG. Family relationships inside a person name keep the same tag.
[str-00004] helpful=0 harmful=0 :: For multi-word entities, the first token carries the B- prefix; subsequent tokens carry the I- prefix. Never mix B- and I- across entities with different types.
[str-00005] helpful=0 harmful=0 :: Numeric tokens (digits, percent signs, currency) are usually O unless they are part of a Measure / Location identifier.

## Formulas & Calculations
[cal-00001] helpful=0 harmful=0 :: Count tokens: the number of input tokens must equal the number of output labels. If they don't match, pad with 'O' or trim.

## Code Snippets & Templates
[ctx-00001] helpful=0 harmful=0 :: Output format: one label per line, in token order. If the model writes 'token\\tlabel' on each line, strip the token and keep only the label.

## Common Mistakes To Avoid
[mis-00001] helpful=0 harmful=0 :: Confusing B- and I- tags is the most common FiNER error. Re-read multi-word entities and confirm the first token is B- while the rest are I-.
[mis-00002] helpful=0 harmful=0 :: Don't tag 'the', 'a', 'of', etc. as anything other than O. Common English determiners and prepositions are not entities.
[mis-00003] helpful=0 harmful=0 :: Don't confuse abbreviations (Inc., Corp., Ltd.) with ORG-only; they ARE part of the ORG entity.

## Problem-Solving Heuristics
[ps-00001] helpful=0 harmful=0 :: For ambiguous names, check whether the token is preceded by a known entity trigger ('Mr.', 'Inc.', 'Ltd.') before tagging.
[ps-00002] helpful=0 harmful=0 :: If a single word could be PER or LOC, look at the immediate context sentence for clue words ('said', 'based in', 'reported').

## Context Clues & Indicators
[cc-00001] helpful=0 harmful=0 :: Quotation marks often enclose an entity. The first quoted token may be PER or ORG.

## Others
[oth-00001] helpful=0 harmful=0 :: When in doubt, prefer O over a wrong label — partial credit is not awarded, so a wrong guess hurts the per-token F1.
