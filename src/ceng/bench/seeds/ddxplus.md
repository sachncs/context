## Strategies & Insights
[str-00001] helpful=0 harmful=0 :: The DDXPlus question lists PATIENT symptoms; the correct answer is the SINGLE most likely diagnosis out of four. Do not list multiple options.
[str-00002] helpful=0 harmful=0 :: Read ALL of the listed symptoms, including 'auxiliary' ones (smoking, alcohol, etc.) — they are the discriminator between similar options.
[str-00003] helpful=0 harmful=0 :: Demographics (age, sex) often pre-filter the option space before symptoms. Apply them first.
[str-00004] helpful=0 harmful=0 :: When two options look similar, look at the SYMPTOM-SPECIFIC features (e.g. fever, blood pressure) that distinguish them.

## Formulas & Calculations
[cal-00001] helpful=0 harmful=0 :: Probability scores in the question context (if present) are baseline priors; combine with symptom likelihoods, don't just pick the highest prior.
[cal-00002] helpful=0 harmful=0 :: When computing likelihood ratios, multiply conditional probabilities: P(D | S1, S2) ~ P(D) * P(S1 | D) * P(S2 | D).

## Code Snippets & Templates
[ctx-00001] helpful=0 harmful=0 :: Output format: a single integer (the option index). No words, no explanation, no list.

## Common Mistakes To Avoid
[mis-00001] helpful=0 harmful=0 :: Don't pick the diagnosis that's most common in the general population — pick the one most consistent with the SPECIFIC symptoms listed.
[mis-00002] helpful=0 harmful=0 :: Don't infer symptoms that aren't in the question. Base the answer on what's there.
[mis-00003] helpful=0 harmful=0 :: Don't output the option name; output the INDEX number. The grader compares numbers, not strings.

## Problem-Solving Heuristics
[ps-00001] helpful=0 harmful=0 :: For each option, ask: 'Does every listed symptom match this diagnosis, or are some unrelated?' The option that requires the fewest contortions is usually right.
[ps-00002] helpful=0 harmful=0 :: When in doubt, the option that mentions the most patient symptoms in its description is the stronger candidate.

## Context Clues & Indicators
[cc-00001] helpful=0 harmful=0 :: Lab values (blood pressure, heart rate, temperature) are diagnostic discriminators — read them carefully even if they look like noise.
[cc-00002] helpful=0 harmful=0 :: Family history is suggestive, not definitive. Don't overweight it relative to current symptoms.

## Others
[oth-00001] helpful=0 harmful=0 :: When in doubt, output the option that matches the most symptoms in the question. Random guessing has a 25% baseline; deliberate choice should beat that comfortably.
