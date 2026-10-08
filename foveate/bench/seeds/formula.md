## Strategies & Insights
[str-00001] helpful=0 harmful=0 :: When the question asks for a numeric answer, respond with ONLY the number. No commas, no currency symbols, no units.
[str-00002] helpful=0 harmful=0 :: When the question is about a ratio, percentage, or rate, the answer is usually a decimal (e.g. 0.15) or a percent (e.g. 15.0). Read the wording carefully.
[str-00003] helpful=0 harmful=0 :: When the question asks for a sum or total, add the numbers in the relevant period, not the cumulative.
[str-00004] helpful=0 harmful=0 :: Currency amounts in XBRL are usually in raw units (e.g. 1500000 not 1.5M). Convert only if the question explicitly asks.

## Formulas & Calculations
[cal-00001] helpful=0 harmful=0 :: Net income = revenue - expenses.
[cal-00002] helpful=0 harmful=0 :: EPS = net income / weighted average shares.
[cal-00003] helpful=0 harmful=0 :: ROA = net income / total assets.
[cal-00004] helpful=0 harmful=0 :: Current ratio = current assets / current liabilities.
[cal-00005] helpful=0 harmful=0 :: Year-over-year growth = (this year - last year) / last year.

## Code Snippets & Templates
[ctx-00001] helpful=0 harmful=0 :: Output format: a single number on a line by itself. No words, no $ sign, no comma separators.

## Common Mistakes To Avoid
[mis-00001] helpful=0 harmful=0 :: Don't return the formula. Return the result of evaluating it.
[mis-00002] helpful=0 harmful=0 :: Don't include units ($ , %, etc.) in the numeric answer.
[mis-00003] helpful=0 harmful=0 :: Don't compute the cumulative figure when the question asks for a single period.

## Problem-Solving Heuristics
[ps-00001] helpful=0 harmful=0 :: When the question gives a fiscal year or quarter, restrict the search to that period only.
[ps-00002] helpful=0 harmful=0 :: When the question asks for a ratio and the data is in two different scales, normalise before dividing.

## Context Clues & Indicators
[cc-00001] helpful=0 harmful=0 :: Words like 'revenue', 'sales', 'turnover' usually point to the top-line figure.
[cc-00002] helpful=0 harmful=0 :: 'Operating income' excludes non-operating items; 'gross income' excludes deductions.

## Others
[oth-00001] helpful=0 harmful=0 :: When in doubt, return a single integer or float with no thousand-separator. The grader uses a numeric tolerance, not a string match.
