# Time Series Figure Scorecard

Compliance tracking per the lab timeseries rubric (`~/.claude/skills/timeseries-figure/SKILL.md`).
**P** = Pass, **F** = Fail, **–** = Not applicable.

| Figure | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8a | 8b | 8c | 9 | 10 | 11 | 12 | 13 | 14 | 15 | 16 | 17 | 18 | 19 | Notes |
|--------|---|---|---|---|---|---|---|----|----|----|---|----|----|----|----|----|----|----|----|----|----|-------|
| figure_manual_worms_over_time.png | P | – | P | P | P | P | P | P | P | P | P | P | P | P | P | P | P | P | P | – | P | Paper tier, 300 DPI. Manual Monday Scene-1 series, mean±SEM over ≤8 slots. Line breaks across >14d gaps. Blur-onset (~Aug 2023) marked as context. |
| figure_tuesday_manual_series.png | P | – | P | P | – | P | P | P | P | P | P | P | P | P | P | P | P | P | P | – | P | Paper tier, 300 DPI. Manual Tuesday Scene-1 series (49 Tuesdays, 2021-2023), mean ± bootstrap 95% CI over ≤8 slots. Line breaks across >21d gaps. No event marker (series predates blur onset). |
| figure_tuesday_ai_saturation.png | P | – | – | P | – | P | P | P | – | – | P | – | – | P | P | P | P | P | P | – | P | Scatter (not time series): AI vs manual per-frame count, 229 pairs, 1:1 + OLS calibration. Date/aggregation/gap criteria N/A. Shows v2 saturation (38% recall, R²=0.20). |
| figure_tuesday_extended_series.png | P | – | P | P | – | P | P | P | P | P | P | P | P | P | P | P | P | P | P | – | P | Paper tier, 300 DPI. Manual anchors (blue, tight CI) + density-corrected v2 AI fill-ins (open vermillion, wide bootstrap 95% CI). Two series distinguished by color + marker fill. |
| figure_weekday_manual_series_2021_2024.png | P | – | P | P | P | P | P | P | P | P | P | P | P | P | P | P | P | P | P | – | P | Paper tier, 300 DPI. Combined manual Monday (blue circles, 128) + Tuesday (vermillion squares, 49) series, 2021-2024, both with same bootstrap 95% CI. Blur-onset marked (vertical label). Persistence across blur on both weekdays. |

## Event-annotation convention (this project)

Record here until a project constitution exists:

- **Orange/vermillion dashed vertical line** = camera image-quality boundary (e.g. ~Aug-2023 blur onset).
