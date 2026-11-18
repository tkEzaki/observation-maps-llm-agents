# Statistical implementation notes

These notes describe the earlier analyses in `analysis/` and their frozen results.
They record the resampling units and aggregation conventions, and preserve the
history of numerical corrections used to produce the released results.

## Resampling and aggregation conventions

- Fourier uncertainty ellipses resample multinomial action counts within each
  offset. The pairwise comparisons use one value per physical field (48 fields).
  Noise-floor intervals use field-cluster resampling as described below.
- `acquisition_block_holdout` separates acquisition blocks of the same physical
  field between training and test sets. Four of the five grouping schemes keep
  physical fields together; acquisition-block holdout does not.
- The centres/intervals branches pool out-of-fold predictions before calculating
  log loss. The general `evaluate_grouped_schemes` routine averages fold-level
  values. The schemes have 41, 6, 6, 3 and 2 folds, so these summaries differ.
- In the earlier reference figures, `style.paired_ci` displays the mean plus or
  minus 1.96 standard errors across six seeds. Separate percentile-bootstrap
  intervals for contrasts are available in the numerical tables.
- Within-block phase reporting for both harmonics applies the amplitude gate
  R_m >= 0.15 and requires the uncertainty ellipse to exclude the origin. For
  block-averaged second harmonics, the ellipse gate is not applied.
- Fourier bootstrap counts are 200 for transmutation, 100 for stimulus_manifold
  and 2,000 for feature_sweep. The code default is 2,000 and the command-line
  option `--bootstrap-samples` can override it.
- The intervals coverage rule uses `nearest > 0.35`; the centres rule uses
  `nearest > 0.35 and k_eff < 5`. Their evaluation order differs, and only the
  centres branch has the additional threshold 0.25.
- Surrogate resampling uses 56 profile clusters, pooling rotations within a
  profile. These differ from the 756 profile-by-offset physical fields.
- Fold counts and assignments, baselines, nearest-neighbour distance, neighbour
  count and local label variance follow the definitions in the released code.

## Corrections included in the frozen results

The earlier `cluster_bootstrap_mean` routine resampled individual input entries.
For pairwise intervals, its 48 entries were already one per field. For the
noise-floor calculation, however, 144 entries contained three correlated values
per field. Resampling those entries separately underestimated interval widths.
The corrected routine accepts cluster identifiers and resamples complete fields.
It reports `n_values`, `n_clusters` and `cluster_unit`, with `n_fields` denoting
the actual number of fields. A dedicated random-number generator isolates this
resampling from other tests. The correction also applies to
`analyze_omap_control.py`.

The replay inference outputs were regenerated with the released implementation.
The regenerated files retain the three effect-test probabilities, aggregate
token counts and costs, and descriptive result fields. Historical cost accounting
uses input/output rates of USD 0.75/4.50 per million tokens, reproducing the
recorded session cost of USD 1.873224 within 1e-6. These are historical accounting
constants, not current service prices. The verdict label remains
`suggested_ns_at_0.05`. The corresponding observation-map analysis retains its
pairwise summary and result-description fields.

| Quantity | Earlier output | Corrected frozen output |
|---|---:|---:|
| Source main-effect p | 0.0768 | 0.0784 |
| Source-by-presentation interaction p | 0.0664 | 0.0716 |
| Presentation effect p | 0.0002 | 0.0002 |
| Centres-minus-intervals 95% CI | 0.221--0.363 | 0.220--0.364 |
| Moments-minus-centres 95% CI | 0.251--0.423 | 0.256--0.422 |
| Moments-minus-intervals 95% CI | 0.323--0.490 | 0.324--0.493 |
| Intervals-minus-moments delta a0 95% CI | 0.021--0.686 | 0.021--0.685 |
| Replay between-condition noise-floor 95% CI | 0.298--0.392 | 0.286--0.405 |
| Replay within-condition noise-floor 95% CI | 0.076--0.109 | 0.074--0.110 |
| Matched-information within-condition noise-floor 95% CI | 0.050--0.089 | 0.049--0.090 |

Mean values, effect sizes and decisions were unchanged. The two changed p-values
reflect different permutation random streams and both remain above the
prespecified 0.05 threshold. The presentation p-value is at the permutation
resolution limit. Wider noise-floor intervals reflect the corrected clustering.
The affected reference figures were regenerated from these frozen outputs.
The analysis uses fixed random seeds to reproduce the retained decision files.
