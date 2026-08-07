# DEPRECATED - superseded by `analysis/figures_main`

The defective renderer scripts that used to live here have been removed from
the release; this directory now carries only `export_figure_data.py` (the
provenance of the shared `data/` files) and this defect record.

**Do not regenerate main figures from this directory.**

## What is wrong

`export_figure_data.py::export_prompts` selects the Figure 2 prompt excerpts by
`profile` alone and keeps the **first** record it meets for each
representation. Every profile is swept over many rotations of the field, so
that rule does not align the three representations on a physical field: it
picked `offset_index` **30** (`moments_m1_m3`), **4** (`centers_24_standard`)
and **29** (`intervals_24_decimal6`) of `profile == "unimodal_k9"`.

Figure 2 titles panel a *"Same field"* and panel b *"Three observations of the
same field"*. Those titles are false for the excerpts this exporter wrote:
the three excerpts are three **different rotations** of the field, not three
observations of one.

## What is affected

- `data/prompt_excerpts_unimodal.json` - the shared data file written by
  `export_prompts`.
- The Figure 2 outputs of this pipeline, and of an earlier legacy figure
  variant (not part of this release), which read the same data file and repeat
  the same selection defect. **Both contain a Figure 2 whose panel-b excerpts
  are three different rotations.**

Nothing else in this directory is known to be affected; the defect is confined
to the Figure 2 prompt-excerpt selection.

## The corrected pipeline

`analysis/figures_main` is the corrected pipeline. Its
`fig2_micro.py::_shared_field_records` reads the three serialisations from a
single `offset_index` of a single acquisition, and does not trust that index
label on its own: it also checks that the three records agree exactly on
`profile`, `concentration`, `offset_index` and `offset_radians`, that a derived
`physical_id` (a sha256 of that tuple) is identical across them, and that the
payloads really encode one field - the 24-bin mass parsed out of
`centers_24_standard` and out of `intervals_24_decimal6` must agree to the six
decimals both encoders print, and `moments_m1_m3` must be the moment summary of
that same mass. Panel a is drawn from that record's own encoded 24-bin mass.

`export_prompts` now carries an assert that fires if the three selected records
do not share one `offset_index`, so re-running this exporter fails loudly
instead of quietly rewriting `data/prompt_excerpts_unimodal.json` with
mismatched excerpts.
