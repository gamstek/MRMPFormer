# ???????

?????2026-09-08??? model/ ?????????? project_review/??

????????? Python AST ???????????????????????????????????????? PROJECT_UNDERSTANDING.md ????????????

## .idea/.gitignore

- ???184 bytes????text????8


## .idea/.name

- ???10 bytes????text????1


## .idea/csv-plugin.xml

- ???417 bytes????text????16


## .idea/inspectionProfiles/profiles_settings.xml

- ???179 bytes????text????6


## .idea/inspectionProfiles/Project_Default.xml

- ???1299 bytes????text????24


## .idea/misc.xml

- ???189 bytes????text????4


## .idea/modules.xml

- ???279 bytes????text????8


## .idea/QuanFormer.iml

- ???576 bytes????text????14


## .idea/vcs.xml

- ???172 bytes????text????6


## checkpoint/mrmpformer.pth

- ???367034138 bytes????checkpoint


## checkpoint/mrmpformer_special_v2.pth

- ???367034458 bytes????checkpoint


## checkpoint/mrmpformer_trainv2.pth

- ???367034394 bytes????checkpoint


## checkpoint/quanformer.pth

- ???322927374 bytes????checkpoint


## checkpoint/quanformerv2.pth

- ???322926705 bytes????checkpoint


## checkpoint/quanformerv3.pth

- ???322926641 bytes????checkpoint


## configs/coco_annotation.json

- ???2844 bytes????text????39

```json
{
  "mzmls": [
    "../data/mzml/20260715_shiyaoyuan_test/20260715_shiyaoyuan_test_1.mzML",
    "../data/mzml/20260715_shiyaoyuan_test/20260715_shiyaoyuan_test_2.mzML",
    "../data/mzml/traindata1/traindata1_1.mzML",
    "../data/mzml/traindata1/traindata1_2.mzML",
    "../data/mzml/traindata2/traindata2_1.mzML",
    "../data/mzml/traindata2/traindata2_2.mzML",
    "../data/mzml/traindata2/traindata2_3.mzML",
    "../data/mzml/traindata2/traindata2_4.mzML",
    "../data/mzml/traindata2/traindata2_5.mzML",
    "../data/mzml/traindata2/traindata2_6.mzML"
  ],
  "labels": "auto",
  "output_dir": "../data/coco/merged",
  "val_stems": [],
  "val_ratio": 0.3,
  "smooth_sigma": 0.0,
  "work_dir": null,
  "force": false,
  "sample_map": null,
  "include_unlabeled": true,
  "qc_label_rt_tol": 1.0,
  "label_dir": "../data/label",
  "qc_root": "../output/QC"
}
```

## configs/coco_annotation_traindata3.json

- ???7595 bytes????text????149

```json
{
  "mzmls": [
    "../data/mzml/traindata3/traindata3-1.mzML",
    "../data/mzml/traindata3/traindata3-2.mzML",
    "../data/mzml/traindata3/traindata3-3.mzML",
    "../data/mzml/traindata3/traindata3-4.mzML",
    "../data/mzml/traindata3/traindata3-5.mzML",
    "../data/mzml/traindata3/traindata3-6.mzML",
    "../data/mzml/traindata3/traindata3-7.mzML",
    "../data/mzml/traindata3/traindata3-8.mzML",
    "../data/mzml/traindata3/traindata3-9.mzML",
    "../data/mzml/traindata3/traindata3-10.mzML",
    "../data/mzml/traindata3/traindata3-11.mzML",
    "../data/mzml/traindata3/traindata3-12.mzML",
    "../data/mzml/traindata3/traindata3-13.mzML",
    "../data/mzml/traindata3/traindata3-14.mzML",
    "../data/mzml/traindata3/traindata3-15.mzML",
    "../data/mzml/traindata3/traindata3-16.mzML",
    "../data/mzml/traindata3/traindata3-17.mzML",
    "../data/mzml/traindata3/traindata3-18.mzML",
    "../data/mzml/traindata3/traindata3-19.mzML",
    "../data/mzml/traindata3/traindata3-20.mzML",
    "../data/mzml/traindata3/traindata3-21.mzML",
    "../data/mzml/traindata3/traindata3-22.mzML",
    "../data/mzml/traindata3/traindata3-23.mzML",
    "../data/mzml/traindata3/traindata3-24.mzML",
    "../data/mzml/traindata3/traindata3-25.mzML",
    "../data/mzml/traindata3/traindata3-26.mzML",
    "../data/mzml/traindata3/traindata3-27.mzML",
    "../data/mzml/traindata3/traindata3-28.mzML",
    "../data/mzml/traindata3/traindata3-29.mzML",
    "../data/mzml/traindata3/traindata3-30.mzML",
    "../data/mzml/traindata3/traindata3-31.mzML",
    "../data/mzml/traindata3/traindata3-32.mzML",
    "../data/mzml/traindata3/traindata3-33.mzML",
    "../data/mzml/traindata3/traindata3-34.mzML",
    "../data/mzml/traindata3/traindata3-35.mzML",
    "../data/mzml/traindata3/traindata3-36.mzML",
    "../data/mzml/traindata3/traindata3-37.mzML",
    "../data/mzml/traindata3/traindata3-38.mzML",
    "../data/mzml/traindata3/traindata3-39.mzML",
    "../data/mzml/traindata3/traindata3-40.mzML",
    "../data/mzml/traindata3/traindata3-41.mzML",
    "../data/mzml/traindata3/traindata3-42.mzML",
    "../data/mzml/traindata3/traindata3-43.mzML",
    "../data/mzml/traindata3/traindata3-44.mzML",
    "../data/mzml/traindata3/traindata3-45.mzML",
    "../data/mzml/traindata3/traindata3-46.mzML",
    "../data/mzml/traindata3/traindata3-47.mzML",
    "../data/mzml/traindata3/traindata3-48.mzML",
    "../data/mzml/traindata3/traindata3-49.mzML",
    "../data/mzml/traindata3/traindata3-50.mzML",
    "../data/mzml/traindata3/traindata3-51.mzML",
    "../data/mzml/traindata3/traindata3-52.mzML",
    "../data/mzml/traindata3/traindata3-53.mzML",
    "../data/mzml/traindata3/traindata3-54.mzML",
    "../data/mzml/traindata3/traindata3-55.mzML",
    "../data/mzml/traindata3/traindata3-56.mzML",
    "../data/mzml/traindata3/traindata3-57.mzML",
    "../data/mzml/traindata3/traindata3-58.mzML",
    "../data/mzml/traindata3/traindata3-59.mzML",
    "../data/mzml/traindata3/traindata3-60.mzML",
    "../data/mzml/traindata3/traindata3-61.mzML",
    "../data/mzml/traindata3/traindata3-62.mzML",
    "../data/mzml/traindata3/traindata3-63.mzML",
    "../data/mzml/traindata3/traindata3-64.mzML",
    "../data/mzml/traindata3/traindata3-65.mzML",
    "../data/mzml/traindata3/traindata3-66.mzML",
    "../data/mzml/traindata3/traindata3-67.mzML",
    "../data/mzml/traindata3/traindata3-68.mzML",
    "../data/mzml/traindata3/traindata3-69.mzML",
    "../data/mzml/traindata3/traindata3-70.mzML",
    "../data/mzml/traindata3/traindata3-71.mzML",
    "../data/mzml/traindata3/traindata3-72.mzML",
    "../data/mzml/traindata3/traindata3-73.mzML",
    "../data/mzml/traindata3/traindata3-74.mzML",
    "../data/mzml/traindata3/traindata3-75.mzML",
    "../data/mzml/traindata3/traindata3-76.mzML",
    "../data/mzml/traindata3/traindata3-77.mzML",
    "../data/mzml/traindata3/traindata3-78.mzML",
    "../data/mzml/traindata3/traindata3-79.mzML",
    "../data/mzml/traindata3/traindata3-80.mzML",
    "../data/mzml/traindata3/traindata3-81.mzML",
    "../data/mzml/traindata3/traindata3-82.mzML",
    "../data/mzml/traindata3/traindata3-83.mzML",
    "../data/mzml/traindata3/traindata3-84.mzML",
    "../data/mzml/traindata3/traindata3-85.mzML",
    "../data/mzml/traindata3/traindata3-86.mzML",
    "../data/mzml/traindata3/traindata3-87.mzML",
    "../data/mzml/traindata3/traindata3-88.mzML",
    "../data/mzml/traindata3/traindata3-89.mzML",
    "../data/mzml/traindata3/traindata3-90.mzML",
    "../data/mzml/traindata3/traindata3-BLANK.mzML",
    "../data/mzml/traindata3/traindata3-BLANK2.mzML",
    "../data/mzml/traindata3/traindata3-BLANK3.mzML",
    "../data/mzml/traindata3/traindata3-BLANK4.mzML",
    "../data/mzml/traindata3/traindata3-BLANK5.mzML",
    "../data/mzml/traindata3/traindata3-BLANK6.mzML",
    "../data/mzml/traindata3/traindata3-BLANK7.mzML",
    "../data/mzml/traindata3/traindata3-BLANK8.mzML",
    "../data/mzml/traindata3/traindata3-BLANK9.mzML",
    "../data/mzml/traindata3/traindata3-QC.mzML",
    "../data/mzml/traindata3/traindata3-QC2.mzML",
    "../data/mzml/traindata3/traindata3-QC3.mzML",
    "../data/mzml/traindata3/traindata3-QC4.mzML",
    "../data/mzml/traindata3/traindata3-QC5.mzML",
    "../data/mzml/traindata3/traindata3-QC6.mzML",
    "../data/mzml/traindata3/traindata3-QC7.mzML",
    "../data/mzml/traindata3/traindata3-QC8.mzML",
    "../data/mzml/traindata3/traindata3-QC9.mzML",
    "../data/mzml/traindata3/traindata3-QC10.mzML"
  ],
  "labels": [
    "../data/label/traindata3.xlsx"
  ],
  "output_dir": "../data/coco/traindata3",
  "val_stems": [
    "traindata3-QC",
    "traindata3-QC2",
    "traindata3-QC3",
    "traindata3-QC4",
    "traindata3-QC5",
    "traindata3-QC6",
    "traindata3-QC7",
    "traindata3-QC8",
    "traindata3-QC9",
    "traindata3-QC10"
  ],
  "val_ratio": 0.0,
  "smooth_sigma": 0.0,
  "work_dir": null,
  "force": false,
  "sample_map": null,
  "include_unlabeled": true,
  "qc_label_rt_tol": 1.0,
  "label_dir": "../data/label",
  "qc_root": "../output/QC"
}
```

## configs/coco_annotation_traindatav1.json

- ???8241 bytes????text????161

```json
{
  "mzmls": [
    "../data/mzml/traindata1/traindata1_1.mzML",
    "../data/mzml/traindata1/traindata1_2.mzML",
    "../data/mzml/traindata2/traindata2_1.mzML",
    "../data/mzml/traindata2/traindata2_2.mzML",
    "../data/mzml/traindata2/traindata2_3.mzML",
    "../data/mzml/traindata2/traindata2_4.mzML",
    "../data/mzml/traindata2/traindata2_5.mzML",
    "../data/mzml/traindata2/traindata2_6.mzML",
    "../data/mzml/traindata3/traindata3-1.mzML",
    "../data/mzml/traindata3/traindata3-2.mzML",
    "../data/mzml/traindata3/traindata3-3.mzML",
    "../data/mzml/traindata3/traindata3-4.mzML",
    "../data/mzml/traindata3/traindata3-5.mzML",
    "../data/mzml/traindata3/traindata3-6.mzML",
    "../data/mzml/traindata3/traindata3-7.mzML",
    "../data/mzml/traindata3/traindata3-8.mzML",
    "../data/mzml/traindata3/traindata3-9.mzML",
    "../data/mzml/traindata3/traindata3-10.mzML",
    "../data/mzml/traindata3/traindata3-11.mzML",
    "../data/mzml/traindata3/traindata3-12.mzML",
    "../data/mzml/traindata3/traindata3-13.mzML",
    "../data/mzml/traindata3/traindata3-14.mzML",
    "../data/mzml/traindata3/traindata3-15.mzML",
    "../data/mzml/traindata3/traindata3-16.mzML",
    "../data/mzml/traindata3/traindata3-17.mzML",
    "../data/mzml/traindata3/traindata3-18.mzML",
    "../data/mzml/traindata3/traindata3-19.mzML",
    "../data/mzml/traindata3/traindata3-20.mzML",
    "../data/mzml/traindata3/traindata3-21.mzML",
    "../data/mzml/traindata3/traindata3-22.mzML",
    "../data/mzml/traindata3/traindata3-23.mzML",
    "../data/mzml/traindata3/traindata3-24.mzML",
    "../data/mzml/traindata3/traindata3-25.mzML",
    "../data/mzml/traindata3/traindata3-26.mzML",
    "../data/mzml/traindata3/traindata3-27.mzML",
    "../data/mzml/traindata3/traindata3-28.mzML",
    "../data/mzml/traindata3/traindata3-29.mzML",
    "../data/mzml/traindata3/traindata3-30.mzML",
    "../data/mzml/traindata3/traindata3-31.mzML",
    "../data/mzml/traindata3/traindata3-32.mzML",
    "../data/mzml/traindata3/traindata3-33.mzML",
    "../data/mzml/traindata3/traindata3-34.mzML",
    "../data/mzml/traindata3/traindata3-35.mzML",
    "../data/mzml/traindata3/traindata3-36.mzML",
    "../data/mzml/traindata3/traindata3-37.mzML",
    "../data/mzml/traindata3/traindata3-38.mzML",
    "../data/mzml/traindata3/traindata3-39.mzML",
    "../data/mzml/traindata3/traindata3-40.mzML",
    "../data/mzml/traindata3/traindata3-41.mzML",
    "../data/mzml/traindata3/traindata3-42.mzML",
    "../data/mzml/traindata3/traindata3-43.mzML",
    "../data/mzml/traindata3/traindata3-44.mzML",
    "../data/mzml/traindata3/traindata3-45.mzML",
    "../data/mzml/traindata3/traindata3-46.mzML",
    "../data/mzml/traindata3/traindata3-47.mzML",
    "../data/mzml/traindata3/traindata3-48.mzML",
    "../data/mzml/traindata3/traindata3-49.mzML",
    "../data/mzml/traindata3/traindata3-50.mzML",
    "../data/mzml/traindata3/traindata3-51.mzML",
    "../data/mzml/traindata3/traindata3-52.mzML",
    "../data/mzml/traindata3/traindata3-53.mzML",
    "../data/mzml/traindata3/traindata3-54.mzML",
    "../data/mzml/traindata3/traindata3-55.mzML",
    "../data/mzml/traindata3/traindata3-56.mzML",
    "../data/mzml/traindata3/traindata3-57.mzML",
    "../data/mzml/traindata3/traindata3-58.mzML",
    "../data/mzml/traindata3/traindata3-59.mzML",
    "../data/mzml/traindata3/traindata3-60.mzML",
    "../data/mzml/traindata3/traindata3-61.mzML",
    "../data/mzml/traindata3/traindata3-62.mzML",
    "../data/mzml/traindata3/traindata3-63.mzML",
    "../data/mzml/traindata3/traindata3-64.mzML",
    "../data/mzml/traindata3/traindata3-65.mzML",
    "../data/mzml/traindata3/traindata3-66.mzML",
    "../data/mzml/traindata3/traindata3-67.mzML",
    "../data/mzml/traindata3/traindata3-68.mzML",
    "../data/mzml/traindata3/traindata3-69.mzML",
    "../data/mzml/traindata3/traindata3-70.mzML",
    "../data/mzml/traindata3/traindata3-71.mzML",
    "../data/mzml/traindata3/traindata3-72.mzML",
    "../data/mzml/traindata3/traindata3-73.mzML",
    "../data/mzml/traindata3/traindata3-74.mzML",
    "../data/mzml/traindata3/traindata3-75.mzML",
    "../data/mzml/traindata3/traindata3-76.mzML",
    "../data/mzml/traindata3/traindata3-77.mzML",
    "../data/mzml/traindata3/traindata3-78.mzML",
    "../data/mzml/traindata3/traindata3-79.mzML",
    "../data/mzml/traindata3/traindata3-80.mzML",
    "../data/mzml/traindata3/traindata3-81.mzML",
    "../data/mzml/traindata3/traindata3-82.mzML",
    "../data/mzml/traindata3/traindata3-83.mzML",
    "../data/mzml/traindata3/traindata3-84.mzML",
    "../data/mzml/traindata3/traindata3-85.mzML",
    "../data/mzml/traindata3/traindata3-86.mzML",
    "../data/mzml/traindata3/traindata3-87.mzML",
    "../data/mzml/traindata3/traindata3-88.mzML",
    "../data/mzml/traindata3/traindata3-89.mzML",
    "../data/mzml/traindata3/traindata3-90.mzML",
    "../data/mzml/traindata3/traindata3-BLANK.mzML",
    "../data/mzml/traindata3/traindata3-BLANK2.mzML",
    "../data/mzml/traindata3/traindata3-BLANK3.mzML",
    "../data/mzml/traindata3/traindata3-BLANK4.mzML",
    "../data/mzml/traindata3/traindata3-BLANK5.mzML",
    "../data/mzml/traindata3/traindata3-BLANK6.mzML",
    "../data/mzml/traindata3/traindata3-BLANK7.mzML",
    "../data/mzml/traindata3/traindata3-BLANK8.mzML",
    "../data/mzml/traindata3/traindata3-BLANK9.mzML",
    "../data/mzml/traindata3/traindata3-QC.mzML",
    "../data/mzml/traindata3/traindata3-QC2.mzML",
    "../data/mzml/traindata3/traindata3-QC3.mzML",
    "../data/mzml/traindata3/traindata3-QC4.mzML",
    "../data/mzml/traindata3/traindata3-QC5.mzML",
    "../data/mzml/traindata3/traindata3-QC6.mzML",
    "../data/mzml/traindata3/traindata3-QC7.mzML",
    "../data/mzml/traindata3/traindata3-QC8.mzML",
    "../data/mzml/traindata3/traindata3-QC9.mzML",
    "../data/mzml/traindata3/traindata3-QC10.mzML"
  ],
  "labels": [
    "../data/label/traindata1.xlsx",
    "../data/label/traindata2.xlsx",
    "../data/label/traindata3.xlsx"
  ],
  "output_dir": "../data/coco/traindatav1",
  "val_stems": [
    "traindata3-QC",
    "traindata3-QC2",
    "traindata3-QC3",
    "traindata3-QC4",
    "traindata3-QC5",
    "traindata3-QC6",
    "traindata3-QC7",
    "traindata3-QC8",
    "traindata3-QC9",
    "traindata3-QC10"
  ],
  "val_ratio": 0.0,
  "smooth_sigma": 0.0,
  "work_dir": null,
  "force": false,
  "sample_map": null,
  "include_unlabeled": true,
  "qc_label_rt_tol": 1.0,
  "label_dir": "../data/label",
  "qc_root": "../output/QC"
}
```

## configs/coco_annotation_traindatav2.json

- ???368 bytes????text????11

```json
{
  "sim_root": "../data/train",
  "output_dir": "../data/coco/traindatav2",
  "val_samples": 10,
  "window_half_min": 1.0,
  "multi_peak_margin_min": 0.15,
  "seed": 61002,
  "limit_samples": 0,
  "force": false
}
```

## configs/evaluation_baseline.json

- ???1860 bytes????text????29

```json
{
  "labels": "../data/label/20260715_shiyaoyuan_test.xlsx",
  "model": "checkpoint/quanformer.pth",
  "output_dir": "../output/evaluation/baseline",
  "run_inference": 1,
  "mzmls": [
    "../data/test/mzml/20260715_shiyaoyuan_test_1.mzML",
    "../data/test/mzml/20260715_shiyaoyuan_test_2.mzML"
  ],
  "prediction_csvs": [],
  "feature_csvs": [],
  "threshold": 0.9,
  "tolerance": 0.1,
  "quant_tolerance": 0.2,
  "smooth_sigma": 0.8,
  "qc_label_rt_tol": 1.0
}
```

## configs/inference_pipeline.json

- ???7774 bytes????text????115

```json
{
  "mode": "pipeline",
  "model": "checkpoint/mrmpformer.pth",
  "labels": "../data/label/test1.xlsx",
  "qc_label_rt_tol": 1.0,
  "threshold": 0.01,
  "integration_method": "linear",
  "smooth_sigma": 0.8,
  "output_dir": null,
  "exp_name": null,
  "mzml": "../data/mzml/test1",
  "batch_dir": null,
  "plot": true,
  "plot_style": "xic",
  "pipeline_min_max_intensity": 1000.0,
  "pipeline_min_chrom_points": 10,
  "snr_min": 10.0,
  "snr_gaussian_sigma": 0.8,
  "snr_min_noise_points": 5,
  "post_output_name": "prediction_refined.csv",
  "post_small_peak_rt_tol": 0.25,
  "post_min_secondary_ratio": 0.04,
  "post_noise_barrier_ratio": 0.25,
  "post_secondary_roi_global_gate_relax_frac": 0.055,
  "post_edge_max_span_min": 0.24,
  "post_edge_noise_percentile": 55.0,
  "post_small_boundary_pad": 0.08,
  "post_boundary_posterior_lookahead": 0,
  "post_boundary_posterior_mean_scale": 1.25,
  "post_disable_valley_fallback": false,
  "post_disable_lr_repredict_on_small_fail": false,
  "post_min_confidence": 0.99,
  "post_min_snr": 10.0,
  "post_small_noise_window_half": 0.3,
  "post_main_boundary_noise_percentile": 20.0,
  "post_plot_sigma": 0.8,
  "post_plot_dir_name": "refined_plots",
  "post_edge_noise_stop_mode": "roi_bottom_decile_mean",
  "post_edge_flat_triplet_step_frac": 0.01,
  "post_refine_width_max_expand_vs_pred": 1.08,
  "post_refine_width_max_frac_of_roi": 0.45,
  "post_enable_small_peak_rt_gate": false,
  "scan_baseline_percentile": 25.0,
  "scan_baseline_mode": "global_percentile",
  "scan_min_peak_ratio": 0.04,
  "scan_prominence_ratio": 0.055,
  "scan_min_prominence_abs": 0.0,
  "scan_min_peak_gap_points": 3,
  "scan_min_peak_width_min": 0.1,
  "scan_void_time_min": 0.5,
  "scan_max_peaks_per_channel": 50,
  "scan_init_half_width_min": 0.05,
  "scan_boundary_posterior_lookahead": 5,
  "scan_boundary_posterior_mean_scale": 1.25,
  "scan_edge_noise_stop_mode": "stable_tail_mean",
  "scan_edge_max_span_min": 1.0,
  "scan_min_snr": 10.0,
  "scan_min_peak_span_points": 5,
  "scan_min_area": 0.0,
  "scan_window_half_min": 1.0,
  "keep_windows": false,
  "no_plots": false,
  "no_timing": false,
  "no_report": false,
  "save_snr_jpeg": false,
  "verbose": false,
  "quiet": false
}
```

## configs/massnova.json

- ???6330 bytes????text????106

```json
{
  "mode": "massnova",
  "model": "checkpoint/mrmpformer.pth",
  "labels": null,
  "qc_label_rt_tol": 1.0,
  "integration_method": "linear",
  "plot": true,
  "threshold": 0.99,
  "smooth_sigma": 0.8,
  "exp_name": null,
  "output_dir": null,
  "mzml": "../data/mzml/test1",
  "batch_dir": null,
  "pipeline_min_max_intensity": 1000.0,
  "pipeline_min_chrom_points": 10,
  "scan_baseline_percentile": 25.0,
  "scan_baseline_mode": "global_percentile",
  "scan_min_peak_ratio": 0.04,
  "scan_prominence_ratio": 0.055,
  "scan_min_prominence_abs": 0.0,
  "scan_min_peak_gap_points": 3,
  "scan_min_peak_width_min": 0.1,
  "scan_void_time_min": 0.5,
  "scan_max_peaks_per_channel": 50,
  "scan_init_half_width_min": 0.05,
  "scan_boundary_posterior_lookahead": 5,
  "scan_boundary_posterior_mean_scale": 1.25,
  "scan_edge_noise_stop_mode": "stable_tail_mean",
  "scan_edge_max_span_min": 1.0,
  "scan_min_snr": 10.0,
  "scan_min_peak_span_points": 5,
  "scan_min_area": 0.0,
  "scan_window_half_min": 1.0,
  "scan_dup_apex_tol": 0.2,
  "scan_width_fuse_ratio": 1.5,
  "snr_min": 10.0,
  "snr_gaussian_sigma": 0.8,
  "snr_min_noise_points": 5,
  "post_output_name": "prediction_refined.csv",
  "post_small_peak_rt_tol": 0.25,
  "post_min_secondary_ratio": 0.04,
  "post_noise_barrier_ratio": 0.45,
  "post_secondary_roi_global_gate_relax_frac": 0.055,
  "post_edge_max_span_min": 0.24,
  "post_edge_noise_percentile": 55.0,
  "post_small_boundary_pad": 0.08,
  "post_boundary_posterior_lookahead": 0,
  "post_boundary_posterior_mean_scale": 1.25,
  "post_disable_valley_fallback": false,
  "post_disable_lr_repredict_on_small_fail": false,
  "post_min_confidence": 0.99,
  "post_min_snr": 10.0,
  "post_small_noise_window_half": 0.3,
  "post_main_boundary_noise_percentile": 20.0,
  "post_plot_sigma": 0.8,
  "post_plot_dir_name": "refined_plots",
  "post_edge_noise_stop_mode": "roi_bottom_decile_mean",
  "post_edge_flat_triplet_step_frac": 0.01,
  "post_refine_width_max_expand_vs_pred": 1.08,
  "post_refine_width_max_frac_of_roi": 0.45,
  "post_enable_small_peak_rt_gate": false,
  "no_timing": false,
  "no_report": false,
  "save_snr_jpeg": false,
  "verbose": false,
  "quiet": false,
  "keep_windows": false,
  "no_plots": false
}
```

## configs/mrmpformer_special_v1.json

- ???3118 bytes????text????96

```json
{
  "model": "mrmpformer_special",
  "backbone": "resnet50",
  "dilation": false,
  "position_embedding": "sine",
  "enc_layers": 1,
  "dec_layers": 3,
  "num_queries": 3,
  "hidden_dim": 256,
  "nheads": 8,
  "dim_feedforward": 2048,
  "dropout": 0.1,
  "pre_norm": false,
  "masks": false,
  "aux_loss": true,
  "iou_type": "ciou",
  "set_cost_class": 1,
  "set_cost_bbox": 5,
  "set_cost_iou": 2,
  "mask_loss_coef": 1,
  "dice_loss_coef": 1,
  "bbox_loss_coef": 5,
  "iou_loss_coef": 2,
  "eos_coef": 0.1,
  "num_fdr_bins": 33,
  "fdr_bin_power": 2.0,
  "fdr_bin_values": null,
  "fdr_scale_mode": "roi_width",
  "fdr_layer_weights": [
    0.5,
    0.7,
    1.0
  ],
  "fdr_layer_sigmas": null,
  "fdr_layer_progress": null,
  "fdr_cascade": false,
  "fdr_loss_coef": 2.0,
  "fdr_min_width": 0.0001,
  "detach_boundary_feedback": false,
  "special_cls": "qfl",
  "qfl_gamma": 2.0,
  "log_width_enabled": true,
  "log_width_center_exp": 0.5,
  "log_width_ref": 0.1273,
  "classification_loss": "focal",
  "focal_alpha": 0.25,
  "focal_gamma": 2.0,
  "cls_loss_coef": 1.0,
  "aux_class_loss": true,
  "aux_class_loss_coef": 1.0,
  "dynamic_l1_enabled": false,
  "dynamic_l1_eps": 1e-06,
  "dynamic_l1_lambda_w": 1.0,
  "dynamic_l1_lambda_h": 1.0,
  "center_weight_clip": null,
  "normalize_dynamic_weights": false,
  "pw_ciou_enabled": true,
  "pw_ciou_weight_mode": "ratio",
  "pw_ciou_eps": 1e-06,
  "pw_ciou_weight_clip": 3.0,
  "pw_ciou_mean_width": 0.1273,
  "recall_loss_enabled": false,
  "dataset_file": "coco",
  "coco_path": "../data/coco/traindatav1",
  "remove_difficult": false,
  "device": "auto",
  "seed": 42,
  "num_workers": 4,
  "world_size": 1,
  "dist_url": "env://",
  "lr": 0.0001,
  "lr_backbone": 1e-05,
  "batch_size": 16,
  "weight_decay": 0.0001,
  "epochs": 30,
  "lr_drop": 20,
  "clip_max_norm": 0.1,
  "amp": true,
  "tf32": true,
  "cudnn_benchmark": true,
  "output_dir": "../output/train/mrmpformer_special_v1",
  "resume": null,
  "start_epoch": 0,
  "reset_optimizer": false,
  "frozen_weights": null,
  "coco_panoptic_path": null,
  "eval": false
}
```

## configs/mrmpformer_special_v2.json

- ???2874 bytes????text????95

```json
{
  "model": "mrmpformer_special",
  "backbone": "resnet50",
  "dilation": false,
  "position_embedding": "sine",
  "enc_layers": 1,
  "dec_layers": 3,
  "num_queries": 3,
  "hidden_dim": 256,
  "nheads": 8,
  "dim_feedforward": 2048,
  "dropout": 0.1,
  "pre_norm": false,
  "masks": false,
  "aux_loss": true,
  "iou_type": "ciou",
  "set_cost_class": 1,
  "set_cost_bbox": 5,
  "set_cost_iou": 2,
  "mask_loss_coef": 1,
  "dice_loss_coef": 1,
  "bbox_loss_coef": 5,
  "iou_loss_coef": 2,
  "eos_coef": 0.1,
  "num_fdr_bins": 33,
  "fdr_bin_power": 2.0,
  "fdr_bin_values": null,
  "fdr_scale_mode": "roi_width",
  "fdr_layer_weights": [
    0.5,
    0.7,
    1.0
  ],
  "fdr_layer_sigmas": null,
  "fdr_layer_progress": null,
  "fdr_cascade": false,
  "fdr_loss_coef": 2.0,
  "fdr_min_width": 0.0001,
  "detach_boundary_feedback": false,
  "special_cls": "focal",
  "qfl_gamma": 2.0,
  "log_width_enabled": true,
  "log_width_center_exp": 0.5,
  "log_width_ref": 0.1273,
  "classification_loss": "focal",
  "focal_alpha": 0.25,
  "focal_gamma": 2.0,
  "cls_loss_coef": 1.0,
  "aux_class_loss": true,
  "aux_class_loss_coef": 1.0,
  "dynamic_l1_enabled": false,
  "dynamic_l1_eps": 1e-06,
  "dynamic_l1_lambda_w": 1.0,
  "dynamic_l1_lambda_h": 1.0,
  "center_weight_clip": null,
  "normalize_dynamic_weights": false,
  "pw_ciou_enabled": true,
  "pw_ciou_weight_mode": "ratio",
  "pw_ciou_eps": 1e-06,
  "pw_ciou_weight_clip": 3.0,
  "pw_ciou_mean_width": 0.1273,
  "recall_loss_enabled": false,
  "dataset_file": "coco",
  "coco_path": "../data/coco/traindatav1",
  "remove_difficult": false,
  "device": "auto",
  "seed": 42,
  "num_workers": 4,
  "world_size": 1,
  "dist_url": "env://",
  "lr": 0.0001,
  "lr_backbone": 1e-05,
  "batch_size": 16,
  "weight_decay": 0.0001,
  "epochs": 22,
  "lr_drop": 15,
  "clip_max_norm": 0.1,
  "amp": true,
  "tf32": true,
  "cudnn_benchmark": true,
  "output_dir": "../output/train/mrmpformer_special_v2",
  "resume": "../output/train/mrmpformer_special_v1/checkpoint.pth",
  "start_epoch": 0,
  "reset_optimizer": true,
  "frozen_weights": null,
  "coco_panoptic_path": null,
  "eval": false
}
```

## configs/mrmpformer_traindatav2.json

- ???4211 bytes????text????102

```json
{
  "model": "mrmpformer_v1",
  "backbone": "resnet50",
  "dilation": false,
  "position_embedding": "sine",
  "enc_layers": 1,
  "dec_layers": 3,
  "num_queries": 3,
  "hidden_dim": 256,
  "nheads": 8,
  "dim_feedforward": 2048,
  "dropout": 0.1,
  "pre_norm": false,
  "masks": false,
  "aux_loss": true,
  "iou_type": "ciou",
  "set_cost_class": 1,
  "set_cost_bbox": 5,
  "set_cost_iou": 2,
  "mask_loss_coef": 1,
  "dice_loss_coef": 1,
  "bbox_loss_coef": 5,
  "iou_loss_coef": 2,
  "eos_coef": 0.1,
  "num_fdr_bins": 33,
  "fdr_bin_power": 2.0,
  "fdr_bin_values": null,
  "fdr_scale_mode": "initial_box_width",
  "fdr_layer_weights": [
    0.5,
    0.7,
    1.0
  ],
  "fdr_loss_coef": 2.0,
  "fdr_min_width": 0.0001,
  "detach_boundary_feedback": false,
  "classification_loss": "focal",
  "focal_alpha": 0.25,
  "focal_gamma": 2.0,
  "cls_loss_coef": 1.0,
  "aux_class_loss": true,
  "aux_class_loss_coef": 1.0,
  "dynamic_l1_enabled": true,
  "dynamic_l1_eps": 1e-06,
  "dynamic_l1_lambda_w": 1.0,
  "dynamic_l1_lambda_h": 1.0,
  "center_weight_clip": null,
  "normalize_dynamic_weights": false,
  "pw_ciou_enabled": true,
  "pw_ciou_weight_mode": "ratio",
  "pw_ciou_eps": 1e-06,
  "pw_ciou_weight_clip": null,
  "pw_ciou_mean_width": 0.3958,
  "recall_loss_enabled": false,
  "dataset_file": "coco",
  "coco_path": "../data/coco/traindatav2",
  "remove_difficult": false,
  "device": "auto",
  "seed": 42,
  "num_workers": 4,
  "world_size": 1,
  "dist_url": "env://",
  "lr": 0.0001,
  "lr_backbone": 1e-05,
  "batch_size": 16,
  "weight_decay": 0.0001,
  "epochs": 30,
  "lr_drop": 20,
  "clip_max_norm": 0.1,
  "amp": true,
  "tf32": true,
  "cudnn_benchmark": true,
  "output_dir": "../output/train/mrmpformer_traindatav2",
  "resume": null,
  "start_epoch": 0,
  "reset_optimizer": false,
  "frozen_weights": null,
  "coco_panoptic_path": null,
  "eval": false
}
```

## configs/mrmpformer_trainv2_ft_v1.json

- ???2944 bytes????text????88

```json
{
  "model": "mrmpformer_v1",
  "backbone": "resnet50",
  "dilation": false,
  "position_embedding": "sine",
  "enc_layers": 1,
  "dec_layers": 3,
  "num_queries": 3,
  "hidden_dim": 256,
  "nheads": 8,
  "dim_feedforward": 2048,
  "dropout": 0.1,
  "pre_norm": false,
  "masks": false,
  "aux_loss": true,
  "iou_type": "ciou",
  "set_cost_class": 1,
  "set_cost_bbox": 5,
  "set_cost_iou": 2,
  "mask_loss_coef": 1,
  "dice_loss_coef": 1,
  "bbox_loss_coef": 5,
  "iou_loss_coef": 2,
  "eos_coef": 0.1,
  "num_fdr_bins": 33,
  "fdr_bin_power": 2.0,
  "fdr_bin_values": null,
  "fdr_scale_mode": "initial_box_width",
  "fdr_layer_weights": [
    0.5,
    0.7,
    1.0
  ],
  "fdr_loss_coef": 2.0,
  "fdr_min_width": 0.0001,
  "detach_boundary_feedback": false,
  "classification_loss": "focal",
  "focal_alpha": 0.25,
  "focal_gamma": 2.0,
  "cls_loss_coef": 1.0,
  "aux_class_loss": true,
  "aux_class_loss_coef": 1.0,
  "dynamic_l1_enabled": true,
  "dynamic_l1_eps": 1e-06,
  "dynamic_l1_lambda_w": 1.0,
  "dynamic_l1_lambda_h": 1.0,
  "center_weight_clip": null,
  "normalize_dynamic_weights": false,
  "pw_ciou_enabled": true,
  "pw_ciou_weight_mode": "ratio",
  "pw_ciou_eps": 1e-06,
  "pw_ciou_weight_clip": null,
  "pw_ciou_mean_width": 0.1273,
  "recall_loss_enabled": false,
  "dataset_file": "coco",
  "coco_path": "../data/coco/traindatav1",
  "remove_difficult": false,
  "device": "auto",
  "seed": 42,
  "num_workers": 4,
  "world_size": 1,
  "dist_url": "env://",
  "lr": 1e-05,
  "lr_backbone": 1e-06,
  "batch_size": 16,
  "weight_decay": 0.0001,
  "epochs": 10,
  "lr_drop": 100,
  "clip_max_norm": 0.1,
  "amp": true,
  "tf32": true,
  "cudnn_benchmark": true,
  "output_dir": "../output/train/mrmpformer_trainv2_ftv1",
  "resume": "checkpoint/mrmpformer_trainv2.pth",
  "start_epoch": 0,
  "reset_optimizer": true,
  "frozen_weights": null,
  "coco_panoptic_path": null,
  "eval": false
}
```

## configs/mrmpformer_v3.json

- ???4166 bytes????text????102

```json
{
  "model": "mrmpformer_v1",
  "backbone": "resnet50",
  "dilation": false,
  "position_embedding": "sine",
  "enc_layers": 1,
  "dec_layers": 3,
  "num_queries": 3,
  "hidden_dim": 256,
  "nheads": 8,
  "dim_feedforward": 2048,
  "dropout": 0.1,
  "pre_norm": false,
  "masks": false,
  "aux_loss": true,
  "iou_type": "ciou",
  "set_cost_class": 1,
  "set_cost_bbox": 5,
  "set_cost_iou": 2,
  "mask_loss_coef": 1,
  "dice_loss_coef": 1,
  "bbox_loss_coef": 5,
  "iou_loss_coef": 2,
  "eos_coef": 0.1,
  "num_fdr_bins": 33,
  "fdr_bin_power": 2.0,
  "fdr_bin_values": null,
  "fdr_scale_mode": "initial_box_width",
  "fdr_layer_weights": [
    0.5,
    0.7,
    1.0
  ],
  "fdr_loss_coef": 2.0,
  "fdr_min_width": 0.0001,
  "detach_boundary_feedback": false,
  "classification_loss": "focal",
  "focal_alpha": 0.25,
  "focal_gamma": 2.0,
  "cls_loss_coef": 1.0,
  "aux_class_loss": true,
  "aux_class_loss_coef": 1.0,
  "dynamic_l1_enabled": true,
  "dynamic_l1_eps": 1e-06,
  "dynamic_l1_lambda_w": 1.0,
  "dynamic_l1_lambda_h": 1.0,
  "center_weight_clip": null,
  "normalize_dynamic_weights": false,
  "pw_ciou_enabled": true,
  "pw_ciou_weight_mode": "ratio",
  "pw_ciou_eps": 1e-06,
  "pw_ciou_weight_clip": null,
  "pw_ciou_mean_width": 0.1273,
  "recall_loss_enabled": false,
  "dataset_file": "coco",
  "coco_path": "../data/coco/traindatav1",
  "remove_difficult": false,
  "device": "auto",
  "seed": 42,
  "num_workers": 4,
  "world_size": 1,
  "dist_url": "env://",
  "lr": 0.0001,
  "lr_backbone": 1e-05,
  "batch_size": 16,
  "weight_decay": 0.0001,
  "epochs": 30,
  "lr_drop": 20,
  "clip_max_norm": 0.1,
  "amp": true,
  "tf32": true,
  "cudnn_benchmark": true,
  "output_dir": "../output/train/mrmpformerv3",
  "resume": null,
  "start_epoch": 0,
  "reset_optimizer": false,
  "frozen_weights": null,
  "coco_panoptic_path": null,
  "eval": false
}
```

## configs/mrmpformer_v3_rebuild.json

- ???3416 bytes????text????96

```json
{
  "model": "mrmpformer_v1",
  "backbone": "resnet50",
  "dilation": false,
  "position_embedding": "sine",
  "enc_layers": 1,
  "dec_layers": 3,
  "num_queries": 3,
  "hidden_dim": 256,
  "nheads": 8,
  "dim_feedforward": 2048,
  "dropout": 0.1,
  "pre_norm": false,
  "masks": false,
  "aux_loss": true,
  "iou_type": "ciou",
  "set_cost_class": 1,
  "set_cost_bbox": 5,
  "set_cost_iou": 2,
  "mask_loss_coef": 1,
  "dice_loss_coef": 1,
  "bbox_loss_coef": 5,
  "iou_loss_coef": 2,
  "eos_coef": 0.1,
  "num_fdr_bins": 33,
  "fdr_bin_power": 2.0,
  "fdr_bin_values": null,
  "fdr_scale_mode": "initial_box_width",
  "fdr_layer_weights": [
    0.5,
    0.7,
    1.0
  ],
  "fdr_loss_coef": 2.0,
  "fdr_min_width": 0.0001,
  "detach_boundary_feedback": false,
  "classification_loss": "focal",
  "focal_alpha": 0.25,
  "focal_gamma": 2.0,
  "cls_loss_coef": 1.0,
  "aux_class_loss": true,
  "aux_class_loss_coef": 1.0,
  "dynamic_l1_enabled": true,
  "dynamic_l1_eps": 1e-06,
  "dynamic_l1_lambda_w": 1.0,
  "dynamic_l1_lambda_h": 1.0,
  "center_weight_clip": null,
  "normalize_dynamic_weights": false,
  "pw_ciou_enabled": true,
  "pw_ciou_weight_mode": "ratio",
  "pw_ciou_eps": 1e-06,
  "pw_ciou_weight_clip": null,
  "pw_ciou_mean_width": 0.1273,
  "recall_loss_enabled": false,
  "dataset_file": "coco",
  "coco_path": "../data/coco/traindatav1",
  "remove_difficult": false,
  "device": "auto",
  "seed": 42,
  "num_workers": 4,
  "world_size": 1,
  "dist_url": "env://",
  "lr": 0.0001,
  "lr_backbone": 1e-05,
  "batch_size": 16,
  "weight_decay": 0.0001,
  "epochs": 30,
  "lr_drop": 20,
  "clip_max_norm": 0.1,
  "amp": true,
  "tf32": true,
  "cudnn_benchmark": true,
  "output_dir": "../output/train/mrmpformerv3_rebuild",
  "resume": null,
  "start_epoch": 0,
  "reset_optimizer": false,
  "frozen_weights": null,
  "coco_panoptic_path": null,
  "eval": false
}
```

## configs/quanformer_baseline.json

- ???5623 bytes????text????124

```json
{
  "model": "quanformer",
  "backbone": "resnet50",
  "dilation": false,
  "position_embedding": "sine",
  "enc_layers": 1,
  "dec_layers": 1,
  "num_queries": 3,
  "hidden_dim": 256,
  "nheads": 8,
  "dim_feedforward": 2048,
  "dropout": 0.1,
  "pre_norm": false,
  "masks": false,
  "aux_loss": true,
  "iou_type": "ciou",
  "set_cost_class": 1,
  "set_cost_bbox": 5,
  "set_cost_iou": 2,
  "mask_loss_coef": 1,
  "dice_loss_coef": 1,
  "bbox_loss_coef": 5,
  "iou_loss_coef": 2,
  "eos_coef": 0.1,
  "classification_loss": "focal",
  "focal_alpha": 0.25,
  "focal_gamma": 2.0,
  "cls_loss_coef": 1.0,
  "aux_class_loss": true,
  "aux_class_loss_coef": 1.0,
  "dynamic_l1_enabled": true,
  "dynamic_l1_eps": 1e-06,
  "dynamic_l1_lambda_w": 1.0,
  "dynamic_l1_lambda_h": 1.0,
  "center_weight_clip": null,
  "normalize_dynamic_weights": false,
  "pw_ciou_enabled": true,
  "pw_ciou_weight_mode": "ratio",
  "pw_ciou_eps": 1e-06,
  "pw_ciou_weight_clip": null,
  "pw_ciou_mean_width": null,
  "recall_loss_enabled": false,
  "num_fdr_bins": 33,
  "fdr_bin_power": 2.0,
  "fdr_bin_values": null,
  "fdr_scale_mode": "initial_box_width",
  "fdr_layer_weights": [
    0.5,
    0.7,
    1.0
  ],
  "fdr_loss_coef": 2.0,
  "fdr_min_width": 0.0001,
  "detach_boundary_feedback": false,
  "dataset_file": "coco",
  "coco_path": "../data/test/coco",
  "remove_difficult": false,
  "device": "auto",
  "seed": 42,
  "num_workers": 4,
  "world_size": 1,
  "dist_url": "env://",
  "lr": 0.0001,
  "lr_backbone": 1e-05,
  "batch_size": 16,
  "weight_decay": 0.0001,
  "epochs": 30,
  "lr_drop": 20,
  "clip_max_norm": 0.1,
  "amp": false,
  "tf32": false,
  "cudnn_benchmark": false,
  "output_dir": "../output/train/baseline",
  "resume": null,
  "start_epoch": 0,
  "reset_optimizer": false,
  "frozen_weights": null,
  "coco_panoptic_path": null,
  "eval": false
}
```

## configs/quanformer_baseline_test1ft.json

- ???2367 bytes????text????80

```json
{
  "model": "quanformer",
  "backbone": "resnet50",
  "dilation": false,
  "position_embedding": "sine",
  "enc_layers": 1,
  "dec_layers": 1,
  "num_queries": 3,
  "hidden_dim": 256,
  "nheads": 8,
  "dim_feedforward": 2048,
  "dropout": 0.1,
  "pre_norm": false,
  "masks": false,
  "aux_loss": true,
  "iou_type": "ciou",
  "set_cost_class": 1,
  "set_cost_bbox": 5,
  "set_cost_iou": 2,
  "mask_loss_coef": 1,
  "dice_loss_coef": 1,
  "bbox_loss_coef": 5,
  "iou_loss_coef": 2,
  "eos_coef": 0.1,
  "classification_loss": "focal",
  "focal_alpha": 0.25,
  "focal_gamma": 2.0,
  "cls_loss_coef": 1.0,
  "aux_class_loss": true,
  "aux_class_loss_coef": 1.0,
  "dynamic_l1_enabled": true,
  "dynamic_l1_eps": 1e-06,
  "dynamic_l1_lambda_w": 1.0,
  "dynamic_l1_lambda_h": 1.0,
  "center_weight_clip": null,
  "normalize_dynamic_weights": false,
  "pw_ciou_enabled": true,
  "pw_ciou_weight_mode": "ratio",
  "pw_ciou_eps": 1e-06,
  "pw_ciou_weight_clip": null,
  "pw_ciou_mean_width": null,
  "recall_loss_enabled": false,
  "num_fdr_bins": 33,
  "fdr_bin_power": 2.0,
  "fdr_bin_values": null,
  "fdr_scale_mode": "initial_box_width",
  "fdr_layer_weights": [
    0.5,
    0.7,
    1.0
  ],
  "fdr_loss_coef": 2.0,
  "fdr_min_width": 0.0001,
  "detach_boundary_feedback": false,
  "dataset_file": "coco",
  "coco_path": "../data/coco/test1_ft",
  "remove_difficult": false,
  "start_epoch": 0,
  "frozen_weights": null,
  "coco_panoptic_path": null,
  "eval": false,
  "resume": "checkpoint/quanformer.pth",
  "reset_optimizer": true,
  "output_dir": "../output/train/quanformer_test1_ft",
  "device": "auto",
  "lr": 1e-05,
  "lr_backbone": 1e-06,
  "batch_size": 8,
  "weight_decay": 0.0001,
  "epochs": 20,
  "lr_drop": 100,
  "clip_max_norm": 0.1,
  "seed": 42,
  "num_workers": 4,
  "amp": true,
  "tf32": true,
  "cudnn_benchmark": true,
  "world_size": 1,
  "dist_url": "env://"
}
```

## configs/quanformer_traindatav1.json

- ???5964 bytes????text????124

```json
{
  "model": "quanformer",
  "backbone": "resnet50",
  "dilation": false,
  "position_embedding": "sine",
  "enc_layers": 1,
  "dec_layers": 1,
  "num_queries": 3,
  "hidden_dim": 256,
  "nheads": 8,
  "dim_feedforward": 2048,
  "dropout": 0.1,
  "pre_norm": false,
  "masks": false,
  "aux_loss": true,
  "iou_type": "ciou",
  "set_cost_class": 1,
  "set_cost_bbox": 5,
  "set_cost_iou": 2,
  "mask_loss_coef": 1,
  "dice_loss_coef": 1,
  "bbox_loss_coef": 5,
  "iou_loss_coef": 2,
  "eos_coef": 0.1,
  "classification_loss": "focal",
  "focal_alpha": 0.25,
  "focal_gamma": 2.0,
  "cls_loss_coef": 1.0,
  "aux_class_loss": true,
  "aux_class_loss_coef": 1.0,
  "dynamic_l1_enabled": true,
  "dynamic_l1_eps": 1e-06,
  "dynamic_l1_lambda_w": 1.0,
  "dynamic_l1_lambda_h": 1.0,
  "center_weight_clip": null,
  "normalize_dynamic_weights": false,
  "pw_ciou_enabled": true,
  "pw_ciou_weight_mode": "ratio",
  "pw_ciou_eps": 1e-06,
  "pw_ciou_weight_clip": null,
  "pw_ciou_mean_width": null,
  "recall_loss_enabled": false,
  "num_fdr_bins": 33,
  "fdr_bin_power": 2.0,
  "fdr_bin_values": null,
  "fdr_scale_mode": "initial_box_width",
  "fdr_layer_weights": [
    0.5,
    0.7,
    1.0
  ],
  "fdr_loss_coef": 2.0,
  "fdr_min_width": 0.0001,
  "detach_boundary_feedback": false,
  "dataset_file": "coco",
  "coco_path": "../data/coco/traindatav1",
  "remove_difficult": false,
  "device": "auto",
  "seed": 42,
  "num_workers": 4,
  "world_size": 1,
  "dist_url": "env://",
  "lr": 0.0001,
  "lr_backbone": 1e-05,
  "batch_size": 16,
  "weight_decay": 0.0001,
  "epochs": 30,
  "lr_drop": 20,
  "clip_max_norm": 0.1,
  "amp": false,
  "tf32": false,
  "cudnn_benchmark": false,
  "output_dir": "../output/train/quanformer_traindatav1",
  "resume": null,
  "start_epoch": 0,
  "reset_optimizer": false,
  "frozen_weights": null,
  "coco_panoptic_path": null,
  "eval": false
}
```

## configs/quanformer_traindatav1_ft1000.json

- ???6145 bytes????text????124

```json
{
  "model": "quanformer",
  "backbone": "resnet50",
  "dilation": false,
  "position_embedding": "sine",
  "enc_layers": 1,
  "dec_layers": 1,
  "num_queries": 3,
  "hidden_dim": 256,
  "nheads": 8,
  "dim_feedforward": 2048,
  "dropout": 0.1,
  "pre_norm": false,
  "masks": false,
  "aux_loss": true,
  "iou_type": "ciou",
  "set_cost_class": 1,
  "set_cost_bbox": 5,
  "set_cost_iou": 2,
  "mask_loss_coef": 1,
  "dice_loss_coef": 1,
  "bbox_loss_coef": 5,
  "iou_loss_coef": 2,
  "eos_coef": 0.1,
  "classification_loss": "focal",
  "focal_alpha": 0.25,
  "focal_gamma": 2.0,
  "cls_loss_coef": 1.0,
  "aux_class_loss": true,
  "aux_class_loss_coef": 1.0,
  "dynamic_l1_enabled": true,
  "dynamic_l1_eps": 1e-06,
  "dynamic_l1_lambda_w": 1.0,
  "dynamic_l1_lambda_h": 1.0,
  "center_weight_clip": null,
  "normalize_dynamic_weights": false,
  "pw_ciou_enabled": true,
  "pw_ciou_weight_mode": "ratio",
  "pw_ciou_eps": 1e-06,
  "pw_ciou_weight_clip": null,
  "pw_ciou_mean_width": null,
  "recall_loss_enabled": false,
  "num_fdr_bins": 33,
  "fdr_bin_power": 2.0,
  "fdr_bin_values": null,
  "fdr_scale_mode": "initial_box_width",
  "fdr_layer_weights": [
    0.5,
    0.7,
    1.0
  ],
  "fdr_loss_coef": 2.0,
  "fdr_min_width": 0.0001,
  "detach_boundary_feedback": false,
  "dataset_file": "coco",
  "coco_path": "../data/coco/traindatav1_sub1000",
  "remove_difficult": false,
  "start_epoch": 0,
  "frozen_weights": null,
  "coco_panoptic_path": null,
  "eval": false,
  "resume": "checkpoint/quanformer.pth",
  "reset_optimizer": true,
  "output_dir": "../output/train/quanformer_traindatav1_ft1000",
  "device": "auto",
  "lr": 1e-05,
  "lr_backbone": 1e-06,
  "batch_size": 8,
  "weight_decay": 0.0001,
  "epochs": 20,
  "lr_drop": 100,
  "clip_max_norm": 0.1,
  "amp": false,
  "tf32": false,
  "cudnn_benchmark": false,
  "seed": 42,
  "num_workers": 4,
  "world_size": 1,
  "dist_url": "env://"
}
```

## configs/quanformer_v2_finetune.json

- ???5703 bytes????text????124

```json
{
  "model": "quanformer",
  "backbone": "resnet50",
  "dilation": false,
  "position_embedding": "sine",
  "enc_layers": 1,
  "dec_layers": 1,
  "num_queries": 3,
  "hidden_dim": 256,
  "nheads": 8,
  "dim_feedforward": 2048,
  "dropout": 0.1,
  "pre_norm": false,
  "masks": false,
  "aux_loss": true,
  "iou_type": "ciou",
  "set_cost_class": 1,
  "set_cost_bbox": 5,
  "set_cost_iou": 2,
  "mask_loss_coef": 1,
  "dice_loss_coef": 1,
  "bbox_loss_coef": 5,
  "iou_loss_coef": 2,
  "eos_coef": 0.1,
  "classification_loss": "focal",
  "focal_alpha": 0.25,
  "focal_gamma": 2.0,
  "cls_loss_coef": 1.0,
  "aux_class_loss": true,
  "aux_class_loss_coef": 1.0,
  "dynamic_l1_enabled": true,
  "dynamic_l1_eps": 1e-06,
  "dynamic_l1_lambda_w": 1.0,
  "dynamic_l1_lambda_h": 1.0,
  "center_weight_clip": null,
  "normalize_dynamic_weights": false,
  "pw_ciou_enabled": true,
  "pw_ciou_weight_mode": "ratio",
  "pw_ciou_eps": 1e-06,
  "pw_ciou_weight_clip": null,
  "pw_ciou_mean_width": null,
  "recall_loss_enabled": false,
  "num_fdr_bins": 33,
  "fdr_bin_power": 2.0,
  "fdr_bin_values": null,
  "fdr_scale_mode": "initial_box_width",
  "fdr_layer_weights": [
    0.5,
    0.7,
    1.0
  ],
  "fdr_loss_coef": 2.0,
  "fdr_min_width": 0.0001,
  "detach_boundary_feedback": false,
  "dataset_file": "coco",
  "coco_path": "../data/test/coco",
  "remove_difficult": false,
  "start_epoch": 0,
  "frozen_weights": null,
  "coco_panoptic_path": null,
  "eval": false,
  "resume": "checkpoint/quanformer.pth",
  "reset_optimizer": true,
  "output_dir": "../output/train/quanformer_v2_finetune",
  "device": "auto",
  "lr": 1e-05,
  "lr_backbone": 1e-06,
  "batch_size": 4,
  "weight_decay": 0.0001,
  "epochs": 10,
  "lr_drop": 100,
  "clip_max_norm": 0.1,
  "amp": false,
  "tf32": false,
  "cudnn_benchmark": false,
  "seed": 42,
  "num_workers": 4,
  "world_size": 1,
  "dist_url": "env://"
}
```

## configs/quanformer_v3_finetune.json

- ???3338 bytes????text????87

```json
{
  "model": "quanformer",
  "backbone": "resnet50",
  "dilation": false,
  "position_embedding": "sine",
  "enc_layers": 1,
  "dec_layers": 1,
  "num_queries": 3,
  "hidden_dim": 256,
  "nheads": 8,
  "dim_feedforward": 2048,
  "dropout": 0.1,
  "pre_norm": false,
  "masks": false,
  "aux_loss": true,
  "iou_type": "ciou",
  "set_cost_class": 1,
  "set_cost_bbox": 5,
  "set_cost_iou": 2,
  "mask_loss_coef": 1,
  "dice_loss_coef": 1,
  "bbox_loss_coef": 5,
  "iou_loss_coef": 2,
  "eos_coef": 0.1,
  "classification_loss": "focal",
  "focal_alpha": 0.25,
  "focal_gamma": 2.0,
  "cls_loss_coef": 1.0,
  "aux_class_loss": true,
  "aux_class_loss_coef": 1.0,
  "dynamic_l1_enabled": true,
  "dynamic_l1_eps": 1e-06,
  "dynamic_l1_lambda_w": 1.0,
  "dynamic_l1_lambda_h": 1.0,
  "center_weight_clip": null,
  "normalize_dynamic_weights": false,
  "pw_ciou_enabled": true,
  "pw_ciou_weight_mode": "ratio",
  "pw_ciou_eps": 1e-06,
  "pw_ciou_weight_clip": null,
  "pw_ciou_mean_width": null,
  "recall_loss_enabled": false,
  "num_fdr_bins": 33,
  "fdr_bin_power": 2.0,
  "fdr_bin_values": null,
  "fdr_scale_mode": "initial_box_width",
  "fdr_layer_weights": [
    0.5,
    0.7,
    1.0
  ],
  "fdr_loss_coef": 2.0,
  "fdr_min_width": 0.0001,
  "detach_boundary_feedback": false,
  "dataset_file": "coco",
  "coco_path": "../data/coco/merged",
  "remove_difficult": false,
  "start_epoch": 0,
  "frozen_weights": null,
  "coco_panoptic_path": null,
  "eval": false,
  "resume": "checkpoint/quanformer.pth",
  "reset_optimizer": true,
  "output_dir": "../output/train/quanformer_v3_finetune",
  "device": "auto",
  "lr": 1e-05,
  "lr_backbone": 1e-06,
  "batch_size": 4,
  "weight_decay": 0.0001,
  "epochs": 10,
  "lr_drop": 100,
  "clip_max_norm": 0.1,
  "seed": 42,
  "num_workers": 4,
  "amp": true,
  "tf32": true,
  "cudnn_benchmark": true,
  "world_size": 1,
  "dist_url": "env://"
}
```

## framework/__init__.py

- ???126 bytes????python????2


## framework/datasets/__init__.py

- ???922 bytes????python????25

- L8 `get_coco_api_from_dataset`
- L18 `build_dataset`

## framework/datasets/coco.py

- ???6838 bytes????python????176

COCO dataset which returns image_id for evaluation.

- L21 `CocoDetection`
- L22 `CocoDetection.__init__`
- L37 `CocoDetection._load_image_and_annotations`
- L45 `CocoDetection.__getitem__`
- L57 `convert_coco_poly_to_mask`
- L74 `ConvertCocoPolysToMask`
- L75 `ConvertCocoPolysToMask.__init__`
- L78 `ConvertCocoPolysToMask.__call__`
- L138 `make_coco_transforms` ? 本项目色谱图专用的数据变换（非 DETR 原版自然照片增强）。
- L156 `build`

## framework/datasets/coco_eval.py

- ???9002 bytes????python????257

COCO evaluator that works in distributed mode.

- L22 `CocoEvaluator`
- L23 `CocoEvaluator.__init__`
- L36 `CocoEvaluator.update`
- L55 `CocoEvaluator.synchronize_between_processes`
- L60 `CocoEvaluator.accumulate`
- L64 `CocoEvaluator.summarize`
- L69 `CocoEvaluator.prepare`
- L79 `CocoEvaluator.prepare_for_coco_detection`
- L103 `CocoEvaluator.prepare_for_coco_segmentation`
- L138 `CocoEvaluator.prepare_for_coco_keypoint`
- L165 `convert_to_xywh`
- L170 `merge`
- L192 `create_common_coco_eval`
- L208 `evaluate` ? Run per image evaluation on given images and store results (a list of dict) in self.evalImgs

## framework/datasets/transforms.py

- ???8820 bytes????python????276

Transforms and data augmentation for both image + bbox.

- L16 `crop`
- L59 `hflip`
- L76 `resize`
- L79 `resize.get_size_with_aspect_ratio`
- L99 `resize.get_size`
- L135 `pad`
- L148 `RandomCrop`
- L149 `RandomCrop.__init__`
- L152 `RandomCrop.__call__`
- L157 `RandomSizeCrop`
- L158 `RandomSizeCrop.__init__`
- L162 `RandomSizeCrop.__call__`
- L169 `CenterCrop`
- L170 `CenterCrop.__init__`
- L173 `CenterCrop.__call__`
- L181 `RandomHorizontalFlip`
- L182 `RandomHorizontalFlip.__init__`
- L185 `RandomHorizontalFlip.__call__`
- L191 `RandomResize`
- L192 `RandomResize.__init__`
- L197 `RandomResize.__call__`
- L202 `RandomPad`
- L203 `RandomPad.__init__`
- L206 `RandomPad.__call__`
- L212 `RandomSelect` ? Randomly selects between transforms1 and transforms2,
- L217 `RandomSelect.__init__`
- L222 `RandomSelect.__call__`
- L228 `ToTensor`
- L229 `ToTensor.__call__`
- L233 `RandomErasing`
- L235 `RandomErasing.__init__`
- L238 `RandomErasing.__call__`
- L242 `Normalize`
- L243 `Normalize.__init__`
- L247 `Normalize.__call__`
- L261 `Compose`
- L262 `Compose.__init__`
- L265 `Compose.__call__`
- L270 `Compose.__repr__`

## framework/engine.py

- ???13348 bytes????python????281

Train and eval functions used in main.py

- L24 `_print_legend_once` ? 训练开始时打印一次图例，代替在每个指标名上重复中英对照。
- L34 `_print_avg_stats` ? epoch 平均摘要：64 字符宽框、一组一行、趋势箭头压缩多层数据。
- L53 `_print_avg_stats.line`
- L118 `train_one_epoch`
- L191 `evaluate`

## framework/hubconf.py

- ???1281 bytes????python????35

- L11 `_make_former`
- L22 `quan_former` ? DETR R50 with 1 encoder and 1 decoder layers.

## framework/util/__init__.py

- ???182 bytes????python????3

公共工具模块：文件I/O、BBox运算、通用辅助函数。


## framework/util/box_ops.py

- ???6567 bytes????python????177

Utilities for bounding box manipulation and GIoU.

- L9 `_upcast`
- L17 `box_cxcywh_to_xyxy`
- L24 `box_xyxy_to_cxcywh`
- L32 `box_iou`
- L48 `generalized_box_iou` ? Generalized IoU from https://giou.stanford.edu/
- L75 `_box_diou_iou`
- L96 `complete_box_iou` ? Return complete intersection-over-union (Jaccard index) between two sets of boxes.
- L129 `distance_box_iou` ? Return distance intersection-over-union (Jaccard index) between two sets of boxes.
- L153 `masks_to_boxes` ? Compute the bounding boxes around the provided masks

## framework/util/logutil.py

- ???5075 bytes????python????158

MRMPFormer 运行时日志过滤工具。

- L52 `configure_log_level` ? 设置全局日志级别。
- L68 `get_log_level` ? 返回当前日志级别名称。
- L79 `_FilteredStdout` ? 按行缓冲，根据前缀级别决定是否写入原始 stdout。
- L82 `_FilteredStdout.__init__`
- L86 `_FilteredStdout.write`
- L97 `_FilteredStdout._flush`
- L106 `_FilteredStdout._write_line`
- L113 `_FilteredStdout.flush`
- L119 `_FilteredStdout.__getattr__`
- L127 `_classify_line` ? 返回行对应的日志级别数值；无法识别时返回 _UNCLASSIFIED（无条件放行）。
- L136 `install_filter` ? 全局安装 stdout 过滤器（幂等）。
- L147 `uninstall_filter` ? 卸载过滤器，恢复原始 stdout。

## framework/util/misc.py

- ???17556 bytes????python????496

Misc functions, including distributed helpers.

- L27 `_get_dist_device` ? Return the best available device for distributed operations.
- L33 `safe_torch_load` ? Safe wrapper for torch.load that handles weights_only parameter
- L41 `SmoothedValue` ? Track a series of values and provide access to smoothed values over a
- L46 `SmoothedValue.__init__`
- L54 `SmoothedValue.update`
- L59 `SmoothedValue.synchronize_between_processes` ? Warning: does not synchronize the deque!
- L73 `SmoothedValue.median`
- L78 `SmoothedValue.avg`
- L83 `SmoothedValue.global_avg`
- L87 `SmoothedValue.max`
- L91 `SmoothedValue.value`
- L94 `SmoothedValue.__str__`
- L103 `all_gather` ? Run all_gather on arbitrary picklable data (not necessarily tensors)
- L146 `reduce_dict` ? Args:
- L173 `MetricLogger`
- L174 `MetricLogger.__init__`
- L184 `MetricLogger.update`
- L191 `MetricLogger.__getattr__`
- L199 `MetricLogger.__str__`
- L215 `MetricLogger.synchronize_between_processes`
- L219 `MetricLogger.add_meter`
- L222 `MetricLogger.log_every`
- L277 `get_sha`
- L280 `get_sha._run`
- L297 `collate_fn`
- L303 `_max_by_axis`
- L312 `NestedTensor`
- L313 `NestedTensor.__init__`
- L317 `NestedTensor.to`
- L328 `NestedTensor.decompose`
- L331 `NestedTensor.__repr__`
- L335 `nested_tensor_from_tensor_list`
- L363 `_onnx_nested_tensor_from_tensor_list`
- L391 `setup_for_distributed` ? This function disables printing when not in master process
- L398 `setup_for_distributed.print`
- L406 `is_dist_avail_and_initialized`
- L414 `get_world_size`
- L420 `get_rank`
- L426 `is_main_process`
- L430 `save_on_master`
- L435 `init_distributed_mode`
- L461 `accuracy` ? Computes the precision@k for the specified values of k
- L479 `interpolate` ? Equivalent to nn.functional.interpolate, but with support for empty batch sizes.

## inference/__init__.py

- ???50 bytes????python????1


## inference/cli.py

- ???79716 bytes????python????1652

MRMPFormer 统一推理入口（3 种模式；roi / pipeline 均支持单文件与目录递归扫描）。

- L41 `_format_elapsed_ms` ? Milliseconds only.
- L49 `_format_elapsed` ? Human-readable duration with milliseconds for console logs.
- L64 `_format_mb`
- L72 `_PipelineResourceMonitor` ? Background CPU / memory sampling for pipeline modes (psutil preferred).
- L77 `_PipelineResourceMonitor.__init__`
- L89 `_PipelineResourceMonitor._init_backend`
- L127 `_PipelineResourceMonitor._read_sample`
- L153 `_PipelineResourceMonitor.start`
- L163 `_PipelineResourceMonitor._loop`
- L170 `_PipelineResourceMonitor.stop`
- L179 `_PipelineResourceMonitor._stats_from_snaps`
- L201 `_PipelineResourceMonitor.stats_for_interval`
- L206 `_PipelineResourceMonitor.stats_for_merged_intervals`
- L216 `_PipelineResourceMonitor.overall_stats`
- L219 `_PipelineResourceMonitor.backend_name`
- L223 `_print_mzml_roi_stats_summary` ? Print per-mzML ROI image counts after testXIC stage.
- L252 `_print_resource_stats_block`
- L280 `_build_pipeline_timing_report` ? Build human-readable timing lines and a JSON-serializable record.
- L298 `_build_pipeline_timing_report.add`
- L431 `_write_pipeline_timing_logs` ? Append timing summary to pipeline_timing.log and pipeline_timing_runs.jsonl.
- L448 `_print_pipeline_timing_summary` ? Print pipeline stage timings to stdout; optionally write logs. Returns JSON record.
- L482 `_resolve_exp_name` ? 确定实验名（Step 7）：--exp_name 优先；缺省回退：单 mzML → 文件名 stem；
- L498 `_collect_mzml_inputs` ? 收集输入 mzML：--mzml 单文件/目录，或 --batch_dir 目录（递归含子目录）。
- L543 `_prepare_label_driven_roi` ? ROI 标注驱动（B 范式）必经准备：解析标注 xlsx + RT 一致性 QC。
- L566 `_group_labels_by_sample` ? 按 sample_id 分组（保持出现顺序），返回 ({sid: [rows]}, [sid, ...])。
- L578 `_pick_sample_labels` ? 多样本标注 → 当前 mzML 对应行（与 coco_annotation 同规则；匹配失败回退全部标注）。
- L603 `_merge_qc_csvs` ? 按 glob 汇总多个同构 QC csv（各样本子目录），加 stem 列；返回 (文件数, 合并行数)。
- L622 `_emit_qc5_sample` ? 从样本的 prediction_refined.csv 抽门控列子集，落盘 qc5_refined_<样本名>.csv（防线5 样本级 QC）。
- L642 `_collect_qc_tables` ? pipeline 结束后统一汇总各环节 QC 表到 ../output/QC/<run_name>/。
- L776 `_write_full_qc_alert` ? 生成 qc_alert.md：汇总全部 QC 防线的人工预警（防线1-5），供人工一站式复核。
- L858 `_write_predictions_model_summary` ? 需求1：predictions_model 根级汇总——predictions_model_all.csv + predictions_model_report.md（放阶段文件夹内）。
- L907 `_write_prediction_refined_summary` ? 需求5：prediction_refined 根级汇总——prediction_refined_all.csv + prediction_refined_report.md（放阶段文件夹内）。
- L963 `main_cli`

## inference/massnova.py

- ???76234 bytes????python????1531

massnova — 整谱 XIC 全峰识别推理模式（Phase 0-4）
适用场景：MassNova 集成 / 仅提供时序数据（不依赖标注）。

- L61 `_native_id_to_str`
- L69 `_parse_q1_from_text` ? 从 native_id 文本解析母离子 m/z（Q1）；与 xic_extraction 的 Q1 规则一致，不解析 Q3。
- L81 `_q1_from_chrom_metadata`
- L91 `_extract_q1`
- L98 `_rt_to_minutes` ? RT 单位判定（D15）：优先中位时间步长，回退 >200 启发式。返回分钟数组。
- L111 `extract_full_xics` ? Phase0：读取 mzML 全部 transition 时序数据。
- L197 `_merge_insignificant_valleys` ? 相邻候选峰之间谷不显著（谷不够深或不在两峰间居中）→ 合并保留较高者。
- L230 `enumerate_peaks` ? Phase1：整谱候选枚举。
- L292 `_load_centwave_detect` ? 以完全隔离方式加载 Centwave/detector.py 的 detect_peaks_centwave。
- L309 `_load_centwave_detect._load`
- L328 `enumerate_peaks_centwave` ? Phase1-centwave：以 CentWave（CWT 脊线 + 峰形趋势 + 联合质量过滤）枚举候选。
- L413 `refine_all_boundaries` ? Phase2（兜底路径）：对给定候选做两遍精修（D13）。
- L436 `refine_all_boundaries._walk`
- L469 `_max_consec_above`
- L485 `gate_peaks` ? Phase3a（兜底路径）：对精修后的候选峰做质量门控。
- L536 `validate_with_model` ? Phase3b（模型前置，每 mzML 一次批量验证）：全部枚举候选（未经信号精修/门控，
- L643 `_finalize_peak_metrics` ? 在最终边界上统一计算 snr / n_points / area（模型框与信号框同口径）。
- L671 `plot_massnova_stage_windows` ? pipeline 式 2min 窗口图：每峰 apex±window_half 切窗，窗内相交峰全部标注（多 query）。
- L696 `plot_massnova_stage_windows._match`
- L756 `plot_massnova_scan` ? Phase4a：整谱标注图（宽按 RT 跨度缩放，D12）。
- L790 `plot_massnova_model_xic` ? Phase4a'：model_plots（--plot）——与其他模式（pipeline/roi2inference）一致的可视化。
- L834 `write_outputs` ? Phase4：按实验布局写输出，返回峰明细行（供跨样本推理报告汇总）。
- L957 `_compound_of` ? uid 形如 '阿维菌素-1'（化合物名-离子通道）→ 化合物名（矩阵按化合物合并离子通道）。
- L963 `write_massnova_report` ? Phase5：跨样本推理报告（对齐 pipeline 的 inference_report_<实验名>.md + all.csv）。
- L1118 `_dedup_overlapping_peaks` ? 跨候选去重：区间重叠且 apex 间距 <= apex_tol（min）视为同一峰的重复框。
- L1129 `_dedup_overlapping_peaks._rank`
- L1150 `_half_width_spans` ? 从峰顶向两侧找强度跌破 baseline+0.5×峰高 的位置，返回 (左跨, 右跨)（min）。
- L1164 `fuse_fallback_boundaries` ? 兜底宽度保险丝：单侧跨度超过 ratio×该侧半高跨度时回缩（只收缩不扩张）。
- L1191 `_collect_mzml_inputs` ? 惰性复用 cli._collect_mzml_inputs，避免 cli ↔ massnova 循环导入。
- L1197 `_scan_params_from_args`
- L1198 `_scan_params_from_args._g`
- L1233 `run_massnova_on_mzml` ? 对单个 mzML 执行整谱全峰识别（模型前置、精修兜底），返回 info dict（含峰明细行）。
- L1423 `main` ? 统一入口（cli --mode massnova 与 python -m inference.massnova 共用）。
- L1465 `build_parser`

## inference/peak_event_merge.py

- ???12915 bytes????python????287

峰事件重组（推理结构优化）：谷底抬升合并 + 信号足点重划。

- L34 `_xic_row_for_image` ? N_mz*.jpeg 命名 → XIC 行索引（N 为 1-based）；无法解析返回 -1。
- L43 `_smooth` ? 轻量滑动平均，稳住 apex/谷底/足点判定；长度不足时原样返回。
- L53 `_finite`
- L58 `_apex_index` ? [lo, hi] RT 区间内 y 最大值索引；区间无点返回 -1。
- L69 `_baseline_and_noise`
- L81 `_nms_dedup` ? 同模态去重：按分数降序保留，若与已保留框的重叠 ≥ 较短框的 contain_ratio
- L98 `_merge_events` ? 相邻事件按"谷底抬升"合并（排序后从左到右，合并结果继续向左尝试吞并）。
- L146 `_foot_rebound` ? 对所有事件按信号足点重划边界（可外扩可内收）。
- L177 `_rt_to_px`
- L182 `reorganize_events` ? 峰事件重组主入口（predictor.run_single 调用）。

## inference/predictor.py

- ???48718 bytes????python????1075

- L28 `_to_result_tuple`
- L57 `_compound_label_from_feature`
- L65 `_expected_roi_name` ? 无检测时用于 prediction.csv 的 image 列占位，与 testXIC.roi_safe_name_base 命名一致。
- L74 `_parse_roi_basename` ? 从 ROI 文件名解析对齐方式。
- L105 `_adapt_prediction_for_quantify` ? 将检测结果对齐到 XIC 行索引，并且每个 XIC 只保留最高置信度框。
- L164 `_load_roi_windows` ? 若存在 roi_windows.csv（testXIC 生成），返回 { image_name: (rt_lo, rt_hi) }，否则返回 {}。
- L181 `_load_external_baselines` ? 加载外部基线 JSON，返回 {(mz, q3): (x_array, y_array)} 供匹配使用。
- L214 `_match_baseline_for_compound` ? 按 (mz, q3) 容差匹配化合物对应的外部基线，返回 (x, y) 或 None。优先匹配 q3 一致的。
- L229 `_integrate_each_predicted_box` ? 为每个预测框计算积分信息，返回可直接写入 prediction.csv 的明细行。
- L380 `_plot_predictions_with_baseline` ? 绘制预测图；若使用线性基线积分，在图上叠加基线。
- L479 `_plot_model_xic` ? Step 6：model_plots（--plot_style xic 默认）——XIC 曲线 + 多 query 阴影 + 置信度信息框。
- L555 `_feature_csv_has_compounds` ? feature.csv 是否含至少一行有效化合物（空文件或仅表头 → False）。
- L569 `run_single` ? 对单个 images_path 执行预测与积分，结果保存到 prediction_output 和 plot_dir。
- L846 `main`

## inference/two_round_detection.py

- ???40644 bytes????python????1005

两轮识别流程：置信度≥0.99 唯一主峰 + XIC(SNR+次峰) 筛选 → 掩蔽主峰 → 第二轮模型 → 合并 newprediction.csv → XIC 图标注

- L39 `_flat_triplet_early_stop_left_rt` ? 左肩外推：若连续三点相邻差分绝对值均很小，停在靠峰侧第一点 rt[i+2]。
- L64 `_flat_triplet_early_stop_right_rt` ? 右肩外推：对称地停在 rt[i-2]。
- L89 `clamp_refined_interval_width_to_pred_and_roi` ? 修正框宽度约束（仅上限）：
- L134 `_image_to_compound_index` ? N_mz* -> N-1 (0-based).
- L141 `_posterior_peer_conflict` ? 后验窗口 [ilo,ihi] 在 RT 上与同伴预测框重叠足够长，且重叠索引上强度仍明显高于噪声阈值时，
- L182 `walk_interval_left_to_noise_with_posterior` ? 从 i_start 向左（索引减小）外推：先到 y<=threshold，再要求外向 lookahead 个点均值仍处低水平；
- L241 `walk_interval_right_to_noise_with_posterior` ? 从 i_start 向右外推，后验逻辑同左；右肩三连微降早停对称。
- L295 `adjust_first_round_interval` ? 将边界移动停止条件改为：沿移动方向逐点移动，直到首次 <= 单侧噪声阈值即停。
- L383 `adjust_first_round_interval._nearest_idx`
- L450 `remove_overlap_from_second_interval` ? 改进二：若第二次区间与第一次重叠，优先第一次，从第二次区间去除重叠部分。
- L473 `images_with_multi_highconf_boxes` ? 同一张 ROI 图像若已有 >=2 行 score>=min_confidence，则认为首轮模型已给出多框，
- L492 `_pick_second_box_row_from_round1` ? 从首轮 prediction 中为同一 image 选另一高置信行作为第二框。
- L505 `_pick_second_box_row_from_round1._near_same_interval`
- L522 `filter_candidates_for_second_round` ? 筛选进入第二轮的候选：score>=min_confidence，SNR>=min_snr，存在次峰>=min_secondary_ratio。
- L578 `build_masked_subdir_for_candidates` ? 仅为候选图像生成掩蔽图。candidates 每项为 (row, xic_idx) 或 (row, xic_idx, rt_min_adj, rt_max_adj)。
- L670 `run_newtest_on_dir` ? 对目录运行 newtest，输出 prediction.csv 到 output_dir。
- L686 `merge_to_newprediction` ? 合并两轮结果。candidates_info: [(orig_image, row_round1), ...]
- L763 `plot_newprediction_on_xic` ? 在原 XIC 平滑图上绘制 1 或 2 个区间框及置信度。
- L840 `main`

## models/__init__.py

- ???1004 bytes????python????23

- L5 `build_model` ? 根据 args.model 选择模型变体。

## models/mrmpformer/__init__.py

- ???183 bytes????python????3


## models/mrmpformer/v1/__init__.py

- ???284 bytes????python????4


## models/mrmpformer/v1/detr.py

- ???37936 bytes????python????708

MRMPFormer v1 — 三层 Decoder + FDR 边界逐层精化（提示词文档实现）。

- L46 `MRMPFormer` ? 三层 Decoder + FDR 边界逐层精化模型。
- L49 `MRMPFormer.__init__`
- L106 `MRMPFormer._make_boundary_pos_fn` ? 闭包内独立维护解码链（与 forward 末尾的输出重算完全同构：
- L113 `MRMPFormer._make_boundary_pos_fn.boundary_pos_fn`
- L142 `MRMPFormer._scale_factor` ? s0：默认初始框宽 w0；roi_width 模式取 1.0（全图归一化尺度）。
- L150 `MRMPFormer.forward`
- L238 `MRMPFormer._set_aux_self_boxes` ? 中间层精化框组装（左右=该层，上下=初始框）。
- L245 `MRMPFormer._set_aux_loss`
- L251 `bots_diff_pos` ? 严格单调递增校验。
- L259 `MRMPSetCriterion` ? MRMPFormer v1 损失（提示词 §8-§11）。
- L272 `MRMPSetCriterion.__init__`
- L346 `MRMPSetCriterion.loss_labels` ? 分类损失：Focal（默认）或 CE 基线。目标编码与 QuanFormer 一致：
- L373 `MRMPSetCriterion.loss_cardinality`
- L381 `MRMPSetCriterion.loss_boxes` ? 定位损失：动态加权 L1（§10.1）+ PW-CIoU（§10.2），均可开关。
- L428 `MRMPSetCriterion.loss_fdr` ? FDR 分布监督（§9）+ 每层边界 MAE / 每层框 IoU / 越界率诊断。
- L500 `MRMPSetCriterion._get_src_permutation_idx`
- L505 `MRMPSetCriterion._get_tgt_permutation_idx`
- L510 `MRMPSetCriterion.get_loss`
- L520 `MRMPSetCriterion.forward`
- L567 `load_legacy_quanformer_state` ? 把单层 Decoder 的 QuanFormer checkpoint 迁移到 MRMPFormer v1。
- L617 `build`

## models/mrmpformer/v1/detr_special.py

- ???8100 bytes????python????165

[隔离实验] special_peak_v1 —— 特殊峰专项模型构建。

- L20 `SpecialSetCriterion` ? 特殊峰专项损失（仅覆盖 loss_labels / loss_boxes，其余继承）。
- L23 `SpecialSetCriterion.__init__`
- L36 `SpecialSetCriterion.loss_labels`
- L67 `SpecialSetCriterion.loss_boxes`
- L108 `build` ? 构建隔离版：MRMPFormer v1 结构 + SpecialSetCriterion。

## models/mrmpformer/v1/fdr.py

- ???20026 bytes????python????401

MRMPFormer v1 — FDR（Fine-grained Distribution Refinement）核心组件。

- L31 `make_bin_values` ? 生成非均匀 Bin 候选偏移 W(n)，n = 0..N-1。
- L48 `FDRHead` ? 每层独立的 FDR 分布头。
- L61 `FDRHead.__init__`
- L70 `FDRHead.forward` ? x: [B, Q, D] → z or Δz: [B, Q, 2, N]
- L81 `decode_expected_offsets` ? 累计分布 → 期望偏移（归一化坐标修正量）。
- L102 `BoundaryPositionMLP` ? (x_L, x_R) → 边界位置编码 [B, Q, D]。
- L109 `BoundaryPositionMLP.__init__`
- L114 `BoundaryPositionMLP.forward` ? lr: [B, Q, 2] 左右边界 → [B, Q, D]
- L126 `DistributionBoundaryLoss` ? 左右边界离散分布监督。
- L137 `DistributionBoundaryLoss.__init__`
- L143 `DistributionBoundaryLoss.soft_labels` ? target: [*] 目标偏移 → 两点软标签 (weights: [* , N], overflow: [*] bool)。
- L168 `DistributionBoundaryLoss.gaussian_soft_labels` ? target: [*] → 高斯核软标签 (weights: [*, N], overflow: [*] bool)。
- L184 `DistributionBoundaryLoss.forward` ? 软标签交叉熵。
- L224 `softmax_focal_loss` ? Softmax 形式 Focal Loss（替换 DETR 的 CE + eos_coef 机制）。
- L262 `dynamic_l1_weights` ? λ_c = 1 / (w_gt + eps)，λ_w = λ_h = 1（可配置）。
- L289 `dynamic_l1_loss` ? 动态加权 L1（提示词 §10.1）。
- L308 `peak_width_weighted_ciou` ? PW-CIoU = IoU - (bar_w/(w_gt+eps)) * ρ²/(c²+eps) - αv
- L345 `_ciou_with_center_weight` ? CIoU 核心计算，中心距离项 ρ²/c² 乘以逐 GT 权重 [M]。
- L399 `merge_weight_stats` ? 把 dynamic_l1 / pw_ciou 的权重分位数展平成日志键。

## models/mrmpformer/v1/losses_special.py

- ???3487 bytes????python????79

[隔离实验] special_peak_v1 —— 特殊峰专项损失组件。

- L21 `quality_focal_loss` ? QFL（Softmax 双类适配版）。输入必须是原始 Logits。
- L48 `log_width_l1_loss` ? 尺度等变 L1（cxcywh）。

## models/mrmpformer/v1/transformer.py

- ???6246 bytes????python????142

MRMPFormer v1 — FDR Transformer。

- L33 `FDRTransformerDecoder` ? 带边界位置反馈的 Transformer Decoder。
- L46 `FDRTransformerDecoder.__init__`
- L52 `FDRTransformerDecoder.forward` ? tgt/memory/query_pos_base: [Q, B, D]；pos: [HW, B, D]
- L86 `FDRTransformer` ? Encoder（与 QuanFormer 相同）+ FDRDecoder（逐层边界反馈）。
- L89 `FDRTransformer.__init__`
- L109 `FDRTransformer._reset_parameters`
- L115 `FDRTransformer.forward`
- L133 `build_fdr_transformer`

## models/quanformer/__init__.py

- ???44 bytes????python????1


## models/quanformer/detr.py

- ???18648 bytes????python????383

DETR model and criterion classes.

- L23 `DETR` ? This is the DETR module that performs object detection 
- L25 `DETR.__init__` ? Initializes the model.
- L46 `DETR.forward` ?  The forward expects a NestedTensor, which consists of:
- L77 `DETR._set_aux_loss`
- L85 `SetCriterion` ? This class computes the loss for DETR.
- L91 `SetCriterion.__init__` ? Create the criterion.
- L111 `SetCriterion.loss_labels` ? Classification loss (NLL)
- L133 `SetCriterion.loss_cardinality` ? Compute the cardinality error, ie the absolute error in the number of predicted non-empty boxes
- L146 `SetCriterion.loss_boxes` ? Compute the losses related to the bounding boxes, the L1 regression loss and the GIoU loss
- L183 `SetCriterion.loss_masks` ? Compute the losses related to the masks: the focal loss and the dice loss.
- L212 `SetCriterion._get_src_permutation_idx`
- L218 `SetCriterion._get_tgt_permutation_idx`
- L224 `SetCriterion.get_loss`
- L234 `SetCriterion.forward` ? This performs the loss computation.
- L277 `PostProcess` ? This module converts the model's output into the format expected by the coco api
- L280 `PostProcess.forward` ? Perform the computation
- L308 `MLP` ? Very simple multi-layer perceptron (also called FFN)
- L311 `MLP.__init__`
- L317 `MLP.forward`
- L323 `build`

## models/quanformer/transformer.py

- ???12459 bytes????python????297

DETR Transformer class.

- L18 `Transformer`
- L20 `Transformer.__init__`
- L42 `Transformer._reset_parameters`
- L47 `Transformer.forward`
- L62 `TransformerEncoder`
- L64 `TransformerEncoder.__init__`
- L70 `TransformerEncoder.forward`
- L86 `TransformerDecoder`
- L88 `TransformerDecoder.__init__`
- L95 `TransformerDecoder.forward`
- L127 `TransformerEncoderLayer`
- L129 `TransformerEncoderLayer.__init__`
- L146 `TransformerEncoderLayer.with_pos_embed`
- L149 `TransformerEncoderLayer.forward_post`
- L164 `TransformerEncoderLayer.forward_pre`
- L178 `TransformerEncoderLayer.forward`
- L187 `TransformerDecoderLayer`
- L189 `TransformerDecoderLayer.__init__`
- L209 `TransformerDecoderLayer.with_pos_embed`
- L212 `TransformerDecoderLayer.forward_post`
- L235 `TransformerDecoderLayer.forward_pre`
- L258 `TransformerDecoderLayer.forward`
- L272 `_get_clones`
- L276 `build_transformer`
- L289 `_get_activation_fn` ? Return an activation function given a string

## models/shared/__init__.py

- ???112 bytes????python????2


## models/shared/backbone.py

- ???6477 bytes????python????157

Backbone modules.

- L21 `FrozenBatchNorm2d` ? BatchNorm2d where the batch statistics and the affine parameters are fixed.
- L30 `FrozenBatchNorm2d.__init__`
- L37 `FrozenBatchNorm2d._load_from_state_dict`
- L47 `FrozenBatchNorm2d.forward`
- L60 `BackboneBase`
- L62 `BackboneBase.__init__`
- L79 `BackboneBase.forward`
- L90 `Backbone` ? ResNet backbone with frozen BatchNorm.
- L92 `Backbone.__init__`
- L103 `MobileNetBackboneBase` ? MobileNet backbone with frozen BatchNorm.
- L105 `MobileNetBackboneBase.__init__`
- L111 `MobileNetBackboneBase.forward`
- L122 `MobileNetBackbone`
- L123 `MobileNetBackbone.__init__`
- L132 `Joiner`
- L133 `Joiner.__init__`
- L136 `Joiner.forward`
- L150 `build_backbone`

## models/shared/matcher.py

- ???4992 bytes????python????97

Modules to compute the matching cost and solve the corresponding LSAP.

- L12 `HungarianMatcher` ? This class computes an assignment between the targets and the predictions of the network
- L20 `HungarianMatcher.__init__` ? Creates the matcher
- L36 `HungarianMatcher.forward` ? Performs the matching
- L94 `build_matcher`

## models/shared/position_encoding.py

- ???3465 bytes????python????89

Various positional encodings for the transformer.

- L12 `PositionEmbeddingSine` ? This is a more standard version of the position embedding, very similar to the one
- L17 `PositionEmbeddingSine.__init__`
- L28 `PositionEmbeddingSine.forward`
- L51 `PositionEmbeddingLearned` ? Absolute pos embedding, learned.
- L55 `PositionEmbeddingLearned.__init__`
- L61 `PositionEmbeddingLearned.reset_parameters`
- L65 `PositionEmbeddingLearned.forward`
- L79 `build_position_encoding`

## models/shared/segmentation.py

- ???15956 bytes????python????363

This file provides the definition of the convolutional heads used to predict masks, as well as the losses

- L24 `DETRsegm`
- L25 `DETRsegm.__init__`
- L37 `DETRsegm.forward`
- L65 `_expand`
- L69 `MaskHeadSmallConv` ? Simple convolutional head, using group norm.
- L75 `MaskHeadSmallConv.__init__`
- L102 `MaskHeadSmallConv.forward`
- L140 `MHAttentionMap` ? This is a 2D attention module, which only returns the attention softmax (no multiplication by value)
- L143 `MHAttentionMap.__init__`
- L158 `MHAttentionMap.forward`
- L172 `dice_loss` ? Compute the DICE loss, similar to generalized IOU for masks
- L190 `sigmoid_focal_loss` ? Loss used in RetinaNet for dense detection: https://arxiv.org/abs/1708.02002.
- L218 `PostProcessSegm`
- L219 `PostProcessSegm.__init__`
- L224 `PostProcessSegm.forward`
- L241 `PostProcessPanoptic` ? This class converts the output of the model to the final panoptic result, in the format expected by the
- L245 `PostProcessPanoptic.__init__` ? Parameters:
- L256 `PostProcessPanoptic.forward` ? This function computes the panoptic prediction from the model's predictions.
- L272 `PostProcessPanoptic.forward.to_tuple`

## postprocessing/__init__.py

- ???65 bytes????python????1


## postprocessing/area_integration.py

- ???32778 bytes????python????784

预测框积分模块：从 newtest 抽出的积分逻辑，接口与输出位置不变。
基线处理改为考虑信噪比（SNR）：估计噪声后对低 SNR 段做保守积分（噪声门限），减少把噪声当信号积分。

- L27 `_estimate_noise_and_snr` ? 从积分段 y 估计噪声与信噪比。
- L49 `integrate_with_snr_baseline` ? 带信噪比考虑的基线校正积分：
- L84 `max_consecutive` ? 连续非零长度（与 utils.quantify 一致）。
- L101 `_safe_float`
- L112 `_image_name_to_xic_index` ? 从 ROI 文件名解析 XIC 行索引：N_mz* -> N-1（0-based）。
- L128 `_integrate_area_on_segment` ? 对已有 RT 区间的 XIC 片段做积分（与 integrate_each_predicted_box 内逻辑一致）。
- L163 `_build_xic_list_from_roi_dir` ? 从 roi_dir/xic_matrix.npy 构建 xic_list（分钟制），与 run_snr_single 一致。
- L179 `integrate_from_refined_dataframe` ? 对 prediction_refined.csv 每行按 main/small/small2/small3 的 RT 区间积分，
- L242 `integrate_from_rt_intervals_df` ? 对含 rt_min/rt_max 的表（如 prediction.csv）逐行用 SNR 等算法重算 area。
- L286 `_resolve_roi_dir` ? 修正 CSV 所在目录与 XIC 目录可能不同：优先 input_root/<样本名>，否则 sample_dir 自带 xic。
- L298 `_find_refined_csv`
- L308 `run_refined_integrate_single` ? 对 post 修正结果积分：读 prediction_refined.csv，XIC 来自 roi 目录，不跑模型。
- L389 `_expected_roi_name`
- L396 `_load_roi_windows` ? 若存在 roi_windows.csv（testXIC 生成），返回 { image_name: (rt_lo, rt_hi) }，否则返回 {}。
- L413 `integrate_each_predicted_box` ? 为每个预测框计算积分信息，返回可直接写入 prediction.csv 的明细行。
- L547 `run_snr_single` ? 与 newtest.run_single 相同流程，仅积分步骤使用 integrate_each_predicted_box（SNR 基线）。
- L672 `main_cli`

## postprocessing/evaluation/__init__.py

- ???63 bytes????python????1


## postprocessing/evaluation/add_compare_error.py

- ???3094 bytes????python????85

将 compare_detail.csv 中的相对误差加入 prediction.csv 对应行。

- L16 `main`

## postprocessing/evaluation/r2_by_compound.py

- ???11365 bytes????python????262

从 integrate_prediction.py 批量输出目录读取各浓度子文件夹中的 prediction.csv，
按 (mz, q3) 对齐同一物质，用浓度(ppb)-面积 线性拟合计算 R²。

- L34 `parse_conc_ppb`
- L41 `load_long_table` ? 读取两种输入格式并拼成长表，列含 concentration_ppb、sample_folder、area：
- L126 `main`

## postprocessing/evaluation/standard_curves.py

- ???14187 bytes????python????365

从多浓度标品的 CSV 提取浓度-面积数据，绘制标准曲线（线性拟合），计算 R²。

- L24 `parse_concentration` ? 解析浓度字符串如 10ppb/0.5ppm，返回 (数值, 单位)。
- L35 `concentration_to_numeric` ? 将浓度转为数值（用于拟合），统一到 ppb 量纲。
- L49 `infer_concentration_from_path` ? 从 CSV 路径或父目录名或文件名推断浓度。
- L65 `collect_compound_area_per_file` ? 从单个 CSV 提取每个化合物 (mz, q3) 的面积。
- L119 `build_compound_data` ? 汇总所有 CSV 的浓度-面积数据。
- L141 `linear_fit_r2` ? 线性拟合 y = k*x + b，返回 (k, b, r2)。排除 x<=0 及非有限值。
- L157 `r2_5pts_after_remove_2_outliers` ? 剔除 2 个残差最大的点后，用剩余 5 点拟合并返回 R²。
- L180 `main`
- L259 `main.gather_csv`

## postprocessing/peak_refinement.py

- ???183987 bytes????python????4119

Unified workflow for:
1) post-newtest peak refinement (interval correction + small-peak + valley fallback),
2) standard (calibration) mode selection/repair,
3) sample mode filtering + final composite confidence.

- L51 `_safe_float`
- L60 `_max_consecutive_positive_points` ? Count longest consecutive run where intensity > 0.
- L80 `_scan_points_in_interval` ? Recompute scan-point metric inside an RT interval using the same
- L102 `_compound_key_from_prediction_row` ? 与 testXIC 通道一致：mz+q3，若有 native_id 则追加 slug（多 transition 同 mz/q3 可区分）。
- L117 `_safe_folder_name` ? Windows 非法路径字符替换为下划线（与 extract_json.py 一致）。
- L123 `_matplotlib_cjk_font` ? Best-effort Chinese labels for plot annotations (Windows / common fonts).
- L129 `_load_optional_csv`
- L138 `_prediction_row_for_image`
- L153 `_load_feature_table_for_mapping` ? Load feature.csv for row alignment (SNR dir first, then xic matrix dir, then --xic_dir).
- L169 `_parse_roi_index_from_image_name` ? Parse 1-based ROI index from stems like ``27_mz706.5000_q3318.2000`` (strip ``_snr...`` suffix).
- L184 `_parse_mz_from_image_stem`
- L194 `_find_feature_row_index` ? Resolve 0-based feature / xic_matrix row index.
- L241 `_feature_nominal_rt_label` ? Return (nominal_rt_min, compound_label) from feature.csv via mz/q3 or image prefix.
- L277 `_df_index_as_pred_row` ? Integer row label from a DataFrame index (e.g. prediction.csv row).
- L290 `_resolve_xic_matrix_row` ? Map one prediction row to an XIC ``intensity_mat`` row (0-based).
- L327 `_peak_lo` ? prediction 表行的左边界：新列 peak_start 优先，兼容旧列 rt_min。
- L332 `_peak_hi` ? prediction 表行的右边界：新列 peak_end 优先，兼容旧列 rt_max。
- L337 `_peer_intervals_from_group` ? 同图其他预测行的 (rt_min, rt_max)，用于边界外推时避免吃进相邻预测峰。
- L350 `_boundary_posterior_kwargs`
- L359 `_adjust_first_round_interval_kwargs`
- L373 `_effective_small_peak_rt_tol`
- L379 `_effective_valley_small_peak_rt_tol`
- L385 `_load_roi_windows`
- L398 `_normalize_roi_map_keys` ? 同时注册完整路径键与 basename，便于 SNR 子目录 roi_windows 带文件夹前缀时仍能命中。
- L411 `_merge_roi_window_maps` ? 后传入的表在同一 basename 上覆盖先传入的（通常 xic 目录的键更贴近 prediction.csv）。
- L423 `_roi_lookup_keys_for_prediction` ? prediction 图名常带 _snr… 后缀，而 ROI 表可能用无 SNR 后缀的 jpeg 名。
- L435 `_resolve_roi_rt_window`
- L466 `_peak_rt_height`
- L476 `_interval_area`
- L485 `_segment_skew`
- L501 `_secondary_min_height`
- L516 `_secondary_min_height_local` ? Candidate-local threshold:
- L541 `_adaptive_small_boundary_alpha` ? Adaptive alpha for small-peak boundary baseline:
- L566 `_small_boundary_baseline_level`
- L577 `_boundary_intensity_at_rt`
- L587 `_enforce_low_noise_boundary_with_mid_reverse` ? If boundary intensity is above low-noise baseline:
- L631 `_enforce_low_noise_boundary_with_mid_reverse._scan`
- L657 `_cap_left_shrink_for_strong_roi_secondary` ? Limit excessive right-shift on left boundary for strong roi_secondary small peaks.
- L687 `_cap_right_shrink_for_strong_roi_secondary` ? Limit excessive left-shift on right boundary for strong roi_secondary small peaks.
- L717 `_local_noise_baseline_around`
- L734 `_interval_overlap_len`
- L740 `_small_peak_is_weak`
- L759 `_constrain_adjusted_interval` ? Add reverse-move safeguard:
- L787 `_constrain_adjusted_interval._side_threshold`
- L805 `_constrain_adjusted_interval._walk_reverse`
- L836 `_refine_main_interval_near_reference` ? Re-locate main interval around the strongest peak near reference RT (if available),
- L939 `_shrink_interval_around_peak` ? Shrink a candidate interval by walking from peak to baseline crossing.
- L987 `_find_secondary_peak_by_roi_rules` ? Find secondary peak in ROI using the same threshold idea as has_secondary_peak_in_roi:
- L1045 `_build_interval_around_peak_in_segment`
- L1073 `_rt_offset_gate_with_width` ? RT gate with center/boundary constraints:
- L1100 `_chord_height_at_rt` ? 两峰顶点连线上在 x 处（RT）的线性插值强度。
- L1111 `_detect_double_peak_with_valley_in_interval` ? Detect two prominent peaks and an in-between valley inside [lo, hi].
- L1185 `_composite_conf`
- L1195 `_rt_linear_conf` ? From the figure: RT shift is the primary factor.
- L1207 `_rt_nonlinear_conf` ? Non-linear RT confidence decay:
- L1223 `_snr_factor_conf` ? Lower SNR lowers confidence (normalized to [0,1]).
- L1230 `_snr_box_over_noise_mean` ? SNR definition requested:
- L1279 `_skew_direction_boost` ? From the figure: if the standard skew is high, boost the small-peak confidence
- L1317 `_final_conf_from_figure` ? Final confidence logic (as requested):
- L1351 `_locate_xic_matrix_from_sample_csv` ? Try to locate xic_matrix.npy near sample_refined_csv path.
- L1369 `_plot_sample_final`
- L1558 `_recover_small_for_sample` ? Recover small peak for sample mode:
- L1676 `_split_overlaps_by_lowest_intensity` ? Split overlapping predicted peak intervals.
- L1701 `_split_overlaps_by_lowest_intensity._lowest_rt_in`
- L1742 `_refine_standard_row_boundaries_at_overlap_valley` ? 同一 XIC 行上 main / small / small2 若有 RT 重叠，在重叠区间内取强度最低点的 RT 作为新分界，
- L1763 `_refine_standard_row_boundaries_at_overlap_valley.collect`
- L1773 `_refine_standard_row_boundaries_at_overlap_valley.low_rt`
- L1824 `_apply_overlap_valley_split_standard_df`
- L1854 `_apply_overlap_valley_split_keep_all_df` ? 对 keep_all 多行结果按 image 分组，在重叠区间最低强度点切分 rt_min/rt_max。
- L1908 `_annotate_refined_plot_axes` ? Overlay nominal RT line + Chinese metric box (RT / compound / response / SNR / scan points).
- L2000 `_plot_refined_predictions`
- L2136 `run_post_newtest`
- L3029 `_fit_line_r2`
- L3044 `_infer_conc_from_name`
- L3055 `run_standard_mode`
- L3305 `run_sample_mode`
- L3655 `run_predict_from_standard_rt` ? 从样品目录的 xic_matrix 按各通道 XIC（可选平滑后）最高峰 RT 裁 ROI，运行 MRMPFormer 输出 prediction.csv。
- L3725 `build_parser`
- L4105 `main`

## postprocessing/snr_filter.py

- ???29070 bytes????python????765

结合 prediction.csv（CNN/Transformer 检测框）与同一样本的 mzML，在整条 XIC 上：
  - 从 mzML 读出色谱后，对强度做 **一维高斯平滑**（`scipy.ndimage.gaussian_filter1d`）；命令行用 **`--gaussian_sigma`**（或 **`--smooth_sigma`**）指定 sigma，默认 0.8；设为 **0** 表示不平滑；
  - 将预测框在 x 方向映射到保留时间区间 [rt_box_lo, rt_box_hi]；
  - **框外**所有数据点视为噪声区，估计均值与 RMS；
  - **框内**信号取 max(强度) − 噪声区均值，SNR = 信号 / RMS_noise（均基于平滑后的强度）。

- L57 `_native_id_to_str`
- L65 `_parse_q1_q3_from_text` ? 从 native_id 文本解析 Q1/Q3（兼容无 Q1=/Q3= 的厂商格式）。
- L82 `_q1_q3_from_chrom_metadata` ? 与 testmzml / pyopenms 一致：优先用 chromatogram 的 precursor / product m/z。
- L100 `_extract_q1_q3`
- L110 `_collect_chroms`
- L147 `_chrom_lookup_by_mzq3`
- L158 `_read_df_csv`
- L167 `_load_roi_windows`
- L182 `_true_rt_from_row`
- L192 `snr_outside_prediction_box` ? 框外噪声 RMS，框内峰高相对框外均值的超出为信号。
- L235 `_snr_suffix`
- L244 `_safe_jpeg_name_from_prediction_row` ? 用于保存图：原 image 列主名 + snr。
- L254 `_pixel_y_to_intensity` ? 模型框 y（0 顶、300 底）→ 与 roi_rt_mapping 一致的强度轴。
- L261 `save_roi_jpeg_with_box` ? 与 testXIC 相同的 ±1 min ROI 图，并绘制 prediction 中的红框与置信度。
- L397 `_snr_box_run_dir_name`
- L406 `_df_to_csv_safe`
- L419 `_npy_save_safe`
- L431 `_match_chrom`
- L472 `run`
- L706 `main`

## postprocessing/valley_split.py

- ???20657 bytes????python????542

Provide valley-split utilities for `run_valley_split_from_predictions.py`.

- L40 `ValleySplitParams` ? 峰谷拆分 gate：全部不通过则保持单框。
- L112 `valley_params_from_preset`
- L121 `print_valley_split_param_help` ? 与 analyze_double_peaks.py --list_params 说明风格一致。
- L142 `_find_valley_between_two_peaks` ? Find valley (argmin) between two peak indices within a segment.
- L164 `_local_maxima_indices` ? Simple local maxima detector with optional minimum height.
- L177 `_smooth_segment` ? 高斯光滑；与 analyze_double_peaks._smooth 一致。
- L186 `_pick_two_dominant_peaks` ? 旧版：取强度最高的两个局部极大（沿 RT 为 i_lo < i_hi）。
- L204 `_valley_central_ok` ? 谷（argmin）在两峰之间的相对位置须居中，避免「谷」落在某一峰边缘（单峰误拆）。
- L221 `_find_best_split_pair_prominence` ? scipy find_peaks(prominence) → 按 prominence 取前 K 个峰 → 按 RT 排序 →
- L278 `_find_best_split_pair_legacy` ? 旧版两峰 + 谷居中 + gate。
- L293 `_pair_passes_double_peak_gate` ? 对已定两峰下标执行 analyze_double_peaks 中的间距、谷深、谷位检验。
- L332 `_split_one_box_by_valley` ? Split a single predicted pixel box into up to two boxes at the valley between two peaks.
- L405 `_split_prediction_by_valley` ? Expand each predicted box by splitting at valley between two peaks.
- L473 `_prediction_to_results` ? Convert expanded prediction structure into `results` format usable by
- L491 `_plot_xic_smoothed_with_valley` ? Best-effort plot: smooth XIC trace and mark integrated intervals by rt_min/rt_max.

## preprocessing/__init__.py

- ???63 bytes????python????1


## preprocessing/coco_annotation.py

- ???34458 bytes????python????724

从人工标注 xlsx + mzML 生成 COCO 格式训练数据集（QuanFormer/MRMPFormer通用）。

- L105 `_peak_label_val` ? peak_label 字段 → int；缺失/空 → None。0=负样本，1=正样本，其余值（如 2）不入数据集。
- L119 `resolve_label_paths` ? 解析标注文件列表：
- L138 `merge_label_files` ? 合并多实验标注行为一份（入参为 [(trial, rows)]，rows 为已解析且 QC 已标记的原始行）。
- L163 `_col_letter`
- L167 `parse_labels_xlsx` ? 纯标准库解析标注 xlsx sheet1（按列字母定位，天然免疫稀疏空单元格错位）。
- L220 `parse_rt_field` ? 解析 '16.428(0.000)' / '16.428' → 16.428（分钟）；空/非法 → None。
- L229 `label_key` ? xlsx 行 → mzML native_id 键：定量离子→-1，定性离子→-2。
- L238 `group_labels_by_sample` ? 按 sample_id 分组并保留 xlsx 行序；返回 (ordered_sample_ids, {sample_id: [rows]})。
- L250 `map_samples_to_mzmls` ? sample_id ↔ mzML stem 映射。三层优先：
- L296 `build_coco_for_mzml` ? 提取（或复用）单 mzML 的 XIC 输出，生成 COCO images/annotations 条目。
- L425 `write_split` ? entries: list[(image_dict, [annotation, ...])]。写图像副本 + COCO json。
- L452 `main`
- L566 `main._stdout_to_log`

## preprocessing/coco_annotation_sim.py

- ???15488 bytes????python????326

从 MRM-XIC 模拟数据集（V6.0：xic_data JSON + V5.0 label.csv）生成 COCO 格式训练数据集。

- L52 `load_labels` ? 读 V5.0 label.csv → {(sample_id, compound_id, ion_type): row}；空峰字段为 NaN。
- L64 `collect_peak_intervals` ? V5.0 行 → [(start, end), ...]（按 RT 升序，最多 3 组）；空/非法区间剔除。
- L78 `decide_window` ? 由标注行 + 序列决定 ROI 窗口 (rt_lo, rt_hi, center, center_source)。
- L107 `build`
- L283 `main`

## preprocessing/ion_zenith.py

- ???14477 bytes????python????418

ion_zenith.py — 离子天顶算法（纯算法 + CLI）
=============================================

- L53 `_resolve_encoding` ? 尝试将非 UTF-8 编码的 mzML 自动转码为 UTF-8 临时文件。
- L100 `_cleanup_temp` ? 安全清理临时转码文件。
- L112 `_parse_rt` ? 从 pymzML spectrum 对象提取保留时间。
- L134 `_get_peaks` ? 从 pymzML spectrum 提取 (m/z, intensity) 二维 numpy 数组。
- L158 `extract_ions_from_ms1` ? 遍历 mzML 的 MS1 谱图，按 m/z 容差聚合，每个离子保留最高强度观测，写入 CSV。
- L352 `main`
- L378 `main._progress`
- L384 `main._stats`

## preprocessing/label_qc.py

- ???10362 bytes????python????228

标注数据质量 QC：RT 一致性检查（跨样品极差 + 样品内双离子极差）。

- L19 `_parse_rt_field` ? 解析 '16.428(0.000)' / '16.428' → 16.428（分钟）；空/非法 → None。
- L29 `check_label_rt_consistency` ? 对标注行列表做 RT 一致性检查。
- L64 `check_label_rt_consistency._row`
- L134 `mark_excluded_labels` ? 给命中 exclude_keys 的标注行打 _qc_excluded 标记（不删行，保持行序对齐）。
- L147 `write_qc_table` ? QC 结果表写出（CSV，utf-8-sig）。返回行数。
- L176 `build_qc_alert_markdown` ? 生成人工预警报告 Markdown 文本。
- L221 `write_qc_alert` ? 写人工预警报告 qc_alert.md（utf-8）。返回告警条数（剔除行数）。

## preprocessing/masked_roi_generator.py

- ???12799 bytes????python????314

将 prediction.csv 识别出的主峰区域删除，替换为基线/噪声，生成用于测试其余小峰识别的掩蔽图像。
与 testXIC 一致：从 XIC 取最高点，高斯光滑后提取 ROI 图像。

- L32 `load_roi_windows`
- L43 `estimate_baseline_noise` ? 从峰前、峰后区域估计基线噪声的均值和标准差。
- L64 `get_last_25pct_noise_values` ? 取图像后 frac（默认25%）RT 区间的强度值，用于随机采样填充。
- L78 `_replace_peak_random_noise` ? 主峰区替换为噪声。use_last_25pct=True 时从图像后25%随机点采样；否则用峰前峰后基线估计。
- L100 `_replace_peak_linear_interp` ? 主峰区用线性插值连接峰前、峰后基线端点。
- L115 `_replace_peak_baseline_interp` ? 主峰区用真实基线区曲线插值（适合重叠峰，保留基线漂移）。
- L132 `mask_main_peak_and_redraw` ? 将 [rt_min, rt_max] 区间的强度按 mask_method 替换，高斯光滑后按 roi_windows 提取 ROI 图（与 testXIC 一致）。
- L178 `main`

## preprocessing/xic_extraction.py

- ???46471 bytes????python????1039

提取 XIC 并生成 ROI 图像。支持两种输入模式：
1. mzML 文件：从 mzML 提取 chromatogram
2. 外部数组：由 (m/z名称、RT数组、强度数组) 生成，可选 (m/z, 预期RT) 替代 feature.csv 中的 RT

- L38 `roi_safe_name_base` ? ROI 文件名主体（无扩展名）：N_mz{母离子}[_q3{子离子}][_{化合物名}]。
- L65 `_load_standard_rt_refs` ? Load standard RT references from standard_refs.csv.
- L106 `_to_loadable_path` ? 返回 pyopenms 可加载的路径。
- L137 `_native_id_to_str`
- L145 `_label_key_from_channel` ? 标注行 → mzML native_id 键：定量离子→-1，定性离子→-2（与 coco_annotation.label_key 一致，内联避免循环依赖）。
- L154 `_parse_label_rt` ? 解析标注 rt 字段 '16.428(0.000)' / '16.428' → 分钟；空/非法 → None（与 label_qc 一致）。
- L163 `_parse_q1_q3_from_text`
- L179 `_q1_q3_from_chrom_metadata`
- L196 `_extract_q1_q3`
- L206 `render_roi_jpeg` ? 将一段 XIC 渲染为 400x300 无坐标轴 JPEG（与训练 ROI 图像素级同款）。
- L237 `extract_xic_with_pyopenms` ? 提取 XIC 并生成 ROI 图像，支持高斯平滑。
- L648 `extract_xic_from_arrays` ? 由外部输入的 (m/z名称、RT数组、强度数组) 生成 XIC 矩阵和 ROI 图像。
- L815 `generate_prediction_plots` ? 使用 MRMPFormer 模型对刚生成的 ROI 图像进行推理，并输出带预测框的可视化图。 
- L829 `run_batch_mzml` ? 批量处理 batch_dir 中所有 .mzml / .mzML 文件，每个文件的结果保存到 output_base/<文件名(无扩展名)>/

## tests/__init__.py

- ???46 bytes????python????1


## tests/test_mrmpformer_v1.py

- ???26077 bytes????python????531

MRMPFormer v1 测试（提示词 §15 全部 8 类）。

- L33 `DummyBackbone` ? 模拟 Joiner 接口：返回 ([NestedTensor], [pos])，num_channels 对齐 input_proj。
- L36 `DummyBackbone.__init__`
- L41 `DummyBackbone.forward`
- L47 `make_v1`
- L58 `make_criterion`
- L73 `make_samples`
- L80 `make_targets` ? spec: list of list[(cx,cy,w,h)] → DETR targets（cxcywh 归一化）。
- L96 `TestShape`
- L97 `TestShape.test_default_shapes`
- L116 `TestShape.test_variable_b_q_n`
- L129 `TestResidual`
- L130 `TestResidual.test_zero_residual_identity` ? FDR 末层零初始化 → Δz=0 → z2=z1、z3=z2。
- L139 `TestResidual.test_known_residual_accumulation` ? 给定非零残差，逐元素验证 z2=z1+Δz2、z3=z2+Δz3。
- L159 `TestDistributionDecode`
- L160 `TestDistributionDecode.setUp`
- L167 `TestDistributionDecode.test_onehot_expectation`
- L176 `TestDistributionDecode.test_sign_direction_and_scale` ? 正偏移向右（x 增大）、负偏移向左；尺度=初始框宽 w0 时 Δx=w0*W(n)。
- L193 `TestDistributionDecode.test_refined_edges_relative_to_initial` ? x^(k) 由累计分布相对【初始边界】解码，不做坐标残差双重累计。
- L212 `TestFinalBoxAssembly`
- L213 `TestFinalBoxAssembly.test_assembly`
- L242 `TestFinalClassificationSource`
- L243 `TestFinalClassificationSource.test_pred_logits_from_layer3_only`
- L247 `TestFinalClassificationSource.test_pred_logits_from_layer3_only.hook`
- L271 `TestBoundaryFeedbackGradient`
- L272 `TestBoundaryFeedbackGradient._grads_after_box_loss`
- L280 `TestBoundaryFeedbackGradient._grads_after_box_loss.flat_grads`
- L288 `TestBoundaryFeedbackGradient.test_feedback_gradient_flows`
- L295 `TestBoundaryFeedbackGradient.test_detach_ablation` ? detach_boundary_feedback=true：位置反馈梯度被切断（消融语义）。
- L306 `TestLosses`
- L307 `TestLosses.setUp`
- L314 `TestLosses._run`
- L321 `TestLosses.test_all_background_empty_targets`
- L328 `TestLosses.test_single_and_multi_peak`
- L342 `TestLosses.test_narrow_peak_and_degenerate_pred`
- L361 `TestLosses.test_focal_no_nan_inf`
- L368 `TestLosses.test_focal_eos_weighting` ? 背景项乘 eos_coef、前景不受影响（承接 DETR no-object 权重）。
- L391 `TestLosses.test_dynamic_l1_formula`
- L401 `TestLosses.test_pw_ciou_weight_one_when_w_eq_bar_w` ? w_gt = bar_w 时中心项权重=1 → PW-CIoU 与原 CIoU 完全一致。
- L415 `TestLosses.test_fdr_soft_labels`
- L433 `TestLosses.test_fdr_loss_backward`
- L446 `TestLegacyMigration`
- L447 `TestLegacyMigration.test_migrate_quanformer`
- L477 `TestTinyOverfit`
- L478 `TestTinyOverfit.test_overfit`
- L500 `TestTinyOverfit.test_overfit.train_epoch`

## tools/__init__.py

- ???99 bytes????python????1


## tools/_shared/__init__.py

- ???67 bytes????python????1


## tools/_shared/artifacts.py

- ???6453 bytes????python????188

评估与可视化工具共用的读取、定位类辅助函数。

- L12 `read_csv_safe` ? 读取 CSV，依次尝试 utf-8-sig / utf-8 / gbk，规避 Windows 中文路径问题。
- L30 `safe_float` ? 转 float，失败或为空时返回 default。
- L40 `load_roi_map` ? 读取 roi_windows.csv，返回 image 全路径及文件名到 (rt_lo, rt_hi) 的映射。
- L57 `resolve_rt_window` ? 在 roi_map 中匹配图像的 RT 窗口，返回 (窗口, 匹配方式)。
- L80 `image_to_row_index` ? 解析 XIC 矩阵的 0 基行索引，两者均按 1 基编号记录。
- L99 `locate_xic_npy` ? 定位 SNR 目录对应的 xic_matrix.npy。
- L121 `locate_roi_csv` ? 定位 roi_windows.csv：explicit 优先，其次 SNR 目录自身，最后到 xic_roi/xic-roi-batch 目录。
- L137 `resolve_pred_root` ? 预测输出根目录：predictions_model，缺失时退回 batch_predictions。
- L145 `resolve_roi_root` ? ROI 根目录：xic_roi，缺失时退回 xic-roi-batch。
- L153 `resolve_snr_root` ? SNR/精修结果根目录：prediction_refined，缺失时退回 snr_filtered。
- L161 `refined_core_stem` ? refined PNG 文件名去掉 _refined 后缀。
- L170 `find_row_for_refined_png` ? 在 prediction_refined.csv 中找到 refined PNG 对应的行：先精确匹配 stem，再前缀包含匹配。

## tools/_shared/chrom_json.py

- ???4040 bytes????python????138

chrom JSON 解析公共函数：时间/强度提取、Q1/Q3 解析、目录批量加载。

- L15 `parse_time_intensity` ? 从 chrom JSON 中提取 RT（秒）和强度数组。
- L54 `parse_q1_q3` ? 从 chrom JSON 中提取 Q1（前体离子 m/z）和 Q3（产物离子 m/z）。
- L89 `load_chrom_json_directory` ? 加载目录下所有 chrom JSON 文件，返回记录列表。

## tools/_shared/table_io.py

- ???3273 bytes????python????105

表格 I/O 公共函数：CSV/Excel 读取、面积解析、化合物名称标准化。

- L15 `read_table` ? 安全读取表格文件（CSV/TSV），自动尝试多种编码。
- L29 `parse_area` ? 解析面积值，支持多种格式：
- L60 `normalize_compound_name` ? 标准化化合物名称：去空格、转小写（casefold）。
- L69 `find_area_column` ? 在 DataFrame 中自动查找面积列。
- L87 `find_compound_column` ? 在 DataFrame 中自动查找化合物名称列。

## tools/batch/__init__.py

- ???28 bytes????python????1


## tools/batch/reprocess.py

- ???18945 bytes????python????496

合并 batch_post_newtest_under_snr_filtered.py 与 rerun_snr_under_snr_filtered.py。

- L63 `_snr_subdir_name` ? 根据 min_snr 生成 SNR 子目录名，与 pipeline 一致。
- L73 `_find_mzml` ? 在 mzML 目录下按 stem 查找 .mzML 文件。
- L82 `_discover_samples` ? 枚举 snr_filtered 下的样品子目录。
- L87 `run_snr` ? 运行 SNR 重跑（可选链式 post_newtest）。
- L205 `_run_post_one` ? 对单个样品运行 post_newtest。
- L257 `run_post_only` ? 仅运行 post_newtest（不重跑 SNR）。
- L370 `main`

## tools/benchmark/__init__.py

- ???22 bytes????python????1


## tools/benchmark/aggregate.py

- ???2601 bytes????python????84

JSONL/CSV 结果读取、统计聚合、多轮运行结果合并。

- L10 `load_jsonl` ? 读取 JSONL 文件，返回记录列表。
- L23 `stat_summary` ? 计算均值、中位数、标准差、min、max。
- L39 `collect_values` ? 从记录列表中提取数值。
- L52 `mean_of` ? 计算 getter 提取值的算术平均。
- L60 `mean_resource` ? 从记录的 resource 段提取特定 key 的平均值。
- L62 `mean_resource.getter`
- L71 `load_records_from_benchmark_dir` ? 从 benchmark 目录读取各次 pipeline_timing_runs.jsonl。

## tools/benchmark/report.py

- ???3549 bytes????python????98

格式化函数、CSV/JSONL 报告生成、人类可读摘要。

- L11 `fmt_ms`
- L17 `fmt_elapsed_from_ms`
- L28 `fmt_mb`
- L34 `fmt_gpu_vram_line`
- L47 `fmt_key_metric_value`
- L61 `write_benchmark_summary_csvs` ? 写入关键指标和明细 CSV。
- L90 `write_merged_jsonl` ? 写入合并的 JSONL。

## tools/benchmark/runner.py

- ???11280 bytes????python????289

重复运行 inference.cli --mode pipeline，汇总耗时与资源占用。

- L60 `_default_main_argv` ? 默认 pipeline 参数（示例，需用户通过 -- 覆盖）。
- L78 `_parse_main_output_dir`
- L85 `_set_main_output_dir`
- L95 `_append_gpu_vram_section`
- L123 `_append_key_metrics_section`
- L136 `_append_pipeline_timing_avg_table`
- L155 `_format_overall_resource_avg`
- L168 `aggregate_records`
- L209 `main`

## tools/benchmark/sampler.py

- ???3671 bytes????python????104

GPU 显存采样与资源监控。

- L12 `read_gpu_vram_mb` ? 读取 GPU 显存已用/总量(MB)。优先 pynvml，否则 nvidia-smi。
- L47 `GpuVramSampler` ? 单次 subprocess 运行期间后台采样显存已用/总量。
- L50 `GpuVramSampler.__init__`
- L60 `GpuVramSampler.start`
- L74 `GpuVramSampler._loop`
- L81 `GpuVramSampler.stop_and_stats`

## tools/diagnostics/__init__.py

- ???46 bytes????python????1


## tools/diagnostics/check_box_rt_mapping.py

- ???3792 bytes????python????117

检测「ROI 图 模型框(像素 x) → RT」是否与 prediction.csv 中 rt_min/rt_max 一致，
并在提供 xic_matrix.npy 时检查 ROI 时间窗内谱峰顶与框在 RT 上是否严重错位。

- L40 `_apex_in_window`
- L50 `check_box_rt`
- L94 `main`

## tools/diagnostics/check_chrom_snr_alignment.py

- ???7652 bytes????python????194

检查 inference.cli --mode pipeline 中与 SNR / chrom 对齐相关的风险（不跑模型，只读盘）：

- L31 `_snr_dict_key` ? 与 mzml_box_outside_snr_pipeline._chrom_lookup_by_mzq3 一致。
- L38 `resolve_chrom_json_sample_dirs` ? batch_dir 下每个含 *.json 的子文件夹 → 一个样品；根目录直铺 *.json 则整目录为一样品。
- L52 `check_sample`
- L166 `main`

## tools/diagnostics/export_case_evidence.py

- ???6648 bytes????python????181

给定 refined_plots 下的 *_refined.png，自动定位「原始数据」并出图：

- L41 `_get_plot_func`
- L49 `find_cnn_roi_jpeg` ? 在 SNR 样品目录附近查找与 roi_windows / prediction 中 image 列一致的 ROI 图像文件。
- L76 `run`
- L168 `main`

## tools/diagnostics/trace_refined_case.py

- ???4659 bytes????python????137

针对「refined_plots 里峰与绿色 Main interval 对不上」等案例，逐步对照：
  Batch newtest → SNR 输出 → prediction_refined / post_newtest

- L33 `_norm_img`
- L37 `_apex_in_window`
- L47 `_find_rows` ? stem 如 89_mznan：只匹配「基名」为 89_mznan 或 89_mznan_snr... 的行，
- L63 `_load_xic`
- L72 `_roi_map_from_df`
- L80 `main`

## tools/diagnostics/verify_rt_axis.py

- ???5834 bytes????python????149

以「原始 chrom JSON + 与 testXIC.extract_xic_from_arrays 一致的 QC/平滑/全局 linspace」
重算第 N 条化合物的对齐曲线，与磁盘上的 batch/SNR xic_matrix 第 N 行逐点对照；
并用互相关估计 batch 与 SNR 两版曲线的时间平移。

- L34 `_qc_passes`
- L42 `_make_global_linspace`
- L49 `main`

## tools/evaluation/__init__.py

- ???0 bytes????python????0


## tools/evaluation/compare_reports.py

- ???3057 bytes????python????79

汇总多份 evaluation_report.json → 模型对比报告（markdown）。

- L33 `main`

## tools/evaluation/dump_queries.py

- ???7235 bytes????python????171

逐 query 诊断：打印每张 ROI 图全部 query 的 P(峰)/P(背景) + 框位置 + 与 GT 的偏差，
验证"好框挂在背景概率高的 query 上"（v1 取类 bug 的铁证）。

- L35 `load_model` ? 与 build_predictor 同源的加载方式。
- L54 `main`

## tools/evaluation/evaluate_baseline.py

- ???27417 bytes????python????549

基线一键精度评测：跑推理管线 → 对齐人工标注 → 输出峰检测 + 定量指标。

- L61 `_parse_area` ? xlsx area 字段（字符串）→ float；空/非法 → NaN。
- L72 `_parse_gt_peaks` ? 标注行 → [(start, end, area), ...]。
- L97 `match_image` ? 单张 ROI 图内两轮贪婪匹配（统一起止偏差口径）。
- L159 `prf`
- L168 `run_inference_for_mzml` ? subprocess 调 inference.cli --mode pipeline（与手工执行完全一致），返回 (pred_csv, feature_csv)。
- L211 `_load_pred_rows` ? 读取预测框行列表 [{image, rt_min, rt_max, score, area}]，兼容两种格式：
- L250 `_build_pred_by_img`
- L261 `evaluate` ? pred_feat_map: {stem: {"pred": path, "feat": path}}。返回 (metrics, details_df, area_df)。
- L391 `evaluate._mean`
- L394 `evaluate._med`
- L420 `_fmt`
- L424 `main`

## tools/evaluation/evaluate_sim_test.py

- ???12865 bytes????python????272

模拟测试集（MRM-XIC V6.0：test_easy / test_medium / test_hard）一键精度评测。

- L44 `load_coco` ? 读 train split 的 COCO json + roi_windows.csv。
- L77 `gt_to_rt` ? COCO bbox [x1, y, w, h]（像素）→ (start, end)（分钟，窗口线性逆映射）。
- L84 `run_prediction` ? build_predictor 推理 → {file_name: [{rt_min, rt_max, score}, ...]}（score 降序）。
- L105 `evaluate_grid` ? score × tol 网格（含每档 TP/FP/FN 与特殊峰命中）。
- L130 `evaluate_headline` ? 核心档：指标 + RT 偏差 + 峰型分组 + 逐条明细。
- L194 `main`

## tools/evaluation/evaluate_special_isolation.py

- ???7243 bytes????python????179

[隔离实验] 特殊峰专项评估。

- L45 `_read_tags` ? 读取 {roi_id: tag}（v2 标签只改峰边界，tag 列保留原样并追加 ;v2 标记）。
- L63 `evaluate_special`
- L137 `evaluate_special.prf`
- L153 `main`

## tools/evaluation/fdr_trace.py

- ???20980 bytes????python????445

FDR 逐层轨迹可视化（MRMPFormer v1）：展示 FDR 三层精化对峰边界的逐层改善。

- L65 `parse_gt_peaks` ? 标注行 → [(start_min, end_min), ...]；peak_label=0 无 GT；兼容旧单数列。
- L83 `gen_rois` ? 调 inference.cli --mode roi 生成 ROI（与正式推理管线同一路径，B 范式标注驱动）。
- L100 `load_model` ? 从 checkpoint（含训练 args）重建模型并加载权重。
- L121 `box_px_to_min` ? 像素 x ∈[0,400] → 分钟（与 roi_rt_mapping.box_x_to_rt_minutes 一致）。
- L127 `interval_tiou`
- L133 `main`
- L283 `main._layer_stats`
- L392 `main._fmt`

## tools/evaluation/inference_report.py

- ???18091 bytes????python????405

推理报告生成器：汇总 pipeline 输出中各样品的精修结果（prediction_refined.csv），
为精修框补算峰面积（复用 area_integration.integrate_from_refined_dataframe），
并输出（Step 7c 新命名，旧固定名接口已删除）：

- L38 `image_row_number` ? 从 ROI 文件名前缀解析 1-based feature 行号：N_mz* -> N。
- L49 `parse_compound_from_nid`
- L56 `parse_compound_from_image`
- L63 `load_feature_map` ? feature.csv: 'Compound Name'(1-based) -> native_id。
- L81 `collect_refined` ? snr_root 下收集 样品 -> [refined_csv,...]（支持 <样品>/SNR_box_*/ 与 <样品>/ 布局）。
- L93 `small_count` ? 次峰数：仅统计存在实际 RT 区间的次峰（small/small2/small3）。
- L102 `build_rows` ? 逐样品积分并构造合并明细行。
- L164 `load_timing`
- L176 `qc_summary_text`
- L186 `render_matrix` ? 化合物 × 样品：主峰面积（—=未检出/无该 ROI）。
- L232 `_appendix`
- L253 `render_report`
- L302 `_fmt`
- L306 `generate_for_pipeline` ? 生成推理报告，返回摘要 dict。
- L376 `main`

## tools/evaluation/refine_ablation.py

- ???22275 bytes????python????452

框修正（peak_refinement）消融对比实验。

- L47 `_non_nan`
- L51 `parse_rt` ? 解析 '16.428(0.000)' / '16.428' / NaN → 分钟；空/非法 → None。
- L62 `load_labels`
- L93 `image_row_number`
- L98 `load_feature_map`
- L115 `integrate_box` ? 用与报告一致的 SNR 基线积分器补算任意 RT 区间面积。
- L128 `build_pred_by_image` ? image basename -> [(rt_min, rt_max, score, area)]（含积分补算）。
- L143 `build_refined_by_image` ? image basename -> [(tag, rt_min, rt_max, score, area)]。
- L167 `match_boxes` ? boxes=[(rt_min,rt_max,score,area)] vs gt_peaks；返回 (hits, fp, fn)。
- L184 `_fmt2`
- L195 `_r2`
- L203 `_rsd`
- L208 `main`
- L325 `main.prf`

## tools/evaluation/visualize_compare.py

- ???14180 bytes????python????308

双模型预测 vs 人工标注 可视化复核工具。

- L44 `load_pred`
- L57 `verdict_of` ? 返回 (best_row, dev_start, dev_end, verdict)。verdict: TP / FP / FN / 无标注
- L73 `main`

## tools/experiments/__init__.py

- ???34 bytes????python????1


## tools/experiments/common.py

- ???3190 bytes????python????104

江南、欧陆实验共用的比较和报表辅助函数。

- L24 `load_csv_areas` ? 从 CSV 加载化合物→面积映射。返回 {normalized_name: area}。
- L44 `load_os_areas` ? 从 OS txt（人工加标）加载指定样品的化合物→面积映射。
- L75 `compute_relative_error` ? 相对误差 = /csv - os/ / os。os 为 0 或 None 时返回 None。
- L84 `pair_area_comparison` ? 对比两组面积映射，返回逐化合物比较结果列表。

## tools/experiments/jiangnan/__init__.py

- ???22 bytes????python????1


## tools/experiments/jiangnan/area_compare.py

- ???4543 bytes????python????116

对比江南大学农残 CSV 与人工加标 OS.txt 的 Area，统计相对误差。

- L37 `compare_one` ? 对比单个 CSV 与 OS 样本。返回 (details, summary)。
- L65 `main`

## tools/experiments/jiangnan/area_ratio.py

- ???6473 bytes????python????176

江南大学农残 两浓度 CSV 与 OS 面积比对比。

- L65 `compute_ratios` ? 计算两浓度面积比及差值。返回 DataFrame。
- L104 `main`

## tools/experiments/jiangnan/compound_presence.py

- ???4559 bytes????python????131

对比两浓度 CSV + 两 OS 样本中，哪些化合物名只出现在某一侧（only）。

- L20 `_display_name`
- L27 `_classify_only` ? 返回 only 类型标签。
- L58 `main`

## tools/experiments/oulu/__init__.py

- ???16 bytes????python????1


## tools/experiments/oulu/area_compare.py

- ???5842 bytes????python????163

读取欧路标注结果目录下全部 CSV，按第一列「样品名」映射到欧陆 pipeline，做面积对比。

- L40 `load_pipeline_predictions` ? 加载 pipeline 的 prediction_refined.csv。
- L58 `match_by_mz_q3` ? 按 (mz, q3) 在 prediction 中匹配标准品行。
- L80 `main`

## tools/experiments/oulu/full_report.py

- ???6669 bytes????python????182

欧路标注标准 vs snr_filtered 主峰(main_area)面积全量对比报告。

- L50 `_load_snr_prediction` ? 加载 SNR 过滤后的 prediction_refined.csv。
- L62 `_build_prediction_index` ? 构建 (mz, q3) → row 索引。
- L75 `_match_prediction` ? 按 (mz, q3) 匹配。
- L92 `main`

## tools/export_onnx.py

- ???7382 bytes????python????170

导出 MRMPFormer checkpoint 为 ONNX。

- L42 `MRMPFormerOnnx` ? ONNX 包装：内置预处理 + 后处理，输出像素坐标结果。
- L45 `MRMPFormerOnnx.__init__`
- L52 `MRMPFormerOnnx.forward`
- L75 `load_model`
- L100 `export`
- L131 `verify` ? Torch vs ONNX Runtime 数值一致性 + 动态尺寸验证。
- L156 `main`

## tools/maintenance/__init__.py

- ???44 bytes????python????1


## tools/maintenance/force_blank_negative.py

- ???5971 bytes????python????143

强制指定样品（默认 BLANK 空白样）的所有行为负样本：peak_label / peak_count → 0。

- L28 `_load_xlsx` ? 返回 (共享字符串列表, sheet1 根元素)。
- L37 `_col_letter`
- L41 `_cell_value`
- L48 `main`

## tools/maintenance/organize_results.py

- ???4529 bytes????python????133

将 pipeline 输出目录下分散在各级子目录中的文件，按「类别」归并到少数文件夹。

- L27 `_safe_flat_name`
- L35 `_is_under`
- L43 `organize`
- L114 `main`

## tools/maintenance/regenerate_xic.py

- ???2726 bytes????python????72

从 chrom JSON 目录重新生成与 testXIC.extract_xic_from_chrom_json_dir 相同的输出，
包括 xic_matrix.npy、feature.csv、roi_windows.csv、ROI jpeg、pipeline_qc_excluded.csv（若有剔除）。

- L28 `main`

## tools/maintenance/subsample_coco.py

- ???4029 bytes????python????92

从现成 COCO 训练集随机抽取 N 张图像，复制生成独立数据集目录（可直接被 train.py 消费）。

- L22 `_write_json`
- L27 `_copy_images`
- L42 `main`

## tools/mzml/__init__.py

- ???39 bytes????python????1


## tools/mzml/chromatogram.py

- ???8451 bytes????python????251

mzML 色谱查看与导出工具。

- L27 `_load_mzml_experiment` ? 延迟加载 pyopenms 并返回 MSExperiment。
- L39 `_collect_native_ids` ? 收集所有色谱的 native ID。
- L49 `_find_chrom_index` ? 按名称或序号查找色谱索引。
- L74 `_chrom_ids_from_bytes` ? 从 mzML 原始字节中提取 chromatogram id（兼容中文编码问题）。
- L98 `cmd_list` ? 列出所有色谱。
- L141 `cmd_show` ? 显示单条色谱摘要。
- L180 `cmd_export` ? 导出单条色谱数据点。
- L215 `main`

## tools/mzml/common.py

- ???2989 bytes????python????88

mzML 公共工具：文件打开、编码兼容、native_id 修复、Q1/Q3 读取。

- L13 `decode_native_id` ? 修复 pyopenms 在 Windows 上的中文乱码问题。
- L45 `parse_q1_q3_from_native_id` ? 从 native_id 文本解析 Q1/Q3（兼容无 Q1=/Q3= 的厂商格式）。
- L62 `_load_ms_experiment_pyopenms` ? 使用 pyopenms 加载 mzML 文件。
- L75 `get_chromatogram_count` ? 返回 MSExperiment 中 chromatogram 数量。
- L83 `get_spectrum_count` ? 返回 MSExperiment 中 spectrum 数量。

## tools/mzml/inspect.py

- ???7875 bytes????python????213

将 mzML 中的可读内容导出为 CSV。

- L34 `_chromatogram_ids_from_xml` ? 收集 <chromatogram id="…"/>；若均有 index 属性则按 index 排序对齐 pyopenms，否则按文档顺序。
- L54 `_load_mzml` ? 延迟加载 pyopenms。
- L67 `inspect_mzml` ? 检查 mzML 文件并返回摘要。
- L189 `main`

## tools/tests/__init__.py

- ???16 bytes????python????1


## tools/tests/test_shared.py

- ???6142 bytes????python????192

_shared 模块单元测试。

- L21 `TestNormalizeCompoundName`
- L22 `TestNormalizeCompoundName.test_normal`
- L25 `TestNormalizeCompoundName.test_casefold`
- L28 `TestNormalizeCompoundName.test_empty`
- L31 `TestNormalizeCompoundName.test_none`
- L36 `TestParseArea`
- L37 `TestParseArea.test_float`
- L40 `TestParseArea.test_int`
- L43 `TestParseArea.test_scientific`
- L46 `TestParseArea.test_scientific_with_space`
- L49 `TestParseArea.test_na`
- L54 `TestParseArea.test_less_than_2_points`
- L57 `TestParseArea.test_empty`
- L60 `TestParseArea.test_none`
- L63 `TestParseArea.test_nan`
- L66 `TestParseArea.test_thousands_separator`
- L70 `TestParseQ1Q3`
- L71 `TestParseQ1Q3.test_dict_format`
- L77 `TestParseQ1Q3.test_scalar_format`
- L83 `TestParseQ1Q3.test_missing`
- L89 `TestParseQ1Q3.test_invalid_string`
- L96 `TestParseTimeIntensity`
- L97 `TestParseTimeIntensity.test_normal_minute`
- L109 `TestParseTimeIntensity.test_second_unit`
- L117 `TestParseTimeIntensity.test_too_short`
- L123 `TestParseTimeIntensity.test_length_mismatch`
- L129 `TestResolveRtWindow`
- L130 `TestResolveRtWindow.setUp`
- L136 `TestResolveRtWindow.test_exact_match`
- L141 `TestResolveRtWindow.test_basename_match`
- L145 `TestResolveRtWindow.test_no_match`
- L150 `TestResolveRtWindow.test_empty`
- L155 `TestImageToRowIndex`
- L156 `TestImageToRowIndex.test_compound_name_numeric`
- L160 `TestImageToRowIndex.test_image_prefix`
- L164 `TestImageToRowIndex.test_no_match`
- L168 `TestImageToRowIndex.test_boundary_zero`
- L172 `TestImageToRowIndex.test_boundary_invalid`
- L177 `TestSafeFloat`
- L178 `TestSafeFloat.test_normal`
- L181 `TestSafeFloat.test_nan`
- L184 `TestSafeFloat.test_invalid`
- L187 `TestSafeFloat.test_default`

## tools/visualization/__init__.py

- ???46 bytes????python????1


## tools/visualization/plot_gt_vs_pred.py

- ???16023 bytes????python????322

人工标注框 vs 模型预测框 对照可视化。

- L57 `_cjk_font`
- L62 `_load_xic` ? 读 xic_matrix.npy 的第 row_idx 行（0-based）→ (rt, y)，失败返回 None。
- L79 `_fmt`
- L83 `plot_one` ? 画单张对照图。
- L152 `main`

## tools/visualization/plot_prediction_rt.py

- ???5267 bytes????python????147

将 newtest 输出的 prediction.csv 中「已映射到 RT」的 rt_min/rt_max 画在原始 xic 上，
风格对齐 run_unified_peak_workflow._plot_refined_predictions（refined_plots：蓝线 + 绿色 Main interval）。

- L40 `_matplotlib_cjk_font`
- L45 `plot_one`
- L90 `main`

## tools/visualization/plot_refined_xic.py

- ???5639 bytes????python????157

根据 refined_plots 下的结果图（*_refined.png），从同批次的 xic_matrix.npy 重绘「原始」XIC。

- L38 `_matplotlib_cjk_font`
- L43 `plot_xic_from_refined_png`
- L134 `main`

## train.py

- ???27013 bytes????python????452

- L18 `get_args_parser`
- L209 `main`

## utils/__init__.py

- ???87 bytes????python????2


## utils/adaptive_integration.py

- ???1825 bytes????python????46

根据 ROI 质量参数自适应选择积分方法，降低相对误差。

- L18 `select_integration_method` ? 根据质量参数选择积分方法。

## utils/detection_helper.py

- ???4560 bytes????python????87

- L6 `PeakList`
- L7 `PeakList.__init__`
- L13 `PeakList.from_df`
- L21 `PeakList.to_csv`
- L24 `PeakList.to_skyline`
- L42 `PeakList.readXCMSPeakList`
- L58 `PeakList.runXCMS`
- L68 `get_features`

## utils/find_peaks.R

- ???8449 bytes????text????244


## utils/integrate_peak_adaptive.py

- ???3782 bytes????python????91

峰自适应积分：针对双峰、宽峰等场景，解决线性端点基线过高导致积分偏小的问题。

- L17 `_find_valleys` ? 在 y 中找局部谷点（局部最小值）。
- L35 `integrate_peak_adaptive` ? 峰自适应积分：谷-谷基线 + 保守回退。

## utils/io_utils.py

- ???8037 bytes????python????227

- L15 `time_master` ? Decorator to measure execution time of a function.
- L18 `time_master.wrapper_time_master`
- L31 `get_files` ? Recursively find all files with given suffix in the directory.
- L59 `validate_file_path` ? Ensure the given path points to an existing file.
- L72 `load_images` ? Load all ROI images from a directory (supports .jpg, .jpeg, etc.).
- L96 `replace_special_characters` ? Replace special characters in compound names that may cause file/path issues.
- L111 `load_features` ? Load and clean targeted feature CSV file (same format as main.py / testXIC output).
- L183 `export_results` ? Export quantification results to CSV.

## utils/mzml_chromatogram_ids.py

- ???6268 bytes????python????203

从 mzML 文件可靠读取 chromatogram 的 id（UTF-8），以及生成可放在文件名里的 nid 片段。

- L16 `_decode_attr_value_bytes`
- L40 `chromatogram_ids_from_mzml_xml`
- L67 `chromatogram_ids_from_mzml_raw_bytes`
- L99 `pick_chrom_native_ids_from_mzml_file`
- L109 `resolve_native_ids_for_chromatograms` ? 与 pyopenms 色谱列表等长：优先 mzML XML/字节 id，否则 chrom.getNativeID()。
- L127 `filesystem_slug_for_native_id` ? 用于文件名：Unicode 规范化、去掉 Windows 非法字符、过长则截断 + 短 hash。
- L156 `roi_image_stem` ? 与 newtest / SNR 流水线兼容的前缀：{N}_mz... ；后缀含 q3 与 nid。
- L185 `transition_dedup_key`

## utils/mzml_load.py

- ???6321 bytes????python????189

Load mzML via pyopenms with Windows path and invalid UTF-8 repair.

- L11 `is_valid_utf8`
- L19 `_repair_invalid_utf8_bytes` ? 局部解码修复：ASCII 段原样保留；非 ASCII 连续段依次试 UTF-8 → GBK → GB18030 解码，
- L48 `repair_mzml_bytes_for_openms` ? Replace invalid UTF-8 bytes so OpenMS XML parser can read the file (GBK preserved).
- L59 `_mzml_load_ok`
- L67 `load_ms_experiment` ? Load mzML into MSExperiment.
- L78 `load_ms_experiment._try_load`
- L140 `_load_ms_experiment_mzml`
- L145 `fix_mzml_encoding` ? 一次性批量修复：将 root_dir 递归下所有 *.mzML 的非 UTF-8 字节就地转正为 UTF-8。
- L172 `main`

## utils/plot_xic_peaks.py

- ???6105 bytes????python????160

共享 XIC 峰标注绘图核心（需求 2/7）。

- L30 `_smooth_nonneg`
- L40 `_scan_points_in_interval` ? 峰区间 [rt_lo, rt_hi] 内强度>0 的最长连续点数（不是整张 ROI 图）。
- L54 `plot_xic_with_queries` ? 在已有 Axes 上绘制 XIC 曲线与多 query 峰标注。

## utils/predict_utils.py

- ???12842 bytes????python????304

- L26 `rescale_bboxes` ? 将归一化的 bbox 坐标 (cx, cy, w, h) 转换为原始图像像素坐标 (x1, y1, x2, y2)
- L38 `predict` ? 对 ROI 图像列表进行预测。verbose=False 时不打印逐图 DEBUG，加快运行。
- L108 `plot_results` ? 可视化预测结果。无检测时也保存图像，并标注 "No detection"。
- L169 `build_predictor`

## utils/quantify.py

- ???14259 bytes????python????308

- L23 `get_baseline_endpoint_heights` ? 用积分窗口左/右端附近多点的分位数作为基线端点高度。
- L43 `get_baseline_endpoint_heights._robust_height`
- L53 `_right_baseline_expand_until_high` ? 从右侧窗口起点向右逐点累加求平均，直到某一点远高于当前平均值则停止，
- L85 `_get_minval_noise_right_baseline_params` ? 谷点-噪声基线参数。返回 (rt_left_lowest, y_left_lowest, y_right_avg)。
- L119 `get_baseline_minval_noise_right` ? 返回 minval_noise_right 基线的 y 值数组（用于绘图）。
- L129 `integrate_with_baseline_minval_noise_right` ? 谷点-噪声基线积分：适用于前肩峰/双峰场景。
- L154 `integrate_with_baseline_correction_avg` ? 线性基线校正积分：用窗口两端附近多点的分位数作为端点高度（优先窗口外），再线性插值作基线。
- L179 `integrate_with_external_baseline` ? 外部基线积分：用用户提供的 (x[], y[]) 定义基线，在模型输出的积分上下限 [left, right] 内，
- L208 `integrate_with_baseline_correction` ? 线性基线校正积分：用窗口两端点做线性基线，只积分高于基线的部分。
- L229 `quantify` ? 定量积分函数
- L293 `max_consecutive`

## utils/roi_quality_params.py

- ???2924 bytes????python????70

ROI 积分段质量参数：SNR、基线斜率、峰宽等，用于预测积分误差分析与自适应策略。

- L16 `compute_roi_quality_params` ? 从积分段 (x=RT, y=intensity) 计算质量参数。
- L67 `compute_roi_quality_params_minimal` ? 仅计算 SNR，用于轻量调用。返回 (snr, noise_std, baseline_level)。

## utils/roi_rt_mapping.py

- ???3200 bytes????python????76

预测框像素坐标 → XIC 上 RT 窗口（分钟）的映射。

- L19 `rt_to_pixel_x` ? RT（分钟）→ 图像像素 x，与 testXIC 一致。
- L27 `intensity_to_pixel_y` ? 强度 → 图像像素 y（0=顶），与 testXIC 绘图一致。
- L36 `rt_window_bounds_minutes` ? 与 testXIC 裁剪窗口一致：apex±1 min，再夹到 rt_axis 的 [min, max]。
- L53 `box_x_to_rt_minutes` ? 单边界：像素 x ∈ [0, 400] 线性对应 [rt_lo, rt_hi]（分钟）。
- L60 `box_to_rt_range` ? 由预测框 (x1,y1,x2,y2) 得到积分用 RT 区间 [left, right]（分钟）。

## utils/torch_device.py

- ???2867 bytes????python????81

PyTorch 推理设备选择：优先 CUDA，不可用则回退 CPU。

- L9 `resolve_torch_device` ? 检测 CUDA 并返回 torch.device。
- L31 `reset_torch_device_cache` ? 测试或切换环境时清空缓存。
- L37 `load_torch_checkpoint` ? 加载本地 .pth 检查点（含 model/args 等完整对象）。
- L51 `_print_device_info`

## utils/xic_peak_utils.py

- ???16202 bytes????python????386

XIC 峰分析工具：SNR 计算、次峰检测。
用于两轮识别流程中的 XIC 条件筛选。

- L10 `get_noise_regions_outside_box` ? 在预测框 [rt_min, rt_max] 外取左右噪声区。
- L31 `compute_snr_outside_box` ? 以预测框外区域为噪声参考的峰-峰信噪比。
- L72 `_compute_snr_peak_to_peak` ? 段内估计 SNR（备用）。
- L99 `get_last_25pct_avg_noise` ? 图像后 frac（默认25%）RT 区间的平均强度，作为噪声水平。
- L109 `roi_full_low_decile_mean_intensity` ? 全 ROI（整条 XIC 采样）强度排序后，取最低 bottom_frac 比例的点对其取平均，
- L122 `one_sided_edge_stop_threshold_stable_tail_mean` ? 边框外推截停阈值：在峰顶沿该侧 max_span 内，取靠外侧（沿 RT）tail_frac 比例的采样点，
- L186 `one_sided_low_noise_baseline` ? 从峰顶沿移动方向一侧（左：rt<=peak；右：rt>=peak）在有限 span 内，
- L221 `has_secondary_peak_in_roi` ? 在 ROI 窗口 [rt_lo, rt_hi] 内检查是否存在次峰。
- L271 `compute_local_snr` ? 本地 SNR（整谱场景专用）：噪声参考取"本峰边界到最近相邻峰边界之间"的安静区段，
- L313 `compute_local_snr._quiet_points` ? 从单侧扇区筛出低强度安静点；点数不足回退整扇区。

## Python ?????

??? 122 ?????????? file_inventory.json ?????????????
