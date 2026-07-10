# Clean up all unnecessary tabular and ML files from the MoCha2026 root directory
$filesToDelete = @(
    "train_features_fusion.csv",
    "train_features_baseline.csv",
    "train_features_momask.csv",
    "classifier.pth",
    "classifier_dann.pth",
    "classifier_fusion.pth",
    "classifier_pca.pth",
    "classifier_robust.pth",
    "best_model.joblib",
    "best_model_metadata.json",
    "fusion_scaler_mean.npy",
    "fusion_scaler_std.npy",
    "robust_scaler_mean.npy",
    "robust_scaler_std.npy",
    "scaler_mean.pt",
    "scaler_std.pt",
    "pca_components.pt",
    "pca_mean.pt",
    "variance_filter_indices.npy",
    "evaluate_leakage.py",
    "extract_features.py",
    "test_baseline_leakage.py",
    "test_baseline_only.py",
    "test_baseline_only_v2.py",
    "train_dann.py",
    "train_fusion.py",
    "train_inception.py",
    "train_lgbm.py",
    "train_lgbm_pca.py",
    "train_pca_pytorch.py",
    "train_pytorch_mlp.py",
    "train_variance_filter.py"
)

foreach ($file in $filesToDelete) {
    if (Test-Path $file) {
        Remove-Item -Path $file -Force
        Write-Host "Deleted $file" -ForegroundColor Green
    }
}

Write-Host "Cleanup complete! Directory is now ready for Kaggle." -ForegroundColor Cyan
