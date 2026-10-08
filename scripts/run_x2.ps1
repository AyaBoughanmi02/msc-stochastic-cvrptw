# X2 phase 1: fine-tune the M0 checkpoints for 5,000 more steps, analytic (control) vs hybrid.
# Run from the repo root:
#     powershell -ExecutionPolicy Bypass -File .\scripts\run_x2.ps1
#
# $common = the flags the original M0_w1_15k_* runs were trained with (from run_seeds.ps1:
# --omega 1 --beta-var-range 0.15 0.6; everything else at its default), plus the fine-tune
# settings. --init-from refuses a simulator / variant / tardiness_weight mismatch, so a wrong
# flag stops the run at once instead of training the wrong thing.
# Interleaved by seed (control then hybrid) so a stopped set still leaves complete pairs.
# No $ErrorActionPreference = "Stop": under Windows PowerShell 5.1 any stderr line would
# abort the whole set.

$common = @("--variant", "m0", "--omega", "1", "--beta-var-range", "0.15", "0.6",
            "--steps", "5000", "--lr", "1e-4")

$runs = @()
foreach ($seed in @(2024, 7, 1234)) {
    $init = "checkpoints\M0_w1_15k_s$($seed)_last.pt"
    $runs += @{ seed = $seed; init = $init; mode = "analytic"; name = "M0_ftA_s$seed" }
    $runs += @{ seed = $seed; init = $init; mode = "hybrid";   name = "M0_ftH_s$seed" }
}

New-Item -ItemType Directory -Force runs | Out-Null
foreach ($r in $runs) {
    if (-not (Test-Path $r.init)) {
        Write-Host "=== MISSING $($r.init) -- skipping $($r.name) ==="
        continue
    }
    $log = "runs\train_$($r.name).log"
    Write-Host "=== $(Get-Date -Format 'HH:mm') starting $($r.name) ($($r.mode)) -> $log ==="
    python -u -m scripts.train_stoch --seed $r.seed --init-from $r.init `
        --reward-mode $r.mode --out "checkpoints\$($r.name).pt" @common 2>&1 |
        Tee-Object -FilePath $log
    Write-Host "=== $(Get-Date -Format 'HH:mm') finished $($r.name) ==="
}
Write-Host "All runs done. Check: Get-ChildItem checkpoints\M0_ft*_last.pt"