# Two EXTRA seeds (42, 99) for the from-scratch comparison of run G, both arms:
#   M0_w1_15k_s<seed>  : M0, ANALYTIC reward (the reference arm)
#   M0H_w1_15k_s<seed> : M0, HYBRID reward   (from scratch)
# 4 runs x ~2.7 h. Same flags as the original 15k runs (scripts\run_seeds.ps1) and as
# run_m0_hybrid_scratch.ps1; the two arms differ only in --reward-mode.
#
# Decided BEFORE these runs (13 Oct): every run is reported, including a collapse
# (val frozen at 22.3909). Seed 2024 stays reported as collapsed and is not replaced.
# The 4-seed result (7, 1234, 42, 99) is reported next to the 2-seed result of run G.
#
# Interleaved by seed so a stopped night still leaves a complete pair.
# Check after ~15 min of each run: val at step 300 must be well below 15.
# No $ErrorActionPreference = "Stop" (PowerShell 5.1 would abort on any stderr line).
# Run from the repo root:
#     powershell -ExecutionPolicy Bypass -File .\scripts\run_m0_extra_seeds.ps1 2>&1 | Tee-Object -FilePath runs\m0_extra_seeds.console.log

$common = @("--variant", "m0", "--omega", "1", "--beta-var-range", "0.15", "0.6",
            "--steps", "15000", "--save-every", "2500")

$runs = @()
foreach ($seed in @(42, 99)) {
    $runs += @{ seed = $seed; mode = "analytic"; name = "M0_w1_15k_s$seed" }
    $runs += @{ seed = $seed; mode = "hybrid";   name = "M0H_w1_15k_s$seed" }
}

New-Item -ItemType Directory -Force runs | Out-Null
foreach ($r in $runs) {
    if (Test-Path "checkpoints\$($r.name)_last.pt") {
        Write-Host "=== EXISTS checkpoints\$($r.name)_last.pt -- skipping (no overwrite) ==="
        continue
    }
    $log = "runs\train_$($r.name).log"
    Write-Host "=== $(Get-Date -Format 'HH:mm') starting $($r.name) (m0, $($r.mode)) -> $log ==="
    python -u -m scripts.train_stoch --seed $r.seed --reward-mode $r.mode `
        --out "checkpoints\$($r.name).pt" @common 2>&1 | Tee-Object -FilePath $log
    Write-Host "=== $(Get-Date -Format 'HH:mm') finished $($r.name) ==="
}
Write-Host "All runs done. Check: Get-ChildItem checkpoints\M0*_w1_15k_s42_last.pt, checkpoints\M0*_w1_15k_s99_last.pt"