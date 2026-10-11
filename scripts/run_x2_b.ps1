# X2 phase 3: variant B (sigma^2 as input, encoder + decoder) fine-tuned with the HYBRID reward.
# Question: once training is correct (real clock, expected lateness), do sigma^2 INPUTS help?
# Compared against M0_ftH (same recipe: 15k analytic `_last` + 5k hybrid, lr 1e-4) -> both at 20k.
#
#   B_ftH : variant b + 5k steps, hybrid reward, init checkpoints\B_w1_15k_s<seed>_last.pt
#
# Flags = B's original training flags, read from the headers of runs\train_B_w1_15k_s7.log,
# runs\train_B_w1_15k_s1234.log and runs\B_w1_s2024.txt (variant B, omega 1, beta_var range
# [0.15, 0.6], reward analytic), plus the X2 fine-tune recipe (--steps 5000 --lr 1e-4), which is
# what M0_ftH used. --init-from refuses a variant / simulator / tardiness_weight mismatch.
# Smoke-tested: 20 steps on CPU from B_w1_15k_s2024_last.pt with --reward-mode hybrid ran and
# printed "initialised from ... | variant b | 15000 steps | its reward analytic".
#
# No $ErrorActionPreference = "Stop" (PowerShell 5.1 would abort on any stderr line).
# Run from the repo root:
#     powershell -ExecutionPolicy Bypass -File .\scripts\run_x2_b.ps1 2>&1 | Tee-Object -FilePath runs\x2_b.console.log

$common = @("--variant", "b", "--omega", "1", "--beta-var-range", "0.15", "0.6",
            "--steps", "5000", "--lr", "1e-4", "--reward-mode", "hybrid")

New-Item -ItemType Directory -Force runs | Out-Null
foreach ($seed in @(2024, 7, 1234)) {
    $init = "checkpoints\B_w1_15k_s$($seed)_last.pt"
    $name = "B_ftH_s$seed"
    if (-not (Test-Path $init)) {
        Write-Host "=== MISSING $init -- skipping $name ==="
        continue
    }
    $log = "runs\train_$name.log"
    Write-Host "=== $(Get-Date -Format 'HH:mm') starting $name (b, hybrid) -> $log ==="
    python -u -m scripts.train_stoch --seed $seed --init-from $init `
        --out "checkpoints\$name.pt" @common 2>&1 |
        Tee-Object -FilePath $log
    Write-Host "=== $(Get-Date -Format 'HH:mm') finished $name ==="
}
Write-Host "All runs done. Check: Get-ChildItem checkpoints\B_ftH_*_last.pt"