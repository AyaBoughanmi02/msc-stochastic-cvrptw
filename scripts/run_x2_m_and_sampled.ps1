# X2 phase 2: (a) the third reward arm on M0 (sampled), (b) control vs hybrid on M (D-prime).
# All fine-tunes: +5,000 steps from the 15k _last checkpoints, lr 1e-4 cosine, same seeds.
#   M0_ftS_s<seed> : M0, reward sampled  (real clock, REALISED lateness)
#   M_ftA_s<seed>  : M,  reward analytic (control: mean clock, expected lateness)
#   M_ftH_s<seed>  : M,  reward hybrid   (real clock, expected lateness)
# M = D-prime with the true-sigma^2 training gate (checkpoints Dp_w1_15k_*). Its gate keeps
# learning during the fine-tune, as in its original recipe; evaluation forces lambda = 0.
# Run from the repo root:
#     powershell -ExecutionPolicy Bypass -File .\scripts\run_x2_m_and_sampled.ps1
# 9 runs x 5k steps, about one night. Interleaved by seed so a stopped set leaves complete
# triples. No $ErrorActionPreference = "Stop" (PowerShell 5.1 aborts on any stderr line).

$sim = @("--omega", "1", "--beta-var-range", "0.15", "0.6", "--steps", "5000", "--lr", "1e-4")

$runs = @()
foreach ($seed in @(2024, 7, 1234)) {
    $runs += @{ seed = $seed; variant = "m0";     mode = "sampled";
                init = "checkpoints\M0_w1_15k_s$($seed)_last.pt"; name = "M0_ftS_s$seed" }
    $runs += @{ seed = $seed; variant = "dprime"; mode = "analytic";
                init = "checkpoints\Dp_w1_15k_s$($seed)_last.pt"; name = "M_ftA_s$seed" }
    $runs += @{ seed = $seed; variant = "dprime"; mode = "hybrid";
                init = "checkpoints\Dp_w1_15k_s$($seed)_last.pt"; name = "M_ftH_s$seed" }
}

New-Item -ItemType Directory -Force runs | Out-Null
foreach ($r in $runs) {
    if (-not (Test-Path $r.init)) {
        Write-Host "=== MISSING $($r.init) -- skipping $($r.name) ==="
        continue
    }
    $log = "runs\train_$($r.name).log"
    Write-Host "=== $(Get-Date -Format 'HH:mm') starting $($r.name) ($($r.variant), $($r.mode)) -> $log ==="
    python -u -m scripts.train_stoch --variant $r.variant --seed $r.seed --init-from $r.init `
        --reward-mode $r.mode --out "checkpoints\$($r.name).pt" @sim 2>&1 |
        Tee-Object -FilePath $log
    Write-Host "=== $(Get-Date -Format 'HH:mm') finished $($r.name) ==="
}
Write-Host "All runs done. Check: Get-ChildItem checkpoints\M*_ft*_last.pt"