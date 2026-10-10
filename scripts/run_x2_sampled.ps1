# X2 phase 2a: third reward arm. Fine-tune the same M0 checkpoints for 5,000 more steps
# with reward_mode = sampled (real clock, REALISED lateness charged). Together with
# M0_ftA (analytic: mean clock, expected lateness) and M0_ftH (hybrid: real clock,
# expected lateness), this separates the two ingredients of the hybrid reward:
#   ftS ~ ftH  -> the gain comes from the real clock; the variance reduction is not needed
#   ftS worse  -> the variance reduction (closed-form charge) is what makes it work
# Run from the repo root:
#     powershell -ExecutionPolicy Bypass -File .\scripts\run_x2_sampled.ps1
#
# Same $common as run_x2.ps1 (= original M0 flags + fine-tune settings), so the three
# arms differ ONLY in --reward-mode. No $ErrorActionPreference = "Stop" (PowerShell 5.1
# would abort on any stderr line).

$common = @("--variant", "m0", "--omega", "1", "--beta-var-range", "0.15", "0.6",
            "--steps", "5000", "--lr", "1e-4")

$runs = @()
foreach ($seed in @(2024, 7, 1234)) {
    $runs += @{ seed = $seed; init = "checkpoints\M0_w1_15k_s$($seed)_last.pt";
                mode = "sampled"; name = "M0_ftS_s$seed" }
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
Write-Host "All runs done. Check: Get-ChildItem checkpoints\M0_ftS_*_last.pt"
