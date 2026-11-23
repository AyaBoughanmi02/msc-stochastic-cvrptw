# X2 night 4: two experiments, 6 runs, all +5,000 steps from the 15k `_last` checkpoints,
# lr 1e-4 (cosine to 1e-6), so every run ends at 20,000 steps (matched with M0_ftH / B_ftH).
#
#   B_ftA    : variant b    + 5k steps, ANALYTIC reward (control for B_ftH)
#              Question: did B's fragility disappear because of the hybrid reward, or just
#              because of 5k more steps? Compared with B_ftH (runs\apriori_b_ftA).
#   VGTR_ftH : variant vgtr + 5k steps, HYBRID reward
#              Question: VGT-R's risk bias over-reacted at test because, under the analytic
#              reward, it never saw a real (already late) clock in training. The hybrid
#              reward trains on the real clock, and P(late) is then computed exactly as at
#              test (time_var = 0 in both). Does the bias now help when kept at test?
#              Compared with M0_ftH (runs\apriori_vgtr_ftH).
#
# Flags = the original 15k runs' flags, read from the log headers:
#   B    : runs\train_B_w1_15k_s7.log, _s1234.log, runs\B_w1_s2024.txt
#   VGT-R: runs\train_VGTR_w1_15k_s{2024,7,1234}.log (gate_init 0, risk_init alpha 1,
#          gate_lr 1e-2, risk signal sigma2; final lambda 0.0010-0.0015, alpha 9.24-9.43)
# all with omega 1, beta_var range [0.15, 0.6], tardiness_weight 2 (defaults).
# The VGT-R gate is NOT removed: --gate-fixed is D-prime only, and training already drove
# lambda to ~0.001; it is also forced to 0 at evaluation.
# Smoke-tested on CPU (20 steps, seed 2024): b + hybrid and vgtr + hybrid both ran.
#
# No $ErrorActionPreference = "Stop" (PowerShell 5.1 would abort on any stderr line).
# Run from the repo root:
#     powershell -ExecutionPolicy Bypass -File .\scripts\run_x2_night4.ps1 2>&1 | Tee-Object -FilePath runs\x2_night4.console.log

$common = @("--omega", "1", "--beta-var-range", "0.15", "0.6", "--steps", "5000", "--lr", "1e-4")

$runs = @()
foreach ($seed in @(2024, 7, 1234)) {
    $runs += @{ seed = $seed; variant = "b"; mode = "analytic"; name = "B_ftA_s$seed";
                init = "checkpoints\B_w1_15k_s$($seed)_last.pt" }
}
foreach ($seed in @(2024, 7, 1234)) {
    $runs += @{ seed = $seed; variant = "vgtr"; mode = "hybrid"; name = "VGTR_ftH_s$seed";
                init = "checkpoints\VGTR_w1_15k_s$($seed)_last.pt" }
}

New-Item -ItemType Directory -Force runs | Out-Null
foreach ($r in $runs) {
    if (-not (Test-Path $r.init)) {
        Write-Host "=== MISSING $($r.init) -- skipping $($r.name) ==="
        continue
    }
    $log = "runs\train_$($r.name).log"
    Write-Host "=== $(Get-Date -Format 'HH:mm') starting $($r.name) ($($r.variant), $($r.mode)) -> $log ==="
    python -u -m scripts.train_stoch --seed $r.seed --variant $r.variant --init-from $r.init `
        --reward-mode $r.mode --out "checkpoints\$($r.name).pt" @common 2>&1 |
        Tee-Object -FilePath $log
    Write-Host "=== $(Get-Date -Format 'HH:mm') finished $($r.name) ==="
}
Write-Host "All runs done. Check: Get-ChildItem checkpoints\B_ftA_*_last.pt, checkpoints\VGTR_ftH_*_last.pt"