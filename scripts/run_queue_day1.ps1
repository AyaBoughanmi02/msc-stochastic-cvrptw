# Day-1 GPU queue for the CPAIOR reward paper. Run from the repo root in PowerShell:
#     powershell -ExecutionPolicy Bypass -File scripts\run_queue_day1.ps1
#
# $FT_FLAGS reproduces the M0_ftS runs exactly (checked against train_M0_ftS_s*.log and
# the saved M0_ftS_s1234_last.pt: m0, omega 1.0, beta_var range [0.15, 0.6], sampled
# reward, 5000 steps, lr 1e-4 -> 1e-6, init from M0_w1_15k_s<seed>_last.pt). E1 differs
# from M0_ftS ONLY by --days-per-instance 4 and the output name.
#
# Scripts are run as modules (-m) so the repo root is on sys.path and `src` imports.
#
# Order: E1 seed 2024 -> 7 -> 1234 (about 1 h each), then the paired evaluation of
# M_ftA vs M0_ftA (about 50 min). It stops at the first failure.

# "Continue", not "Stop": in Windows PowerShell 5.1, "Stop" turns any line Python writes
# to stderr (a harmless warning) into a fatal error. Failures are caught by $LASTEXITCODE.
$ErrorActionPreference = "Continue"
New-Item -ItemType Directory -Force -Path runs | Out-Null

$FT_FLAGS = @(
    "--variant", "m0",
    "--omega", "1.0",
    "--beta-var-range", "0.15", "0.6",
    "--tardiness-weight", "2.0",
    "--reward-mode", "sampled",
    "--steps", "5000",
    "--lr", "1e-4",
    "--min-lr", "1e-6"
)

foreach ($seed in 2024, 7, 1234) {
    $log = "runs\train_M0_ftS4_s$seed.log"
    Write-Host "=== $(Get-Date -Format HH:mm) E1 seed $seed -> $log ==="
    python -u -m scripts.train_stoch @FT_FLAGS `
        --init-from "checkpoints\M0_w1_15k_s${seed}_last.pt" `
        --seed $seed `
        --days-per-instance 4 `
        --out "checkpoints\M0_ftS4_s$seed.pt" 2>&1 | Tee-Object -FilePath $log
    if ($LASTEXITCODE -ne 0) { throw "E1 seed $seed failed (exit $LASTEXITCODE)" }
}

Write-Host "=== $(Get-Date -Format HH:mm) P2: M_ftA vs M0_ftA (paired) ==="
python -u -m scripts.apriori_study `
    --arm M0_ftA checkpoints\M0_ftA_s2024_last.pt checkpoints\M0_ftA_s7_last.pt checkpoints\M0_ftA_s1234_last.pt `
    --arm M_ftA  checkpoints\M_ftA_s2024_last.pt  checkpoints\M_ftA_s7_last.pt  checkpoints\M_ftA_s1234_last.pt `
    --rank-rule planned_distance --out runs\apriori_m_ftA
if ($LASTEXITCODE -ne 0) { throw "P2 failed (exit $LASTEXITCODE)" }

Write-Host "=== $(Get-Date -Format HH:mm) done. Send me: runs\train_M0_ftS4_s*.log (first 5 and last 5 lines), runs\apriori_m_ftA.log ==="