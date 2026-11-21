# Queue for the three strengthening analyses of the CPAIOR reward paper. Run from the repo
# root in PowerShell:
#     powershell -ExecutionPolicy Bypass -File scripts\run_queue_strengthen.ps1
#
# 1. E4  validation curves of the existing fine-tuning runs     (no GPU, seconds)
# 2. E3  ranking bias of the rewards on M0_ftH routes            (about 40 min)
# 3. E5  short-budget fine-tunes, sampled vs hybrid, 1000 steps   (about 80 min)
#        + their paired test evaluation                          (about 50 min)
# It stops at the first failure.
#
# E5 READING (fixed before running). Same flags as M0_ftS / M0_ftH except --steps 1000
# (cosine 1e-4 -> 1e-6 over those 1000 steps), from the same initial checkpoints.
#   R5  claim "at a short budget the hybrid reward trains a better policy than the
#       sampled one" only if M0_ftS1k - M0_ftH1k (section 4 of the evaluation log) has a
#       95% CI above 0 at BOTH beta 0.3 and beta 0.6 (the training range) and all three
#       seeds are positive at both. Otherwise report "no detectable difference at 1000
#       steps either", next to the 5000-step result.

$ErrorActionPreference = "Continue"
New-Item -ItemType Directory -Force -Path runs | Out-Null

Write-Host "=== $(Get-Date -Format HH:mm) E4: validation curves ==="
python -u -m scripts.e4_val_curves 2>&1 | Tee-Object -FilePath runs\e4_val_curves.log
if ($LASTEXITCODE -ne 0) { throw "E4 failed (exit $LASTEXITCODE)" }

Write-Host "=== $(Get-Date -Format HH:mm) E3: ranking bias ==="
python -u -m scripts.e3_ranking_bias `
    --ckpt checkpoints\M0_ftH_s2024_last.pt checkpoints\M0_ftH_s7_last.pt checkpoints\M0_ftH_s1234_last.pt `
    --out runs\e3_ranking_bias
if ($LASTEXITCODE -ne 0) { throw "E3 failed (exit $LASTEXITCODE)" }

$FT_FLAGS = @(
    "--variant", "m0",
    "--omega", "1.0",
    "--beta-var-range", "0.15", "0.6",
    "--tardiness-weight", "2.0",
    "--steps", "1000",
    "--lr", "1e-4",
    "--min-lr", "1e-6"
)

foreach ($seed in 2024, 7, 1234) {
    foreach ($arm in @(@("H", "hybrid"), @("S", "sampled"))) {
        $tag = $arm[0]; $mode = $arm[1]
        $log = "runs\train_M0_ft${tag}1k_s$seed.log"
        Write-Host "=== $(Get-Date -Format HH:mm) E5 $mode seed $seed -> $log ==="
        python -u -m scripts.train_stoch @FT_FLAGS `
            --reward-mode $mode `
            --init-from "checkpoints\M0_w1_15k_s${seed}_last.pt" `
            --seed $seed `
            --out "checkpoints\M0_ft${tag}1k_s$seed.pt" 2>&1 | Tee-Object -FilePath $log
        if ($LASTEXITCODE -ne 0) { throw "E5 $mode seed $seed failed (exit $LASTEXITCODE)" }
    }
}

Write-Host "=== $(Get-Date -Format HH:mm) E5 evaluation: M0_ftS1k vs M0_ftH1k (paired) ==="
python -u -m scripts.apriori_study `
    --arm M0_ftH1k checkpoints\M0_ftH1k_s2024_last.pt checkpoints\M0_ftH1k_s7_last.pt checkpoints\M0_ftH1k_s1234_last.pt `
    --arm M0_ftS1k checkpoints\M0_ftS1k_s2024_last.pt checkpoints\M0_ftS1k_s7_last.pt checkpoints\M0_ftS1k_s1234_last.pt `
    --rank-rule planned_distance --out runs\apriori_short
if ($LASTEXITCODE -ne 0) { throw "E5 evaluation failed (exit $LASTEXITCODE)" }

Write-Host "=== $(Get-Date -Format HH:mm) done. Send me: runs\e4_val_curves.log, runs\e3_ranking_bias.log, runs\apriori_short.log ==="
