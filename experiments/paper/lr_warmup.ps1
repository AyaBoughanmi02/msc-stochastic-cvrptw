# E6: does a learning-rate warm-up stop the from-scratch collapse of the hybrid reward?
# Run from the repo root in PowerShell (about 3 h per seed, 15 h in total; two nights):
#     powershell -ExecutionPolicy Bypass -File scripts\run_queue_warmup.ps1
#
# Same flags as scripts\run_m0_hybrid_scratch.ps1 (M0, hybrid, omega 1, beta in [0.15, 0.6],
# 15000 steps, lr 3e-4 cosine to 1e-6, no clipping) plus --warmup-steps 500. Same seeds as
# the earlier hybrid runs, so each seed is compared with itself:
#     without warm-up: 2024 FAILED, 42 FAILED (re-run), 99 FAILED, 7 ok, 1234 ok.
# Every earlier collapse was visible by step 100-300 (validation frozen at 22.39).
#
# READING (fixed before running). A run FAILED if its last validation cost is >= 20.
#   R6  claim "a 500-step warm-up prevented the collapse" only if all 5 runs train,
#       including the three seeds that failed without it. Otherwise report the count
#       (k of 5) next to 3 of 6 without warm-up and 0 of 5 with the analytic reward.
# The runs keep going after a collapse (no early stop), so every log is complete.

$ErrorActionPreference = "Continue"
New-Item -ItemType Directory -Force -Path runs | Out-Null

$common = @("--variant", "m0", "--reward-mode", "hybrid", "--omega", "1",
            "--beta-var-range", "0.15", "0.6", "--steps", "15000", "--save-every", "2500",
            "--warmup-steps", "500")

foreach ($seed in 2024, 42, 99, 7, 1234) {
    $name = "M0Hw_w1_15k_s$seed"
    $log = "runs\train_$name.log"
    Write-Host "=== $(Get-Date -Format HH:mm) E6 $name (hybrid, from scratch, warm-up 500) -> $log ==="
    python -u -m scripts.train_stoch --seed $seed --out "checkpoints\$name.pt" @common 2>&1 |
        Tee-Object -FilePath $log
    if ($LASTEXITCODE -ne 0) { throw "E6 seed $seed crashed (exit $LASTEXITCODE)" }
}
Write-Host "=== $(Get-Date -Format HH:mm) done. Send me runs\train_M0Hw_w1_15k_s*.log (or just tell me when it finished: I can read them) ==="
