# M0 trained with the HYBRID reward FROM SCRATCH, 15k steps, 3 seeds.
# Question: does the hybrid gain depend on fine-tuning an analytic-trained model?
# Compared (paired, honest) with M0_w1_15k_s<seed>_last.pt: same variant, same seeds, same
# 15k budget, same schedule (lr 3e-4 cosine to 1e-6, the train_stoch defaults); the ONLY
# difference is the reward (analytic there, hybrid here).
#
# $common = the flags of the original 15k runs (scripts\run_seeds.ps1:
# --omega 1 --beta-var-range 0.15 0.6 --steps 15000 --save-every 2500), so the two arms
# differ only in --reward-mode. Smoke-tested on CPU (20 steps, m0 + hybrid, no init).
# About 2.5-3 h per seed.
#
# No $ErrorActionPreference = "Stop" (PowerShell 5.1 would abort on any stderr line).
# Run from the repo root:
#     powershell -ExecutionPolicy Bypass -File .\scripts\run_m0_hybrid_scratch.ps1 2>&1 | Tee-Object -FilePath runs\m0_hybrid_scratch.console.log

$common = @("--variant", "m0", "--reward-mode", "hybrid", "--omega", "1",
            "--beta-var-range", "0.15", "0.6", "--steps", "15000", "--save-every", "2500")

New-Item -ItemType Directory -Force runs | Out-Null
foreach ($seed in @(2024, 7, 1234)) {
    $name = "M0H_w1_15k_s$seed"
    $log = "runs\train_$name.log"
    Write-Host "=== $(Get-Date -Format 'HH:mm') starting $name (m0, hybrid, from scratch) -> $log ==="
    python -u -m scripts.train_stoch --seed $seed --out "checkpoints\$name.pt" @common 2>&1 |
        Tee-Object -FilePath $log
    Write-Host "=== $(Get-Date -Format 'HH:mm') finished $name ==="
}
Write-Host "All runs done. Check: Get-ChildItem checkpoints\M0H_w1_15k_*_last.pt"