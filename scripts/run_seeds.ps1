# Night 1 -- 5 training runs, ~70 min each (~6 h). Run from the repo root:
#     powershell -ExecutionPolicy Bypass -File .\night1.ps1
#
# BEFORE LAUNCHING: $common must reproduce the flags of the command that produced
# B_w1_15k_s2024. Everything not listed uses train_stoch's defaults (eta 0.6,
# tardiness_weight 2.0, reward analytic, batch 32, lr 3e-4 cosine, beta_var 0.3 for
# validation). If your B command had any other flag, add it here. After the night,
# scripts/inspect_ckpt.py confirms the settings match.
#
# Runs are independent: if one crashes, the next still starts. Each has its own log.
# No $ErrorActionPreference = "Stop": under Windows PowerShell 5.1, anything Python writes
# to stderr (including harmless warnings) would then abort the whole night.

$common = @("--omega", "1", "--beta-var-range", "0.15", "0.6", "--steps", "15000")

$runs = @(
    @{ variant = "dprime"; seed = 7;    name = "Dp_w1_15k_s7"    },
    @{ variant = "dprime"; seed = 1234; name = "Dp_w1_15k_s1234" }
)

New-Item -ItemType Directory -Force runs | Out-Null
foreach ($r in $runs) {
    $log = "runs\train_$($r.name).log"
    Write-Host "=== $(Get-Date -Format 'HH:mm') starting $($r.name) -> $log ==="
    python -u -m scripts.train_stoch --variant $r.variant --seed $r.seed `
        --out "checkpoints\$($r.name).pt" @common 2>&1 | Tee-Object -FilePath $log
    Write-Host "=== $(Get-Date -Format 'HH:mm') finished $($r.name) ==="
}
Write-Host "Night 1 done. Check: Get-ChildItem checkpoints\*_w1_15k_*_last.pt"