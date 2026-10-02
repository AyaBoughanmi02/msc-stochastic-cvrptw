# Reproduce a set of training runs, one after another. Run from the repo root:
#     powershell -ExecutionPolicy Bypass -File .\scripts\run_seeds.ps1
#
# $common = the settings shared by every run (they must match how the existing
# B / M0 / D' checkpoints were trained). Each run may add its own flags in `extra`.
# Runs are independent: if one crashes, the next still starts. One log per run.
# No $ErrorActionPreference = "Stop": under Windows PowerShell 5.1, anything Python
# writes to stderr (including harmless warnings) would then abort the whole set.
#
# CURRENT SET: Stage 5, wave 1 -- what about sigma^2 teaches the policy?
#   DpShuf  : gate penalises the instance's sigma^2 SHUFFLED among its roads
#   DpConst : gate penalises the instance's MEAN sigma^2 on every road
#   DpSnap  : plain D' (true sigma^2), retrained with snapshots -> learning curve of M
# --save-every 2500 keeps <name>_step2500.pt ... so every run also gives a learning curve.
# Interleaved by seed: if stopped early, each control still has the same seeds done.

$common = @("--omega", "1", "--beta-var-range", "0.15", "0.6", "--steps", "15000",
            "--save-every", "2500")

$runs = @(
    @{ variant = "dprime"; seed = 2024; name = "DpShuf_w1_15k_s2024";  extra = @("--gate-signal", "shuffle") },
    @{ variant = "dprime"; seed = 2024; name = "DpConst_w1_15k_s2024"; extra = @("--gate-signal", "const") },
    @{ variant = "dprime"; seed = 7;    name = "DpShuf_w1_15k_s7";     extra = @("--gate-signal", "shuffle") },
    @{ variant = "dprime"; seed = 7;    name = "DpConst_w1_15k_s7";    extra = @("--gate-signal", "const") },
    @{ variant = "dprime"; seed = 1234; name = "DpShuf_w1_15k_s1234";  extra = @("--gate-signal", "shuffle") },
    @{ variant = "dprime"; seed = 1234; name = "DpConst_w1_15k_s1234"; extra = @("--gate-signal", "const") },
    @{ variant = "dprime"; seed = 2024; name = "DpSnap_w1_15k_s2024";  extra = @() }
)

New-Item -ItemType Directory -Force runs | Out-Null
foreach ($r in $runs) {
    $log = "runs\train_$($r.name).log"
    $extra = $r.extra
    Write-Host "=== $(Get-Date -Format 'HH:mm') starting $($r.name) -> $log ==="
    python -u -m scripts.train_stoch --variant $r.variant --seed $r.seed `
        --out "checkpoints\$($r.name).pt" @common @extra 2>&1 | Tee-Object -FilePath $log
    Write-Host "=== $(Get-Date -Format 'HH:mm') finished $($r.name) ==="
}
Write-Host "All runs done. Check: Get-ChildItem checkpoints\*_w1_15k_*_last.pt"