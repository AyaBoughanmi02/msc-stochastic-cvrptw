# Reproduce a set of training runs, one after another. Run from the repo root:
#     powershell -ExecutionPolicy Bypass -File .\scripts\run_seeds.ps1
#
# $common = the settings shared by every run (they must match how the existing
# B / M0 / D' checkpoints were trained). Each run may add its own flags in `extra`.
# Runs are independent: if one crashes, the next still starts. One log per run.
# No $ErrorActionPreference = "Stop": under Windows PowerShell 5.1, anything Python
# writes to stderr (including harmless warnings) would then abort the whole set.
#
# CURRENT SET: Stage 5, wave 3 -- is VGT-R's training benefit variance INFORMATION?
#   VGTRc : VGT-R whose P(late) is built from the instance's MEAN sigma^2 on every road
#           (keeps slack, the day's variance level and its accumulation; removes WHICH
#           roads are riskier). Evaluated like VGT-R: lambda 0, alpha chosen on held-out
#           data. Compared against VGTR_w1_15k_s* (same seeds, same budget).
# Interleaved by seed so a stopped set still leaves complete seeds.

$common = @("--omega", "1", "--beta-var-range", "0.15", "0.6", "--steps", "15000",
            "--save-every", "2500")

$runs = @(
    @{ variant = "vgtr"; seed = 2024; name = "VGTRc_w1_15k_s2024"; extra = @("--risk-signal", "const") },
    @{ variant = "vgtr"; seed = 7;    name = "VGTRc_w1_15k_s7";    extra = @("--risk-signal", "const") },
    @{ variant = "vgtr"; seed = 1234; name = "VGTRc_w1_15k_s1234"; extra = @("--risk-signal", "const") }
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