# Gate location: WHERE does the constant gate (DpConst) act? From scratch, same recipe as
# the original DpConst runs (variant dprime, gate signal const, omega 1, beta_var drawn
# in [0.15, 0.6], 15k steps), with ONE change: the gate is kept in one place only.
#   DpConstEnc : gate in the ENCODER only (self-attention bonus)
#   DpConstDec : gate in the DECODER only (pre-tanh shift = temperature)
# Run from the repo root:
#     powershell -ExecutionPolicy Bypass -File .\scripts\run_gate_location.ps1
#
# 6 runs x 15k steps, about two nights. Interleaved by seed so a stopped set still leaves
# complete Enc/Dec pairs. One log per run in runs\. No $ErrorActionPreference = "Stop"
# (PowerShell 5.1 would abort on any stderr line).

$common = @("--variant", "dprime", "--gate-signal", "const",
            "--omega", "1", "--beta-var-range", "0.15", "0.6",
            "--steps", "15000", "--save-every", "2500")

$runs = @()
foreach ($seed in @(2024, 7, 1234)) {
    $runs += @{ seed = $seed; loc = "encoder"; name = "DpConstEnc_w1_15k_s$seed" }
    $runs += @{ seed = $seed; loc = "decoder"; name = "DpConstDec_w1_15k_s$seed" }
}

New-Item -ItemType Directory -Force runs | Out-Null
foreach ($r in $runs) {
    $log = "runs\train_$($r.name).log"
    Write-Host "=== $(Get-Date -Format 'HH:mm') starting $($r.name) (gate in $($r.loc)) -> $log ==="
    python -u -m scripts.train_stoch --seed $r.seed --gate-location $r.loc `
        --out "checkpoints\$($r.name).pt" @common 2>&1 | Tee-Object -FilePath $log
    Write-Host "=== $(Get-Date -Format 'HH:mm') finished $($r.name) ==="
}
Write-Host "All runs done. Check: Get-ChildItem checkpoints\DpConst*_w1_15k_*_last.pt"
