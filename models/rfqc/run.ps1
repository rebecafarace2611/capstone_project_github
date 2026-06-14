param(
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$RfqcArguments
)

$projectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$rscript = Get-Command Rscript -ErrorAction SilentlyContinue

if (-not $rscript) {
    $candidates = Get-ChildItem -LiteralPath "C:\Program Files\R" `
        -Directory -ErrorAction SilentlyContinue |
        Sort-Object Name -Descending |
        ForEach-Object { Join-Path $_.FullName "bin\Rscript.exe" }
    $rscriptPath = $candidates | Where-Object {
        Test-Path -LiteralPath $_
    } | Select-Object -First 1
} else {
    $rscriptPath = $rscript.Source
}

if (-not $rscriptPath) {
    throw "Rscript was not found. Install R or add its bin directory to PATH."
}

& $rscriptPath (Join-Path $projectRoot "run.R") @RfqcArguments
exit $LASTEXITCODE
