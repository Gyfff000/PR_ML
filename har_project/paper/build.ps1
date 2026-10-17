param([string]$TeXBin = "C:\texlive\2026\bin\windows")
$ErrorActionPreference = "Stop"
function Resolve-TeXTool([string]$name) {
    $candidate = Join-Path $TeXBin ($name + ".exe")
    if (Test-Path -LiteralPath $candidate) { return $candidate }
    $command = Get-Command $name -ErrorAction SilentlyContinue
    if ($null -eq $command) { throw "Cannot find $name. Install TeX Live or pass -TeXBin." }
    return $command.Source
}
$xe = Resolve-TeXTool "xelatex"
$bi = Resolve-TeXTool "biber"
$previousTeXInputs = $env:TEXINPUTS
$env:TEXINPUTS = (($PSScriptRoot -replace "\\", "/") + "/vendor/tex-packages//;" + $previousTeXInputs + ";")
Push-Location -LiteralPath $PSScriptRoot
try {
    $stages = @(
        @{Name="XeLaTeX 1/3"; Tool=$xe; Args=@("-interaction=nonstopmode", "-halt-on-error", "-file-line-error", "-no-shell-escape", "main.tex")},
        @{Name="Biber"; Tool=$bi; Args=@("main")},
        @{Name="XeLaTeX 2/3"; Tool=$xe; Args=@("-interaction=nonstopmode", "-halt-on-error", "-file-line-error", "-no-shell-escape", "main.tex")},
        @{Name="XeLaTeX 3/3"; Tool=$xe; Args=@("-interaction=nonstopmode", "-halt-on-error", "-file-line-error", "-no-shell-escape", "main.tex")}
    )
    for ($i = 0; $i -lt $stages.Count; $i++) {
        $stage = $stages[$i]
        Write-Host ("Building: " + $stage.Name)
        $toolPath = $stage.Tool
        $toolArgs = $stage.Args
        $stageLog = Join-Path $PSScriptRoot (".build-" + $i + ".txt")
        & $toolPath @toolArgs *> $stageLog
        if ($LASTEXITCODE -ne 0) {
            Get-Content -LiteralPath $stageLog -Tail 45
            throw ("Compilation failed at " + $stage.Name)
        }
    }
    Write-Host ("PDF ready: " + (Join-Path $PSScriptRoot "main.pdf"))
} finally { Pop-Location; $env:TEXINPUTS = $previousTeXInputs }
