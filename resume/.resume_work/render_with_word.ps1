param(
    [Parameter(Mandatory = $true)][string]$InputDocx,
    [Parameter(Mandatory = $true)][string]$OutputDir
)

$ErrorActionPreference = 'Stop'
$resolvedInput = (Resolve-Path -LiteralPath $InputDocx).Path
$resolvedOutput = [System.IO.Path]::GetFullPath($OutputDir)
[System.IO.Directory]::CreateDirectory($resolvedOutput) | Out-Null
$pdfPath = Join-Path $resolvedOutput (([System.IO.Path]::GetFileNameWithoutExtension($resolvedInput)) + '.pdf')

$word = $null
$doc = $null
try {
    $word = New-Object -ComObject Word.Application
    $word.Visible = $false
    $word.DisplayAlerts = 0
    $doc = $word.Documents.Open($resolvedInput, $false, $true)
    $doc.ExportAsFixedFormat($pdfPath, 17)
    Write-Output $pdfPath
}
finally {
    if ($null -ne $doc) {
        $doc.Close([ref]0)
        [void][System.Runtime.InteropServices.Marshal]::ReleaseComObject($doc)
    }
    if ($null -ne $word) {
        $word.Quit()
        [void][System.Runtime.InteropServices.Marshal]::ReleaseComObject($word)
    }
    [GC]::Collect()
    [GC]::WaitForPendingFinalizers()
}
