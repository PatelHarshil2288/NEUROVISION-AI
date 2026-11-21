param(
    [Parameter(Mandatory = $true)]
    [string]$ZipPath,

    [Parameter(Mandatory = $true)]
    [string]$ChecksumJsonPath
)

$errorActionPreference = "Stop"

$resolvedZip = (Resolve-Path $ZipPath).Path
$resolvedChecksum = (Resolve-Path $ChecksumJsonPath).Path
$metadata = Get-Content $resolvedChecksum -Raw | ConvertFrom-Json
$actual = (Get-FileHash $resolvedZip -Algorithm SHA256).Hash.ToLower()

if ($actual -ne $metadata.sha256.ToLower()) {
    throw "SHA256 mismatch: expected $($metadata.sha256), got $actual"
}

Write-Output "Verified $resolvedZip"
