$allPyFiles = Get-ChildItem -Path ".\modality" -Filter "*.py" -Recurse
$pyFilesToProcess = $allPyFiles | Where-Object { $_.FullName -notlike "*models*" }

$resultsFile = "pylint_report.txt"
"Pylint Results $(Get-Date)" | Out-File -FilePath $resultsFile

$totalFiles = $pyFilesToProcess.Count
$processedCount = 0

foreach ($file in $pyFilesToProcess) {
    $processedCount++
    $percentComplete = if ($totalFiles -gt 0) { [int](($processedCount / $totalFiles) * 100) } else { 0 }
    Write-Progress -Activity "Running Pylint" -Status "Processing $($file.Name) ($processedCount of $totalFiles)" -PercentComplete $percentComplete

    $pylintOutput = pylint $file.FullName | Out-String

    if ($pylintOutput -match "Your code has been rated at (\d+\.\d+|\d+)/10") {
        $score = $matches[1]
        if ($score -eq "10.00" -or $score -eq "10") {
            Write-Host "Skip (Score 10/10): $($file.FullName)" -ForegroundColor Green
            continue
        }
    }
    
    "Exec: $($file.FullName)" | Out-File -Append -FilePath $resultsFile
    "--------------------------" | Out-File -Append -FilePath $resultsFile
    $pylintOutput | Out-File -Append -FilePath $resultsFile
    "--------------------------" | Out-File -Append -FilePath $resultsFile
}

Write-Progress -Activity "Running Pylint" -Completed
Write-Host "Report written to $resultsFile" -ForegroundColor Green