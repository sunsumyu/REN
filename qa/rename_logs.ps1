$logsDir = "d:\REN\qa\logs"
$docsDir = "d:\REN\qa\docs"

function Rename-Logs($dir) {
    Write-Host "Scanning $dir..."
    if (-not (Test-Path $dir)) {
        Write-Host "Directory $dir does not exist."
        return
    }
    
    Get-ChildItem -Path $dir -Filter "purification_run_*.md" | ForEach-Object {
        $file = $_
        if ($file.Name -like "*[*" -or $file.Name -eq "purification_run.md" -or $file.Name -eq "purification_run_recovered_partial.md") {
            return
        }
        
        if ($file.Name -match "purification_run_(\d{8}_\d{6})\.md") {
            $timestamp = $Matches[1]
            $content = Get-Content -Path $file.FullName -Raw -Encoding utf8
            
            # Find all line numbers in format "数据集第 X 行"
            $matches = [regex]::Matches($content, "数据集第\s*(\d+)\s*行")
            $lineNums = @()
            foreach ($m in $matches) {
                $lineNums += [int]$m.Groups[1].Value
            }
            
            if ($lineNums.Count -gt 0) {
                $lineNums = $lineNums | Sort-Object
                $minLine = $lineNums[0]
                $maxLine = $lineNums[-1]
                $newName = "purification_run_[$minLine-$maxLine]_$timestamp.md"
                Write-Host "Renaming $($file.Name) to $newName"
                Rename-Item -Path $file.FullName -NewName $newName -Force
            } else {
                Write-Host "Skipping $($file.Name) (no line numbers)"
            }
        }
    }
}

Rename-Logs $logsDir
Rename-Logs $docsDir
Write-Host "Done!"
