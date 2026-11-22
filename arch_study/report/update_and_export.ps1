# update_and_export.ps1 - open the report in Microsoft Word, update all fields
# (table of contents, lists of figures/tables, captions, page numbers),
# save the .docx and export a PDF.
# Usage: powershell.exe -NoProfile -ExecutionPolicy Bypass -File update_and_export.ps1 <in.docx> <out.pdf>
param([string]$InDocx, [string]$OutPdf)
$ErrorActionPreference = 'Stop'
$word = New-Object -ComObject Word.Application
$word.Visible = $false
$word.DisplayAlerts = 0
try {
    $doc = $word.Documents.Open($InDocx)
    $doc.Fields.Update() | Out-Null
    foreach ($t in $doc.TablesOfContents) { $t.Update() | Out-Null }
    foreach ($t in $doc.TablesOfFigures) { $t.Update() | Out-Null }
    $doc.Repaginate()
    foreach ($t in $doc.TablesOfContents) { $t.UpdatePageNumbers() | Out-Null }
    foreach ($t in $doc.TablesOfFigures) { $t.UpdatePageNumbers() | Out-Null }
    $doc.Save()
    $doc.ExportAsFixedFormat($OutPdf, 17)   # 17 = wdExportFormatPDF
    Write-Output ("pages=" + $doc.ComputeStatistics(2))
    $doc.Close()
} finally {
    $word.Quit()
}
