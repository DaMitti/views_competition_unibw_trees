$WebClient = New-Object System.Net.WebClient
Write-Output "Downloading views prediction competition files..."
$WebClient.DownloadFile("https://www.dropbox.com/sh/yxk5w04p2e1xtqk/AACU2k5EUOuEeMq2kZ3gpZZwa?dl=1", "views.zip")
Write-Output "Unpacking views prediction competition files..."
Expand-Archive views.zip -DestinationPath .\views_data 
Write-Output "Downloading CGAZ Geoboundaries ADM0 dataset..."
$WebClient.DownloadFile("https://github.com/wmgeolab/geoBoundaries/raw/main/releaseData/CGAZ/geoBoundariesCGAZ_ADM0.gpkg", "src/data/geoBoundariesCGAZ_ADM0.gpkg")