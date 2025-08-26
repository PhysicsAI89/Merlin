if (Test-Path .env) {
  Get-Content .env | ForEach-Object {
    if ($_ -match '^\s*#') { return }
    if ($_ -match '^\s*$') { return }
    $kv = $_ -split '=',2
    if ($kv.Length -eq 2) { $env:$($kv[0]) = $kv[1] }
  }
}
python -m streamlit run app/ui_app.py --server.headless=true
