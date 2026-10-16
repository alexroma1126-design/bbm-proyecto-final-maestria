$ErrorActionPreference = "Stop"

$root = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$src = Join-Path $root "src\validation"
$build = Join-Path $root ".build"

New-Item -ItemType Directory -Force -Path $build |
    Out-Null

Push-Location $root

try {
    Write-Host "`n===== PYTHON: VALIDACION EXACTA ====="

    & python -I -B (Join-Path $src "parameter_regions_exact_test.py")

    if ($LASTEXITCODE -ne 0) {
        throw "Fallo el validador exacto."
    }

    $tests = @(
        "redundancy_game_test",
        "parameter_regions_test",
        "finite_gramian_game_test"
    )

    foreach ($test in $tests) {
        Write-Host "`n===== C++: $test ====="

        $cpp = Join-Path $src "$test.cpp"
        $exe = Join-Path $build "$test.exe"

        & g++ -std=c++17 -O2 -Wall -Wextra $cpp -o $exe

        if ($LASTEXITCODE -ne 0) {
            throw "Error compilando $test"
        }

        & $exe

        if ($LASTEXITCODE -ne 0) {
            throw "Fallo la prueba $test"
        }
    }

    Write-Host "`nRESULTADO FINAL: 4/4 PRUEBAS PASS"
}
finally {
    Pop-Location
}
