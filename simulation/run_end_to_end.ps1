param(
    [switch]$UseWSL = $true,
    # Empty = use the default WSL distro (see `wsl -l -v`, the one marked *).
    [string]$Distro = ""
)

$ErrorActionPreference = "Stop"

function Invoke-LinuxSetup {
    param(
        [string]$RepoLinuxPath
    )

    # Do not embed multiline bash here: CRLF from Windows and PowerShell quoting break
    # `set -o pipefail`, variable expansion, and `wsl bash -lc`. Run the repo shell script instead.
    $escapedPath = $RepoLinuxPath -replace "'", "'\''"
    # PowerShell: escape `$ so sed's end-anchor `$ is not expanded
    $cmd = "cd '$escapedPath' && sed -i 's/\r`$//' simulation/run_end_to_end.sh simulation/verify_sim_topics.sh 2>/dev/null || true && bash simulation/run_end_to_end.sh"

    if ($UseWSL) {
        if ([string]::IsNullOrWhiteSpace($Distro)) {
            wsl bash -lc "$cmd"
        } else {
            wsl -d $Distro bash -lc "$cmd"
        }
    } else {
        bash -lc "$cmd"
    }
}

Write-Host "=== RobotX VRX end-to-end setup ==="

$repoWindowsPath = (Get-Location).Path

$ros2Native = Get-Command ros2 -ErrorAction SilentlyContinue
$gzNative = Get-Command gz -ErrorAction SilentlyContinue

if ($ros2Native -and $gzNative -and -not $UseWSL) {
    Write-Host "Native ros2/gz detected. Running with local bash."
    if (-not (Get-Command bash -ErrorAction SilentlyContinue)) {
        throw "bash is required for native mode. Install Git Bash or use -UseWSL."
    }
    # Naive path conversion for Git Bash environments.
    $repoLinuxPath = $repoWindowsPath -replace "^([A-Za-z]):", "/$([char]::ToLower($matches[1]))" -replace "\\", "/"
    Invoke-LinuxSetup -RepoLinuxPath $repoLinuxPath
    exit 0
}

if ($UseWSL) {
    $wslCmd = Get-Command wsl -ErrorAction SilentlyContinue
    if (-not $wslCmd) {
        throw "WSL is not installed. Run 'wsl --install', reboot, then rerun this script."
    }

    $drive = $repoWindowsPath.Substring(0, 1).ToLower()
    $rest = $repoWindowsPath.Substring(2).Replace("\", "/")
    $repoLinuxPath = "/mnt/$drive$rest"

    if ([string]::IsNullOrWhiteSpace($Distro)) {
        Write-Host "Using default WSL distro (wsl -l -v shows * on the active one)"
    } else {
        Write-Host "Using WSL distro '$Distro'"
    }
    Write-Host "Repo path in WSL: $repoLinuxPath"
    Invoke-LinuxSetup -RepoLinuxPath $repoLinuxPath
    exit 0
}

throw "No usable runtime found. Use -UseWSL and install WSL Ubuntu, or install native ros2/gz + bash."
