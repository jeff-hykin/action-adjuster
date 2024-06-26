#!/usr/bin/env sh
"\"",`$(echo --% ' |out-null)" >$null;function :{};function dv{<#${/*'>/dev/null )` 2>/dev/null;dv() { #>
echo "1.27.2"; : --% ' |out-null <#'; }; version="$(dv)"; deno="$HOME/.deno/$version/bin/deno"; if [ -x "$deno" ]; then  exec "$deno" run -A "$0" "$@";  elif [ -f "$deno" ]; then  chmod +x "$deno" && exec "$deno" run -A "$0" "$@";  fi; bin_dir="$HOME/.deno/$version/bin"; exe="$bin_dir/deno"; has () { command -v "$1" >/dev/null; } ;  if ! has unzip; then if ! has apt-get; then  has brew && brew install unzip; else  if [ "$(whoami)" = "root" ]; then  apt-get install unzip -y; elif has sudo; then  echo "Can I install unzip for you? (its required for this command to work) ";read ANSWER;echo;  if [ "$ANSWER" =~ ^[Yy] ]; then  sudo apt-get install unzip -y; fi; elif has doas; then  echo "Can I install unzip for you? (its required for this command to work) ";read ANSWER;echo;  if [ "$ANSWER" =~ ^[Yy] ]; then  doas apt-get install unzip -y; fi; fi;  fi;  fi;  if ! has unzip; then  echo ""; echo "So I couldn't find an 'unzip' command"; echo "And I tried to auto install it, but it seems that failed"; echo "(This script needs unzip and either curl or wget)"; echo "Please install the unzip command manually then re-run this script"; exit 1;  fi;  repo="denoland/deno"; if [ "$OS" = "Windows_NT" ]; then target="x86_64-pc-windows-msvc"; else :;  case $(uname -sm) in "Darwin x86_64") target="x86_64-apple-darwin" ;; "Darwin arm64") target="aarch64-apple-darwin" ;; "Linux aarch64") repo="LukeChannings/deno-arm64" target="linux-arm64" ;; "Linux armhf") echo "deno sadly doesn't support 32-bit ARM. Please check your hardware and possibly install a 64-bit operating system." exit 1 ;; *) target="x86_64-unknown-linux-gnu" ;; esac; fi; deno_uri="https://github.com/$repo/releases/download/v$version/deno-$target.zip"; exe="$bin_dir/deno"; if [ ! -d "$bin_dir" ]; then mkdir -p "$bin_dir"; fi;  if ! curl --fail --location --progress-bar --output "$exe.zip" "$deno_uri"; then if ! wget --output-document="$exe.zip" "$deno_uri"; then echo "Howdy! I looked for the 'curl' and for 'wget' commands but I didn't see either of them. Please install one of them, otherwise I have no way to install the missing deno version needed to run this code"; exit 1; fi; fi; unzip -d "$bin_dir" -o "$exe.zip"; chmod +x "$exe"; rm "$exe.zip"; exec "$deno" run -A "$0" "$@"; #>}; $DenoInstall = "${HOME}/.deno/$(dv)"; $BinDir = "$DenoInstall/bin"; $DenoExe = "$BinDir/deno.exe"; if (-not(Test-Path -Path "$DenoExe" -PathType Leaf)) { $DenoZip = "$BinDir/deno.zip"; $DenoUri = "https://github.com/denoland/deno/releases/download/v$(dv)/deno-x86_64-pc-windows-msvc.zip";  [Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12;  if (!(Test-Path $BinDir)) { New-Item $BinDir -ItemType Directory | Out-Null; };  Function Test-CommandExists { Param ($command); $oldPreference = $ErrorActionPreference; $ErrorActionPreference = "stop"; try {if(Get-Command "$command"){RETURN $true}} Catch {Write-Host "$command does not exist"; RETURN $false}; Finally {$ErrorActionPreference=$oldPreference}; };  if (Test-CommandExists curl) { curl -Lo $DenoZip $DenoUri; } else { curl.exe -Lo $DenoZip $DenoUri; };  if (Test-CommandExists curl) { tar xf $DenoZip -C $BinDir; } else { tar -Lo $DenoZip $DenoUri; };  Remove-Item $DenoZip;  $User = [EnvironmentVariableTarget]::User; $Path = [Environment]::GetEnvironmentVariable('Path', $User); if (!(";$Path;".ToLower() -like "*;$BinDir;*".ToLower())) { [Environment]::SetEnvironmentVariable('Path', "$Path;$BinDir", $User); $Env:Path += ";$BinDir"; } }; & "$DenoExe" run -A "$PSCommandPath" @args; Exit $LastExitCode; <# 
# */0}`;
import { FileSystem, glob } from "https://deno.land/x/quickr@0.6.47/main/file_system.js"
import { returnAsString, run, Stderr, Stdout, Out } from "https://deno.land/x/quickr@0.6.47/main/run.js"
import { recursivelyAllKeysOf, get, set, remove, merge, compareProperty } from "https://deno.land/x/good@1.4.4.0/object.js"
import ProgressBar from "https://deno.land/x/progress@v1.3.8/mod.ts"
import { makeIterable, asyncIteratorToList, concurrentlyTransform } from "https://deno.land/x/good@1.1.1.2/iterable.js"

// go to project root
FileSystem.cwd = await FileSystem.walkUpUntil(".git/")

let depsFolder
let pathInfo = await FileSystem.info(FileSystem.thisFolder+"/__dependencies__")
if (pathInfo.isFolder) {
    depsFolder = pathInfo.path
} else {
    for await (const each of FileSystem.recursivelyIterateItemsIn('.', { searchOrder: 'breadthFirstSearch'})) {
        if (each.path.startsWith(".git")) {
            continue
        }
        if (each.isFolder && each.basename == "__dependencies__") {
            depsFolder = each.path
            break
        }
    }
}

const settingsJsonPath = `${FileSystem.parentPath(depsFolder)}/settings.json`
const dependencies = JSON.parse(await FileSystem.read(settingsJsonPath))?.pure_python_packages ?? {}

const statusOutput = await run`git status ${Out(returnAsString)}`
let changesWereStashed = false
if (statusOutput.match(/Changes to be committed:|Changes not staged for commit:|Untracked files:/)) {
    await run`git add ${settingsJsonPath} ${Out(null)}`
    await run`git commit -m ${"add settings path"} ${Out(null)}`
    await run`git add -A`
    if (await run`git stash`.success) {
        changesWereStashed = true
    }
}
try {
    
    for (const [importName, value] of Object.entries(dependencies)) {
        const { path, git_url: gitUrl } = value
        const repoPath = `${depsFolder}/__sources__/${importName}`
        if (gitUrl) {
            if (!(await FileSystem.info(repoPath)).exists) {
                console.log(`cloning: ${gitUrl}`)
                const {success} = await run`git subrepo clone ${gitUrl} ${repoPath}`
                if (!success) {
                    break
                }
            }
        }
    }
    const globPath = `${depsFolder}/__sources__/*/.gitrepo`
    for await (const eachPath of FileSystem.globIterator(globPath, { searchOrder: 'breadthFirstSearch'})) {
        const folderToPull = FileSystem.parentPath(eachPath)
        console.log(`pulling: ${eachPath}`)
        const { success } = (await run`git subrepo pull --force ${folderToPull}`) 
        // if (!success) {
        //     break
        // }
    }
    console.log(`done`)
} finally {
    if (changesWereStashed) {
        await run`git stash pop`
        await run`git add -A`
    }
}

// (this comment is part of deno-guillotine, dont remove) #>