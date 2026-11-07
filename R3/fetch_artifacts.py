"""Download the actual R3 data and checkpoints from this repository's release."""
from pathlib import Path
import argparse,hashlib,json,re,subprocess,tarfile,urllib.request
ROOT=Path(__file__).resolve().parent

def repository_name(value=None):
    if value:return value
    remote=subprocess.check_output(['git','config','--get','remote.origin.url'],cwd=ROOT,text=True).strip()
    match=re.search(r'github\.com[:/]([^/]+/[^/]+?)(?:\.git)?$',remote)
    if not match:raise SystemExit('Pass --repository OWNER/REPOSITORY for a checkout without a GitHub origin.')
    return match.group(1)

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--repository',help='Defaults to the GitHub origin of this checkout')
    ap.add_argument('--check',action='store_true',help='Check that all artifact files are present, without downloading')
    args=ap.parse_args()
    inventory=json.loads((ROOT/'artifact_inventory.json').read_text())['files']
    missing=[r['file'] for r in inventory if not (ROOT/r['file']).is_file()]
    if args.check:
        if missing:raise SystemExit(f'{len(missing)} artifact files missing; run python fetch_artifacts.py first.')
        print(f'All {len(inventory)} data/checkpoint/record files are present.');return
    meta=json.loads((ROOT/'artifacts.json').read_text())
    cache=ROOT/'.cache';cache.mkdir(exist_ok=True)
    archive=cache/meta['asset'];url=f"https://github.com/{repository_name(args.repository)}/releases/download/{meta['tag']}/{meta['asset']}"
    if not archive.exists():
        partial=archive.with_suffix(archive.suffix+'.part')
        request=urllib.request.Request(url,headers={'User-Agent':'CQS-R3-reproduction'})
        print(f"Downloading {meta['bytes']/1024**2:.1f} MiB of data and model checkpoints...",flush=True)
        with urllib.request.urlopen(request,timeout=120) as response,partial.open('wb') as stream:
            while chunk:=response.read(1024*1024):stream.write(chunk)
        partial.replace(archive)
    with archive.open('rb') as stream:digest=hashlib.file_digest(stream,'sha256').hexdigest()
    if digest!=meta['sha256']:raise SystemExit('Archive checksum mismatch; remove the cached archive and download again.')
    with tarfile.open(archive,'r:gz') as package:
        members=package.getmembers()
        for member in members:
            target=(ROOT/member.name).resolve()
            if not target.is_relative_to(ROOT) or not (member.isfile() or member.isdir()):
                raise SystemExit('Unexpected archive member')
        package.extractall(ROOT,members=members,filter='data')
    missing=[r['file'] for r in inventory if not (ROOT/r['file']).is_file()]
    if missing:raise SystemExit(f'Incomplete extraction: {len(missing)} files missing')
    print(f'Ready: {len(inventory)} data/checkpoint/record files extracted under R3.')

if __name__=='__main__':main()
