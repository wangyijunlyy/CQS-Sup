"""Fetch the published R1 data, checkpoints and complete search records."""
from pathlib import Path
import argparse,hashlib,json,re,subprocess,tarfile,urllib.request
ROOT=Path(__file__).resolve().parent
def repository_name(value):
    if value:return value
    try:remote=subprocess.check_output(['git','config','--get','remote.origin.url'],cwd=ROOT,text=True,stderr=subprocess.DEVNULL).strip()
    except subprocess.CalledProcessError:remote=''
    match=re.search(r'github\.com[:/]([^/]+/[^/]+?)(?:\.git)?$',remote)
    if not match:raise SystemExit('Pass --repository OWNER/REPOSITORY when using a source ZIP without Git metadata.')
    return match.group(1)
def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--repository',help='Default: derive the repository from Git origin')
    ap.add_argument('--archive',type=Path,help='Use an already downloaded release archive')
    ap.add_argument('--check',action='store_true',help='Check required file presence without downloading')
    args=ap.parse_args();inventory=json.loads((ROOT/'artifact_inventory.json').read_text())['files']
    if args.check:
        missing=[x['file'] for x in inventory if not (ROOT/x['file']).is_file()]
        if missing:raise SystemExit(f'{len(missing)} artifact files missing. Run python fetch_artifacts.py first.')
        print(f'All {len(inventory)} published artifact files are present.');return
    meta=json.loads((ROOT/'artifacts.json').read_text());cache=ROOT/'.cache';cache.mkdir(exist_ok=True)
    archive=args.archive.resolve() if args.archive else cache/meta['asset']
    if not archive.exists():
        url=f"https://github.com/{repository_name(args.repository)}/releases/download/{meta['tag']}/{meta['asset']}"
        partial=archive.with_suffix(archive.suffix+'.part');request=urllib.request.Request(url,headers={'User-Agent':'CQS-R1-reproduction'})
        print(f"Downloading {meta['bytes']/1024**2:.1f} MiB of R1 artifacts...",flush=True)
        with urllib.request.urlopen(request,timeout=120) as response,partial.open('wb') as output:
            while chunk:=response.read(1024*1024):output.write(chunk)
        partial.replace(archive)
    with archive.open('rb') as stream:digest=hashlib.file_digest(stream,'sha256').hexdigest()
    if digest!=meta['sha256']:raise SystemExit('Archive digest mismatch. Remove the cached file and download again.')
    with tarfile.open(archive,'r:gz') as package:
        members=package.getmembers();expected={x['file'] for x in inventory}
        assert {x.name for x in members if x.isfile()}==expected
        for member in members:
            if not (ROOT/member.name).resolve().is_relative_to(ROOT) or not (member.isfile() or member.isdir()):
                raise SystemExit('Unexpected archive member')
        package.extractall(ROOT,filter='data')
    assert all((ROOT/x['file']).is_file() for x in inventory)
    print(f'Extracted all {len(inventory)} R1 artifact files.',flush=True)
if __name__=='__main__':main()
