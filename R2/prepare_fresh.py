"""Copy code and original inputs to a new directory for full CUDA retraining."""
from pathlib import Path
import argparse,shutil

ROOT=Path(__file__).resolve().parent
def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('destination',type=Path)
    args=ap.parse_args();dest=args.destination.resolve()
    if dest.exists() or dest.is_relative_to(ROOT):
        raise SystemExit('Choose a new directory outside this reported-run package.')
    dest.mkdir(parents=True)
    for folder in ('world_model_surrogate','paper_iclr2027'):
        for source in (ROOT/folder).rglob('*'):
            if not source.is_file() or '__pycache__' in source.parts:continue
            relative=source.relative_to(ROOT)
            if folder=='paper_iclr2027' and source.suffix!='.py' and source.name!='protocol.json':continue
            target=dest/relative;target.parent.mkdir(parents=True,exist_ok=True)
            shutil.copyfile(source,target)
    print('Fresh code and original inputs copied to',dest)

if __name__=='__main__':main()
