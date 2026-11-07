"""Generate fresh measured data, refit all candidates and repeat chain searches."""
from pathlib import Path
import argparse
import json
import shutil
import sys

ROOT = Path(__file__).resolve().parent
EXPERIMENT = ROOT / 'paper_iclr2027/rebuttal_20261107/reviewer_weak_reject_2/fixed_data_chain_20261108'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seed', type=int, action='append', help='Repeat to run several of 3101–3110; default: all ten')
    parser.add_argument('--output', type=Path, default=ROOT / 'outputs/fixed_chain')
    parser.add_argument('--score', action='store_true', help='Audit and score after all ten datasets have completed')
    args = parser.parse_args()
    protocol = json.loads((EXPERIMENT / 'confirmation_protocol.json').read_text())
    seeds = args.seed or protocol['seeds']
    if not set(seeds).issubset(protocol['seeds']):
        parser.error('Seeds must be selected from 3101 through 3110')
    output = args.output.resolve()
    if output == EXPERIMENT:
        parser.error('Use a separate output folder to preserve the released evidence')
    output.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(EXPERIMENT / 'confirmation_protocol.json', output / 'confirmation_protocol.json')
    sys.path.insert(0, str(EXPERIMENT))
    import confirm
    confirm.HERE = output
    for seed in seeds:
        confirm.run(seed)
    if args.score:
        missing = [seed for seed in protocol['seeds'] if not (output / f'confirmation/seed{seed}/all_saved.json').exists()]
        if missing:
            parser.error(f'Complete all ten datasets before scoring; missing: {missing}')
        import analyze_confirmation
        analyze_confirmation.HERE = output
        analyze_confirmation.main()


if __name__ == '__main__':
    main()
