"""Command-line interface for Conformerity."""

import argparse
import sys

from fileio.paths import DEFAULT_OUTPUT_BASE, resolve_run_output
from numbering.pipeline import USER_ORDERABLE
from pipelines.alignment import align_batch, align_pair
from pipelines.renumbering import renumber_batch, renumber_pair


def _add_fit_arg(parser):
    """Reference atoms used in the alignment loss (formerly --atoms)."""
    parser.add_argument(
        "--fit", "--anchors", nargs="+", type=int, default=None, dest="fit",
        metavar="REF_IDX",
        help="Reference atom indices (0-based) for alignment loss "
             "(ref atoms to fit). Query partners come from the current map. "
             "--anchors is an alias. Legacy --atoms still accepted.",
    )
    parser.add_argument(
        "--atoms", nargs="+", type=int, default=None, dest="atoms",
        metavar="REF_IDX",
        help=argparse.SUPPRESS,  # legacy alias for --fit
    )


def _add_anchor_pair_args(parser):
    """Forced query<->ref pairs injected into the seed map."""
    parser.add_argument(
        "--anchor-ref", nargs="+", type=int, default=None, dest="anchor_ref",
        metavar="REF_IDX",
        help="Reference indices (0-based) of forced anchor pairs; same length "
             "as --anchor-query. Pairs are injected even if absent from MCS.",
    )
    parser.add_argument(
        "--anchor-query", nargs="+", type=int, default=None, dest="anchor_query",
        metavar="QUERY_IDX",
        help="Query indices (0-based) of forced anchor pairs; same length "
             "as --anchor-ref.",
    )


def _add_debug_arg(parser):
    parser.add_argument(
        "--debug",
        action="store_true",
        help=(
            "Verbose diagnostics: query->ref mapping after each numbering stage, "
            "and rigid-alignment attempts (lr, initial/end loss, retries)"
        ),
    )


def _add_match_charge_arg(parser):
    parser.add_argument(
        "--match-charge",
        dest="match_formal_charge",
        action="store_true",
        help=(
            "Require formal-charge / radical agreement in MCS "
            "(off by default; use for nature-like charged centers)"
        ),
    )


def _add_center_arg(parser):
    parser.add_argument(
        "--no-center",
        dest="center",
        action="store_false",
        help="Keep original coordinate origin (default: shift so the reference centroid is at (0, 0, 0))",
    )
    parser.set_defaults(center=True)


def _add_stages_arg(parser):
    parser.add_argument(
        "--stages",
        nargs="+",
        metavar="STAGE",
        choices=list(USER_ORDERABLE),
        default=None,
        help=(
            "Middle numbering order: any combination of "
            f"{', '.join(USER_ORDERABLE)}. "
            "List them in the order to run. All 6 permutations of the three "
            "are valid, as are 1- or 2-stage subsets. "
            "Seed is always first; spatial_corrections onward is always last. "
            "Default: complete_rings extend_mcs coords_proximity_any"
        ),
    )


def _add_opt_args(parser):
    parser.add_argument(
        "--lr", type=float, default=None,
        help="Initial Adam step size for rigid alignment; cosine-annealed to "
             "lr/10 over --max-iters. If max-iters is hit without early stop, "
             "retry with 2x lr (default: mean distance from centroid / 100, "
             "snapped to 1, 0.1, 0.01, ...)",
    )
    parser.add_argument(
        "--max-iters", type=int, default=2000, dest="max_iters",
        help="Max Adam iterations for rigid alignment (default: 2000)",
    )


def _resolve_fit_atoms(args):
    fit = getattr(args, "fit", None)
    atoms = getattr(args, "atoms", None)
    if fit is not None and atoms is not None and list(fit) != list(atoms):
        raise SystemExit("Use only one of --fit/--anchors or legacy --atoms")
    return fit if fit is not None else atoms


def _resolve_anchor_pairs(args):
    refs = getattr(args, "anchor_ref", None)
    queries = getattr(args, "anchor_query", None)
    if refs is None and queries is None:
        return None
    if refs is None or queries is None:
        raise SystemExit("--anchor-ref and --anchor-query must be set together")
    if len(refs) != len(queries):
        raise SystemExit(
            f"--anchor-ref/--anchor-query length mismatch: {len(refs)} vs {len(queries)}"
        )
    return list(zip(refs, queries))


def build_parser():
    parser = argparse.ArgumentParser(
        prog="main.py",
        description="Align and renumber molecular structures (XYZ format).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py align -t test_cases/01_structural_alignment/cyclohexane_chair.xyz -q test_cases/01_structural_alignment/cyclohexane_boat.xyz
  python main.py pair -t test_cases/02_prochirality/cyclohexanol.xyz -q test_cases/02_prochirality/cyclohexane_boat.xyz
  python main.py pair -t test_cases/03_ring_numbering/stage_order_ref.xyz -q test_cases/03_ring_numbering/stage_order_query.xyz --stages coords_proximity_any

By default, results are written to outputs/{data_folder}_{mode}_{date-time}/
        """,
    )
    subparsers = parser.add_subparsers(dest="mode", required=True, metavar="MODE")

    output_help = (
        f"Output directory (default: {DEFAULT_OUTPUT_BASE}/"
        "{{data_folder}}_{{mode}}_{{MM-DD-YYYY-HH-MM}}/)"
    )

    pair = subparsers.add_parser(
        "pair",
        help="Renumber one query structure to match a reference",
    )
    pair.add_argument("-t", "--target", required=True, help="Reference XYZ file")
    pair.add_argument("-q", "--query", required=True, help="Query XYZ file to renumber")
    pair.add_argument("-o", "--output", default=None, help=output_help)
    pair.add_argument("--shuffle", action="store_true", help="Shuffle target before renumbering")
    _add_center_arg(pair)
    _add_fit_arg(pair)
    _add_anchor_pair_args(pair)
    _add_debug_arg(pair)
    _add_match_charge_arg(pair)
    _add_stages_arg(pair)
    _add_opt_args(pair)

    batch = subparsers.add_parser("batch", help="Batch renumber all XYZ files in a folder")
    batch.add_argument("-f", "--folder", required=True, help="Folder with XYZ files")
    batch.add_argument("-r", "--reference", default="", help="Reference XYZ (default: smallest molecule)")
    batch.add_argument("-o", "--output", default=None, help=output_help)
    batch.add_argument(
        "--anchors-file", default=None,
        help="Excel file (Mol, idx columns; idx 1-based) defining one forced "
             "anchor pair per molecule vs the reference",
    )
    _add_center_arg(batch)
    _add_fit_arg(batch)
    _add_debug_arg(batch)
    _add_match_charge_arg(batch)
    _add_stages_arg(batch)
    _add_opt_args(batch)

    align_p = subparsers.add_parser("align", help="Align coordinates only (no renumbering)")
    align_p.add_argument("-t", "--target", required=True, help="Reference XYZ file")
    align_p.add_argument("-q", "--query", required=True, help="Query XYZ file")
    align_p.add_argument("-o", "--output", default=None, help=output_help)
    align_p.add_argument(
        "--loss-type", default="mixed", choices=["msd", "mad", "rmsd", "sig", "mixed"],
    )
    align_p.add_argument("--sigma", type=float, default=30)
    align_p.add_argument(
        "--with-hs", action="store_true",
        help="Include hydrogens in MCS / alignment loss (default: heavy atoms only)",
    )
    _add_opt_args(align_p)
    _add_fit_arg(align_p)
    _add_center_arg(align_p)
    _add_debug_arg(align_p)

    align_batch_p = subparsers.add_parser("align-batch", help="Batch align folder to a reference")
    align_batch_p.add_argument("-f", "--folder", required=True)
    align_batch_p.add_argument("-r", "--reference", default=None)
    align_batch_p.add_argument("-o", "--output", default=None, help=output_help)
    align_batch_p.add_argument("--loss-type", default="mixed", choices=["msd", "mad", "rmsd", "sig", "mixed"])
    align_batch_p.add_argument("--sigma", type=float, default=30)
    _add_opt_args(align_batch_p)
    _add_fit_arg(align_batch_p)
    _add_center_arg(align_batch_p)
    _add_debug_arg(align_batch_p)

    return parser


def main(argv=None, base_output: str = DEFAULT_OUTPUT_BASE):
    parser = build_parser()
    args = parser.parse_args(argv)

    output_dir = resolve_run_output(
        args.mode,
        output=args.output,
        target=getattr(args, "target", None),
        query=getattr(args, "query", None),
        folder=getattr(args, "folder", None),
        base_output=base_output,
    )
    print(f"Output directory: {output_dir}")

    center_coords = getattr(args, "center", True)
    stages = getattr(args, "stages", None)
    fit_atoms = _resolve_fit_atoms(args)

    if args.mode == "pair":
        renumber_pair(
            args.target, args.query,
            output_directory=output_dir,
            anchors=_resolve_anchor_pairs(args),
            shuffle=args.shuffle,
            center_coords=center_coords,
            optimize_atoms=fit_atoms,
            stage_order=stages,
            debug=args.debug,
            lr=args.lr,
            max_iters=args.max_iters,
            match_formal_charge=getattr(args, "match_formal_charge", False),
        )
        print(f"Done. Results in: {output_dir}")

    elif args.mode == "batch":
        renumber_batch(
            args.folder,
            basic_path=args.reference,
            anchors_file=args.anchors_file,
            output_directory=output_dir,
            center_coords=center_coords,
            optimize_atoms=fit_atoms,
            stage_order=stages,
            debug=args.debug,
            lr=args.lr,
            max_iters=args.max_iters,
            match_formal_charge=getattr(args, "match_formal_charge", False),
        )

    elif args.mode == "align":
        align_pair(
            args.target, args.query,
            output_directory=output_dir,
            loss_type=args.loss_type,
            sigma=args.sigma,
            lr=args.lr,
            max_iters=args.max_iters,
            optimize_atoms=fit_atoms,
            center_coords=center_coords,
            debug=args.debug,
            with_hs=args.with_hs,
        )

    elif args.mode == "align-batch":
        align_batch(
            args.folder,
            ref_path=args.reference,
            output_directory=output_dir,
            loss_type=args.loss_type,
            sigma=args.sigma,
            lr=args.lr,
            max_iters=args.max_iters,
            optimize_atoms=fit_atoms,
            center_coords=center_coords,
            debug=args.debug,
        )

    else:
        parser.print_help()
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
