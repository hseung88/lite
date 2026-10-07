from __future__ import annotations

import argparse

from lite.experiments.bo.plotting import (
    plot_bo_appendix_time_regret,
    plot_bo_main_regret,
    plot_bo_resource_summary,
    plot_tera_m_sweep,
)


def main() -> None:
    p = argparse.ArgumentParser(description="Plot BO regret, wall-clock, and memory figures.")
    p.add_argument("--csv", required=True)
    p.add_argument(
        "--benchmark", default=None, help="Benchmark override for resource or appendix plots."
    )
    p.add_argument("--main-regret-out", default=None)
    p.add_argument("--main-benchmarks", default=None)
    p.add_argument("--resource-out", default=None)
    p.add_argument("--resource-benchmark", default="ackley_200")
    p.add_argument("--m-sweep-out", default=None)
    p.add_argument("--m-sweep-method", default="lite")
    p.add_argument("--appendix-time-regret-out", default=None)
    p.add_argument("--title", default=None)
    args = p.parse_args()

    if args.main_regret_out:
        benchmarks = None
        if args.main_benchmarks:
            benchmarks = [x.strip() for x in args.main_benchmarks.split(",") if x.strip()]
        print(
            plot_bo_main_regret(
                args.csv, args.main_regret_out, benchmarks=benchmarks, title=args.title
            )
        )
    if args.resource_out:
        resource_benchmark = args.benchmark or args.resource_benchmark
        print(
            plot_bo_resource_summary(
                args.csv, args.resource_out, benchmark=resource_benchmark, title=args.title
            )
        )
    if args.m_sweep_out:
        print(
            plot_tera_m_sweep(
                args.csv,
                args.m_sweep_out,
                benchmark=args.benchmark,
                method=args.m_sweep_method,
                title=args.title,
            )
        )
    if args.appendix_time_regret_out:
        resource_benchmark = args.benchmark or args.resource_benchmark
        print(
            plot_bo_appendix_time_regret(
                args.csv,
                args.appendix_time_regret_out,
                benchmark=resource_benchmark,
                title=args.title,
            )
        )
    if not (
        args.main_regret_out
        or args.resource_out
        or args.m_sweep_out
        or args.appendix_time_regret_out
    ):
        raise SystemExit(
            "Provide --main-regret-out, --resource-out, --m-sweep-out, or --appendix-time-regret-out."
        )


if __name__ == "__main__":
    main()
