# Benchmark: where does a remote Datadeps region's wall-clock actually go?
#
# Companion to `spill_matmul.jl`, which measures peak memory and total time.
# This one measures *idle* -- the time the compute worker spends not running a
# task -- and attributes it, because "idle %" on its own turned out to be easy
# to misread in three separate ways (see `idle_analysis.jl` for the accounting):
#
#   * `:move` / `:datadeps_copy` on the worker are not idle, they are data
#     movement, and scoring them as idle conflates "starved" with "transferring";
#   * summing overlapping `:compute` intervals double-counts a multi-threaded
#     worker, so busy could exceed the measurement window;
#   * taking the denominator from the worker's own first/last compute event
#     makes the result move with task placement and stragglers.
#
# It also reports the *distribution* of idle gaps, not just a total: a handful of
# long stalls and ten thousand sub-millisecond ones have completely different
# causes, and a single percentage cannot tell them apart.
#
# Two effects dominate any naive version of this measurement, so both are
# handled here:
#
#   * Compilation. With a single warmup region the first measured repetition ran
#     2.7-3.7x slower than the second, and its idle was dominated by a few
#     0.1-1 s stalls that never reappeared. Deep warmup (per configuration --
#     the spill path compiles separately) collapsed the spill-configuration
#     spread from 23.5 to 1.4 percentage points.
#   * Tile size. Idle is largely a fixed per-task cost (one gap per task, ~0.5-1.5
#     ms), so it dominates small tiles and vanishes on large ones: the same code
#     measured 43% idle at bs=256 and 4.7% idle (95.0% busy) at bs=1024. Quote
#     an idle figure only together with its block size.
#
# N.B. Arithmetic intensity of a tile gemm is `bs/12` flops per byte, so *larger*
# blocks are more compute-bound, not less. To push a run toward being I/O-bound,
# grow `N` (more tiles -> larger footprint against the budget) and keep `bs`
# modest; raising `bs` moves the balance the other way.
#
# Usage:
#   julia --project=. -t <nthreads> benchmark/idle_matmul.jl [N] [bs] [budget_MiB] [reps] [warmup] [configs] [check]
#
# Defaults: N=4096 bs=1024 budget=64MiB reps=8 warmup=4 configs=off,on check=auto
#   configs : comma-separated subset of `off,on` (`on` = memory-aware + spill)
#   check   : `full` (dense reference; small N only), `tile` (verify one output
#             tile; cost is NT tile-gemms), or `auto`

using Distributed, Printf, Statistics, LinearAlgebra

const N       = length(ARGS) >= 1 ? parse(Int, ARGS[1]) : 4096
const BS      = length(ARGS) >= 2 ? parse(Int, ARGS[2]) : 1024
const BUDGET  = UInt64((length(ARGS) >= 3 ? parse(Int, ARGS[3]) : 64) * 2^20)
const REPS    = length(ARGS) >= 4 ? parse(Int, ARGS[4]) : 8
const WARMUP  = length(ARGS) >= 5 ? parse(Int, ARGS[5]) : 4
const CONFIGS = length(ARGS) >= 6 ? split(ARGS[6], ',') : ["off", "on"]
@assert N % BS == 0 "N ($N) must be divisible by bs ($BS)"

const NT = N ÷ BS
# A dense reference needs an N^3 host matmul and two N^2 host matrices, which is
# hopeless past a few thousand. Verifying one output tile costs NT tile-gemms
# instead of NT^3 and catches the failure modes that matter here (a spilled tile
# reloading as the wrong buffer shows up immediately).
const CHECK = let c = length(ARGS) >= 7 ? ARGS[7] : "auto"
    c == "auto" ? (N <= 4096 ? "full" : "tile") : c
end

if nworkers() < 2
    addprocs(1; exeflags = "--project=$(Base.active_project())")
end
@everywhere using Dagger, LinearAlgebra
@everywhere gemm_acc!(C, A, B) = (mul!(C, A, B, 1.0, 1.0); C)

include(joinpath(@__DIR__, "idle_analysis.jl"))
using .IdleAnalysis

const w2  = Dagger.scope(worker = 2)
const sp2 = Dagger.CPURAMMemorySpace(2)

# Independent per-tile arrays, NOT views into one parent: a shared parent is
# materialized on the worker as a single buffer, so per-tile spill and free have
# nothing to reclaim. Same reasoning as `spill_matmul.jl`.
maketiles(f) = [f(i, j) for i in 1:NT, j in 1:NT]

function matmul_region!(Ct, At, Bt)
    Dagger.spawn_datadeps() do
        for i in 1:NT, j in 1:NT, k in 1:NT
            Dagger.@spawn compute_scope=w2 gemm_acc!(InOut(Ct[i, j]), In(At[i, k]), In(Bt[k, j]))
        end
    end
    return Ct
end

humanbytes(n) = n < 1024^2 ? @sprintf("%.1f KiB", n/1024) :
                n < 1024^3 ? @sprintf("%.1f MiB", n/1024^2) : @sprintf("%.2f GiB", n/1024^3)

"""
Run one region under logging, returning the logs, the exact wall-clock window
the region occupied (relative to the log time origin), and worker-2 GC time.

The window is taken with `time_ns()` on worker 1; that is the same system-wide
monotonic clock the workers stamp log events with, so the values are comparable.
Dagger has no log category for GC, so a collection would otherwise land in
"unexplained" idle -- sampling the counter on the worker keeps it separable.
"""
function run_instrumented(Ct, At, Bt)
    foreach(t -> fill!(t, 0.0), Ct)
    GC.gc()
    Dagger.enable_logging!(; all_task_deps = true)
    gc0 = remotecall_fetch(() -> Base.gc_time_ns(), 2)
    t0 = time_ns()
    elapsed = @elapsed matmul_region!(Ct, At, Bt)
    t1 = time_ns()
    gc1 = remotecall_fetch(() -> Base.gc_time_ns(), 2)
    logs = Dagger.fetch_logs!()
    Dagger.disable_logging!()
    origin = IdleAnalysis.log_time_origin(logs)
    window = ((Float64(t0) - Float64(origin)) / 1e9, (Float64(t1) - Float64(origin)) / 1e9)
    return (; logs, origin, window, elapsed, gc = (Float64(gc1) - Float64(gc0)) / 1e9)
end

enable_spill!() = Dagger.enable_memory_aware_scheduling!(;
    limits = Dict{Dagger.MemorySpace,UInt64}(sp2 => BUDGET), reassign = false, spill_to_disk = true)

function main()
    tile_bytes = BS * BS * 8
    @printf("N=%d  bs=%d  tiles=%dx%d (%s each)  budget=%s  reps=%d  warmup=%d  check=%s\n",
            N, BS, NT, NT, humanbytes(tile_bytes), humanbytes(BUDGET), REPS, WARMUP, CHECK)
    @printf("footprint: %s per matrix, %s for A+B+C   tasks/region: %d\n",
            humanbytes(NT*NT*tile_bytes), humanbytes(3*NT*NT*tile_bytes), NT^3)
    @printf("arithmetic intensity: %.0f flop/byte (bs/12) -- higher is more compute-bound\n", BS/12)
    println("procs: ", procs(), "   configs: ", join(CONFIGS, ","))

    At = maketiles((i, j) -> rand(BS, BS))
    Bt = maketiles((i, j) -> rand(BS, BS))
    Ct = maketiles((i, j) -> zeros(BS, BS))

    # Reference, computed once.
    untile(T) = reduce(vcat, [reduce(hcat, [T[i, j] for j in 1:NT]) for i in 1:NT])
    local checkfn
    if CHECK == "full"
        ref = untile(At) * untile(Bt)
        checkfn = C -> maximum(abs, untile(C) .- ref) / max(1.0, maximum(abs, ref))
    else
        ref11 = zeros(BS, BS)
        for k in 1:NT
            mul!(ref11, At[1, k], Bt[k, 1], 1.0, 1.0)
        end
        nrm = max(1.0, maximum(abs, ref11))
        checkfn = C -> maximum(abs, C[1, 1] .- ref11) / nrm
    end

    println("\nwarmup ($(WARMUP) regions per configuration) ...")
    for _ in 1:WARMUP
        Dagger.disable_memory_aware_scheduling!()
        matmul_region!(Ct, At, Bt)
    end
    if "on" in CONFIGS
        for _ in 1:WARMUP
            try
                enable_spill!()
                matmul_region!(Ct, At, Bt)
            finally
                Dagger.disable_memory_aware_scheduling!()
            end
        end
    end
    GC.gc(); GC.gc()

    results = Dict{String,Vector{NamedTuple}}("OFF" => [], "ON" => [])
    for rep in 1:REPS
        if "off" in CONFIGS
            Dagger.disable_memory_aware_scheduling!()
            r = run_instrumented(Ct, At, Bt)
            err = checkfn(Ct)
            rep == 1 && (println("\n### category inventory, OFF"); IdleAnalysis.category_totals(r.logs, 2; origin = r.origin))
            rr = IdleAnalysis.idle_report(r.logs, 2; window = r.window, origin = r.origin,
                label = @sprintf("OFF  rep %d/%d   region %.3f s   rel.err %.2e", rep, REPS, r.elapsed, err))
            rr === nothing || push!(results["OFF"], (; rr..., elapsed = r.elapsed, err, gc = r.gc))
        end
        if "on" in CONFIGS
            try
                enable_spill!()
                r = run_instrumented(Ct, At, Bt)
                err = checkfn(Ct)
                rep == 1 && (println("\n### category inventory, ON"); IdleAnalysis.category_totals(r.logs, 2; origin = r.origin))
                rr = IdleAnalysis.idle_report(r.logs, 2; window = r.window, origin = r.origin,
                    label = @sprintf("ON   rep %d/%d   region %.3f s   rel.err %.2e", rep, REPS, r.elapsed, err))
                rr === nothing || push!(results["ON"], (; rr..., elapsed = r.elapsed, err, gc = r.gc))
            finally
                Dagger.disable_memory_aware_scheduling!()
            end
        end
    end

    println("\n", "="^70)
    println("SPREAD ACROSS REPS (same process, same fixtures)")
    println("="^70)
    for cfg in ("OFF", "ON")
        all_rs = results[cfg]
        isempty(all_rs) && continue
        # Rep 1 can still carry first-touch effects even after warmup. Both
        # figures are printed so the difference stays visible.
        rs = length(all_rs) > 2 ? all_rs[2:end] : all_rs
        winlen(r) = r.window[2] - r.window[1]
        if length(all_rs) > 2
            ip = [100 * r.idle_total / winlen(r) for r in all_rs]
            @printf("%s  (incl. rep 1) idle%%: min %.1f med %.1f max %.1f  spread %.1f pp\n",
                    cfg, minimum(ip), median(ip), maximum(ip), maximum(ip) - minimum(ip))
        end
        idlepct = [100 * r.idle_total / winlen(r) for r in rs]
        @printf("%s  n=%d\n", cfg, length(rs))
        @printf("   region s : min %.3f  med %.3f  max %.3f\n",
                minimum(r.elapsed for r in rs), median([r.elapsed for r in rs]), maximum(r.elapsed for r in rs))
        @printf("   idle %%   : min %.1f  med %.1f  max %.1f   (spread %.1f pp)\n",
                minimum(idlepct), median(idlepct), maximum(idlepct), maximum(idlepct) - minimum(idlepct))
        for (name, f) in (("moving  ", r -> r.idle_moving), ("starved ", r -> r.idle_waiting),
                          ("unexpl  ", r -> r.idle_unexplained))
            v = [100 * f(r) / max(r.idle_total, eps()) for r in rs]
            @printf("   %s%% : min %.1f  med %.1f  max %.1f   (of idle)\n", name, minimum(v), median(v), maximum(v))
        end
        gcs = [r.gc for r in rs]
        @printf("   worker-2 GC s : min %.3f  med %.3f  max %.3f\n", minimum(gcs), median(gcs), maximum(gcs))
        maxerr = maximum(r.err for r in rs)
        @printf("   rel.err  : max %.2e %s\n", maxerr, maxerr < 1e-8 ? "(ok)" : "<-- INCORRECT RESULTS")
    end
end

main()
