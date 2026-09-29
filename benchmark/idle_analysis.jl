### Idle-duration accounting for Dagger regions.
#
# Replaces the three weaknesses of the original `test.jl` accounting:
#
#  1. `total_busy = sum(t1 - t0)` over `:compute` intervals double-counts when a
#     worker runs several tasks concurrently (one `ThreadProc` per thread), so
#     "busy" could exceed the span and idle% went nonsensical. We take the
#     *union* of intervals instead, and report per-processor occupancy separately.
#
#  2. Only `:compute` counted as busy, so every `:move` / `:datadeps_copy` on the
#     worker was scored as idle. Since spilling exists to create exactly that
#     copy traffic, the headline "idle %" mixed "starved" with "moving data".
#     Here each idle interval is intersected with the movement intervals, so the
#     split is exact rather than a +/-0.5s coincidence test against debug messages.
#
#  3. The denominator was the span between the worker's own first and last
#     `:compute` event, which moves with placement and stragglers. We take the
#     measurement window from the caller (region wall-clock) and also report the
#     old span-based number so results stay comparable to earlier runs.
#
# The output is the *distribution* of idle durations, not just a percentage:
# a few long stalls and thousands of short ones need different fixes.

module IdleAnalysis

using Printf, Statistics
import Dagger

# Categories that mean "this worker is doing useful compute".
const COMPUTE_CATEGORIES = Set([:compute])

# Categories that mean "this worker is moving data" (transfer, spill, reload).
# `:datadeps_copy_skip` is deliberately excluded: an elided copy consumes no time.
#
# N.B. Measured on a remote gemm region these carry ~0 s (435 `:move` events
# totalling under a millisecond, no `:datadeps_copy` on the compute worker at
# all). Data movement is NOT where the idle time goes -- keep the split so that
# stays visible rather than assumed.
const MOVEMENT_CATEGORIES = Set([:move, :datadeps_copy])

# Categories that mean "this worker had nothing to run". `:proc_run_wait` is the
# processor runner blocked waiting for a task to appear in its queue (Sch.jl's
# runner loop), so it measures starvation *from upstream* -- planning and
# scheduling on the submitting worker -- not local I/O. On the validation run it
# was 0.835 s against 0.334 s of compute, i.e. the dominant term.
const STARVATION_CATEGORIES = Set([:proc_run_wait, :proc_run_fetch, :proc_steal_local])

"""
    merge_intervals(ivals) -> Vector{Tuple{Float64,Float64}}

Union of possibly-overlapping intervals, as a sorted disjoint set. This is the
fix for double-counting concurrent tasks on one worker.
"""
function merge_intervals(ivals::Vector{Tuple{Float64,Float64}})
    isempty(ivals) && return Tuple{Float64,Float64}[]
    s = sort(ivals; by = first)
    out = Tuple{Float64,Float64}[s[1]]
    for k in 2:length(s)
        a, b = s[k]
        la, lb = out[end]
        if a <= lb                      # overlapping or touching -> extend
            out[end] = (la, max(lb, b))
        else
            push!(out, (a, b))
        end
    end
    return out
end

total_duration(ivals) = isempty(ivals) ? 0.0 : sum(b - a for (a, b) in ivals)

"""
    complement(ivals, lo, hi) -> Vector{Tuple{Float64,Float64}}

The gaps between `ivals` (assumed merged/disjoint) inside the window `[lo, hi]`.
These are the idle intervals.
"""
function complement(ivals::Vector{Tuple{Float64,Float64}}, lo::Float64, hi::Float64)
    gaps = Tuple{Float64,Float64}[]
    cursor = lo
    for (a, b) in ivals
        b <= lo && continue
        a >= hi && break
        a > cursor && push!(gaps, (cursor, min(a, hi)))
        cursor = max(cursor, b)
    end
    cursor < hi && push!(gaps, (cursor, hi))
    return gaps
end

"""
    overlap_total(as, bs) -> Float64

Total length of the intersection of two merged interval sets. Used to split each
idle interval into "explained by data movement" and "unexplained".
"""
function overlap_total(as::Vector{Tuple{Float64,Float64}}, bs::Vector{Tuple{Float64,Float64}})
    i = j = 1
    acc = 0.0
    while i <= length(as) && j <= length(bs)
        lo = max(as[i][1], bs[j][1])
        hi = min(as[i][2], bs[j][2])
        hi > lo && (acc += hi - lo)
        as[i][2] < bs[j][2] ? (i += 1) : (j += 1)
    end
    return acc
end

"""
    gather_events(logs) -> Vector

Flatten Dagger's logs into `(worker, category, t0_ns, t1_ns, processor)` records.
`t0/t1` stay in raw nanoseconds; the caller picks the time origin.
"""
function gather_events(logs)
    recs = NamedTuple[]
    Dagger.logs_event_pairs(logs) do w, start_idx, finish_idx
        core = logs[w][:core]
        id = logs[w][:id][start_idx]
        push!(recs, (; worker = w,
                       category = core[start_idx].category,
                       t0 = core[start_idx].timestamp,
                       t1 = core[finish_idx].timestamp,
                       processor = hasproperty(id, :processor) ? id.processor : nothing))
    end
    return recs
end

"""
    log_time_origin(logs) -> UInt64

Earliest timestamp across all workers, for converting to seconds-since-start.
"""
log_time_origin(logs) = minimum(minimum(e.timestamp for e in logs[w][:core]) for w in keys(logs))

"""
    idle_report(logs, worker; window=nothing, origin=nothing, label="") -> NamedTuple

Idle-duration accounting for one worker.

`window` is `(t_start, t_end)` in seconds relative to `origin`, normally the
measured region wall-clock. When omitted it falls back to the worker's own
compute span, which reproduces the original script's denominator -- reported
alongside so old and new numbers can be compared.
"""
function idle_report(logs, worker::Int; window = nothing, origin = nothing, label::AbstractString = "")
    origin = something(origin, log_time_origin(logs))
    recs = filter(r -> r.worker == worker, gather_events(logs))
    secs(t) = (Float64(t) - Float64(origin)) / 1e9

    compute_raw = [(secs(r.t0), secs(r.t1)) for r in recs if r.category in COMPUTE_CATEGORIES]
    move_raw    = [(secs(r.t0), secs(r.t1)) for r in recs if r.category in MOVEMENT_CATEGORIES]
    starve_raw  = [(secs(r.t0), secs(r.t1)) for r in recs if r.category in STARVATION_CATEGORIES]

    if isempty(compute_raw)
        @warn "no :compute events for worker $worker; nothing to report"
        return nothing
    end

    compute = merge_intervals(compute_raw)
    movement = merge_intervals(move_raw)
    starvation = merge_intervals(starve_raw)

    # Naive sum kept only to quantify how much concurrency the old metric
    # double-counted.
    naive_busy = total_duration(compute_raw)
    busy = total_duration(compute)

    span_lo, span_hi = compute[1][1], compute[end][2]
    lo, hi = window === nothing ? (span_lo, span_hi) : (Float64(window[1]), Float64(window[2]))

    idle_ivals = complement(compute, lo, hi)
    idle_total = total_duration(idle_ivals)
    idle_moving = overlap_total(idle_ivals, movement)
    # Starvation is attributed only where it is NOT already counted as movement,
    # so the three buckets sum to the idle total instead of double-counting.
    starve_only = merge_intervals(vcat(starvation, movement))
    idle_waiting = overlap_total(idle_ivals, starve_only) - idle_moving
    idle_unexplained = idle_total - idle_moving - idle_waiting

    durs = sort!([b - a for (a, b) in idle_ivals])
    pct(x) = 100 * x / max(hi - lo, eps())

    if !isempty(label)
        println("\n", "="^70)
        println(label)
        println("="^70)
    end
    @printf("worker %d   window %.3f s  (%s)\n", worker, hi - lo,
            window === nothing ? "compute span - old denominator" : "region wall-clock")
    @printf("  compute tasks:        %d\n", length(compute_raw))
    @printf("  busy (union):         %8.3f s  (%5.1f%%)\n", busy, pct(busy))
    @printf("  busy (naive sum):     %8.3f s   <- old metric; %.2fx overlap\n",
            naive_busy, naive_busy / max(busy, eps()))
    @printf("  idle (no compute):    %8.3f s  (%5.1f%%)\n", idle_total, pct(idle_total))
    @printf("    moving data:        %8.3f s  (%5.1f%% of idle)  [:move/:datadeps_copy]\n",
            idle_moving, 100 * idle_moving / max(idle_total, eps()))
    @printf("    starved (no work):  %8.3f s  (%5.1f%% of idle)  [:proc_run_wait/fetch/steal]\n",
            idle_waiting, 100 * idle_waiting / max(idle_total, eps()))
    @printf("    unexplained:        %8.3f s  (%5.1f%% of idle)\n",
            idle_unexplained, 100 * idle_unexplained / max(idle_total, eps()))

    if !isempty(durs)
        println("  idle-gap durations (s):")
        @printf("    count %d   min %.6f   p50 %.6f   p90 %.6f   p99 %.6f   max %.6f\n",
                length(durs), durs[1], quantile(durs, 0.5), quantile(durs, 0.9),
                quantile(durs, 0.99), durs[end])
        # Where the idle time actually lives: a few long stalls vs. many short ones.
        edges = [0.0, 1e-4, 1e-3, 1e-2, 1e-1, 1.0, Inf]
        names = ["<0.1ms", "0.1-1ms", "1-10ms", "10-100ms", "0.1-1s", ">1s"]
        println("    histogram (count / summed seconds / % of idle):")
        for b in 1:length(edges)-1
            sel = filter(d -> edges[b] <= d < edges[b+1], durs)
            isempty(sel) && continue
            @printf("      %-9s %6d  %8.3f s  %5.1f%%\n",
                    names[b], length(sel), sum(sel), 100 * sum(sel) / max(idle_total, eps()))
        end
    end

    # Per-processor occupancy: distinguishes "worker had nothing to do" from
    # "worker had one thread busy and the rest starved".
    byproc = Dict{Any,Vector{Tuple{Float64,Float64}}}()
    for r in recs
        r.category in COMPUTE_CATEGORIES || continue
        push!(get!(Vector{Tuple{Float64,Float64}}, byproc, r.processor), (secs(r.t0), secs(r.t1)))
    end
    if length(byproc) > 1
        println("  per-processor busy (union):")
        for (p, ivals) in sort(collect(byproc); by = x -> string(x[1]))
            b = total_duration(merge_intervals(ivals))
            @printf("    %-42s %8.3f s  (%5.1f%%)\n", string(p), b, pct(b))
        end
        occ = sum(total_duration(merge_intervals(v)) for v in values(byproc))
        @printf("  aggregate occupancy: %.1f%% of %d processors x window\n",
                100 * occ / (length(byproc) * max(hi - lo, eps())), length(byproc))
    end

    return (; worker, window = (lo, hi), busy, naive_busy, idle_total,
              idle_moving, idle_waiting, idle_unexplained, durations = durs,
              span = (span_lo, span_hi))
end

"""
    category_totals(logs, worker) -> Nothing

Union-total time per log category on `worker`. Run this first on any new
workload: it shows which categories actually carry time, so nothing significant
is silently scored as idle.
"""
function category_totals(logs, worker::Int; origin = nothing)
    origin = something(origin, log_time_origin(logs))
    recs = filter(r -> r.worker == worker, gather_events(logs))
    secs(t) = (Float64(t) - Float64(origin)) / 1e9
    bycat = Dict{Symbol,Vector{Tuple{Float64,Float64}}}()
    for r in recs
        push!(get!(Vector{Tuple{Float64,Float64}}, bycat, r.category), (secs(r.t0), secs(r.t1)))
    end
    println("worker $worker - union time by category:")
    for (cat, ivals) in sort(collect(bycat); by = x -> -total_duration(merge_intervals(x[2])))
        @printf("  %-22s %6d events  %8.3f s\n", cat, length(ivals),
                total_duration(merge_intervals(ivals)))
    end
    return nothing
end

end # module
