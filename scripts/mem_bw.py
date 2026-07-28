"""Measure achievable DRAM read/copy bandwidth on this x86 CPU, multithreaded (numpy releases the
GIL during the C loop, so threads parallelize across cores). Reports peak read GB/s -- the analogue
of the STREAM-measured 13.98 GB/s used for the Raspberry Pi 5 roofline."""
import numpy as np, time, os
from concurrent.futures import ThreadPoolExecutor

def bench_read(total_bytes, nthreads):
    n = total_bytes // 8  # float64
    a = np.ones(n, dtype=np.float64)
    chunks = np.array_split(np.arange(n), nthreads)
    def work(idx): return float(a[idx[0]:idx[-1]+1].sum())
    best = 0.0
    for _ in range(6):
        t = time.perf_counter()
        with ThreadPoolExecutor(max_workers=nthreads) as ex:
            list(ex.map(work, chunks))
        dt = time.perf_counter() - t
        best = max(best, a.nbytes / dt / 1e9)
    return best

def bench_copy(total_bytes, nthreads):
    n = total_bytes // 8
    a = np.ones(n, dtype=np.float64); b = np.empty_like(a)
    bounds = np.array_split(np.arange(n), nthreads)
    def work(idx): s=idx[0]; e=idx[-1]+1; b[s:e]=a[s:e]; return 0
    best=0.0
    for _ in range(6):
        t=time.perf_counter()
        with ThreadPoolExecutor(max_workers=nthreads) as ex:
            list(ex.map(work, bounds))
        dt=time.perf_counter()-t
        best=max(best, 2*a.nbytes/dt/1e9)  # read+write
    return best

if __name__ == "__main__":
    GB = 2_000_000_000  # 2 GB working set (exceeds caches)
    ncpu = os.cpu_count()
    results = {}
    for nt in [1, 4, 8, ncpu]:
        r = bench_read(GB, nt)
        results[nt] = r
        print(f"read BW @ {nt} threads: {r:.1f} GB/s", flush=True)
    cp = bench_copy(GB, ncpu)
    print(f"copy BW @ {ncpu} threads: {cp:.1f} GB/s (read+write)", flush=True)
    peak = max(results.values())
    print(f"PEAK_READ_GBs={peak:.2f}")
    import json; json.dump({"peak_read_GBs":peak,"copy_GBs":cp,"by_threads":results,
                            "theoretical_ddr5_4800_dual":76.8},
                           open(r"C:\llmpc\mem_bw.json","w"), indent=2)
