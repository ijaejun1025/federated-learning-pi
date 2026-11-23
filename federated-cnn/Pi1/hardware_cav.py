#!/usr/bin/env python
# coding: utf-8
"""Hardware measurements for the results (hardware_metrics.csv on the server).

Standard library only (no psutil), so nothing extra has to be installed on a
Pi. Every reading is optional: one this machine cannot provide (Windows has no
``resource`` module or /sys files, only a Raspberry Pi has ``vcgencmd``) is
left out of the result rather than raised, so a measurement can never stop
training. All keys start with ``hw_``; the server strips the prefix and
writes them to hardware_metrics.csv, not to client_metrics.csv.

- wall time, CPU time and CPU use (``time.process_time`` covers all of the
  process's threads, so TensorFlow's too), as a share of all cores;
- memory: the process's peak resident set since it started
  (``ru_maxrss``), its current one, and the machine's available memory and
  used swap at the end;
- CPU temperature and clock, sampled every ``SAMPLE_SEC`` seconds in a
  background thread (start / max / end temperature, min / mean clock: a
  clock below the maximum means the Pi throttled);
- ``vcgencmd get_throttled`` (Raspberry Pi): bits 0-3 are the state now
  (under-voltage, frequency capped, throttled, soft temperature limit),
  bits 16-19 the same since boot;
- inference latency of one CAN frame at a time (``frame_latency_ms``).
"""
import os
import platform
import subprocess
import threading
import time

try:
    import resource  # Linux / macOS only
except ImportError:  # Windows
    resource = None

SAMPLE_SEC = 5
TEMP_FILE = "/sys/class/thermal/thermal_zone0/temp"
FREQ_FILE = "/sys/devices/system/cpu/cpu0/cpufreq/scaling_cur_freq"


def _read_number(path, scale):
    try:
        with open(path) as f:
            return int(f.read().strip()) / scale
    except Exception:  # noqa: BLE001
        return None


def _temp_c():
    return _read_number(TEMP_FILE, 1000.0)


def _freq_mhz():
    return _read_number(FREQ_FILE, 1000.0)


def _meminfo_mb():
    """/proc/meminfo in MB ({} where there is none)."""
    out = {}
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                key, value = line.split(":", 1)
                out[key] = int(value.split()[0]) / 1024.0
    except Exception:  # noqa: BLE001
        pass
    return out


def _rss_mb():
    try:
        with open("/proc/self/status") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) / 1024.0
    except Exception:  # noqa: BLE001
        pass
    return None


def _peak_rss_mb():
    if resource is None:
        return None
    try:
        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0  # KB on Linux
    except Exception:  # noqa: BLE001
        return None


def _throttled():
    try:
        out = subprocess.run(["vcgencmd", "get_throttled"], capture_output=True, text=True, timeout=5).stdout
        return out.strip().split("=", 1)[1]  # e.g. "0x50005"
    except Exception:  # noqa: BLE001
        return None


def device_info():
    """What machine this is (sent with every fit result; constant per Pi)."""
    info = {"hw_cpu_count": os.cpu_count()}
    try:
        with open("/proc/device-tree/model") as f:
            info["hw_device_model"] = f.read().strip("\x00\n ")
    except Exception:  # noqa: BLE001
        info["hw_device_model"] = platform.platform()
    info["hw_mem_total_mb"] = _meminfo_mb().get("MemTotal")
    return _clean(info)


def _clean(metrics):
    """Flower accepts only int / float / str / bytes / bool values: drop
    the readings this machine lacks (None) and turn the rest into plain
    Python numbers."""
    out = {}
    for key, value in metrics.items():
        if value is None:
            continue
        if isinstance(value, (bool, str, bytes)):
            out[key] = value
        elif isinstance(value, int):
            out[key] = int(value)
        else:
            try:
                out[key] = float(value)
            except Exception:  # noqa: BLE001
                continue
    return out


def array_bytes(arrays):
    """Raw size of a list of weight arrays (Flower's message adds a few bytes per array)."""
    try:
        return int(sum(a.nbytes for a in arrays))
    except Exception:  # noqa: BLE001
        return None


class HardwareMeter:
    """Measures one phase (a client's fit or evaluate, a server round):
    create it at the start, call ``stop()`` at the end."""

    def __init__(self):
        self._wall0 = time.time()
        self._cpu0 = time.process_time()
        self._temp_start = _temp_c()
        self._temps = [] if self._temp_start is None else [self._temp_start]
        first_freq = _freq_mhz()
        self._freqs = [] if first_freq is None else [first_freq]
        self._stop = threading.Event()
        self._thread = None
        if self._temp_start is not None or first_freq is not None:
            self._thread = threading.Thread(target=self._sample, daemon=True)
            self._thread.start()

    def _sample(self):
        while not self._stop.wait(SAMPLE_SEC):
            temp, freq = _temp_c(), _freq_mhz()
            if temp is not None:
                self._temps.append(temp)
            if freq is not None:
                self._freqs.append(freq)

    def stop(self, rows=None, **extra):
        """The ``hw_*`` metrics of this phase; ``rows`` adds rows per second,
        ``extra`` (e.g. received_bytes=...) is added as ``hw_<name>``."""
        self._stop.set()
        metrics = {}
        try:
            wall = time.time() - self._wall0
            cpu = time.process_time() - self._cpu0
            metrics["hw_wall_seconds"] = wall
            metrics["hw_cpu_seconds"] = cpu
            if wall > 0 and os.cpu_count():
                metrics["hw_cpu_util_pct"] = 100.0 * cpu / wall / os.cpu_count()
            if rows:
                metrics["hw_rows"] = int(rows)
                if wall > 0:
                    metrics["hw_rows_per_sec"] = rows / wall
            metrics["hw_peak_rss_mb"] = _peak_rss_mb()
            metrics["hw_rss_mb"] = _rss_mb()
            mem = _meminfo_mb()
            metrics["hw_mem_available_mb"] = mem.get("MemAvailable")
            if "SwapTotal" in mem and "SwapFree" in mem:
                metrics["hw_swap_used_mb"] = mem["SwapTotal"] - mem["SwapFree"]
            temps, freqs = list(self._temps), list(self._freqs)
            end_temp, end_freq = _temp_c(), _freq_mhz()
            if end_temp is not None:
                temps.append(end_temp)
            if end_freq is not None:
                freqs.append(end_freq)
            if temps:
                metrics["hw_temp_start_c"] = self._temp_start
                metrics["hw_temp_max_c"] = max(temps)
                metrics["hw_temp_end_c"] = temps[-1]
            if freqs:
                metrics["hw_cpu_freq_min_mhz"] = min(freqs)
                metrics["hw_cpu_freq_mean_mhz"] = sum(freqs) / len(freqs)
            metrics["hw_throttled"] = _throttled()
            for key, value in extra.items():
                metrics[f"hw_{key}"] = value
        except Exception:  # noqa: BLE001
            pass
        return _clean(metrics)


def frame_latency_ms(model, x, frames=500, warmup=20):
    """Median and 95th-percentile time (ms) to classify one CAN frame on its
    own (batch of 1, as a deployed IDS would see frames arrive), over
    ``frames`` frames of ``x`` after ``warmup`` untimed calls. Features are
    precomputed, so this is the model alone."""
    try:
        n = min(len(x), frames + warmup)
        times = []
        for i in range(n):
            t0 = time.perf_counter()
            model(x[i:i + 1], training=False)
            if i >= warmup:
                times.append((time.perf_counter() - t0) * 1000.0)
        if not times:
            return {}
        times.sort()
        return _clean({"hw_infer_frame_ms_median": times[len(times) // 2],
                       "hw_infer_frame_ms_p95": times[min(len(times) - 1, int(0.95 * len(times)))],
                       "hw_infer_frames": len(times)})
    except Exception:  # noqa: BLE001
        return {}
