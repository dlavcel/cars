import argparse
import atexit
import os
import sqlite3
import subprocess
import sys
import time
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = BASE_DIR.parent

WORKER_PATH = BASE_DIR / "worker.py"
DEFAULT_ACCOUNTS_PATH = BASE_DIR / "accounts.txt"
DB_PATH = BASE_DIR / "tasks.db"
LOGS_DIR = BASE_DIR / "logs"
LOCK_FILE = BASE_DIR / "manager.lock"


def valid_account_line(line: str) -> bool:
    parts = line.strip().split(":", 5)
    return len(parts) == 6 and all(p.strip() for p in parts)


def load_accounts(accounts_path: Path) -> list[str]:
    if not accounts_path.exists():
        raise FileNotFoundError(f"Accounts file not found: {accounts_path}")

    raw_lines = accounts_path.read_text(encoding="utf-8").splitlines()
    lines = [line.strip() for line in raw_lines if line.strip()]

    if not lines:
        raise ValueError(f"Accounts file is empty: {accounts_path}")

    valid = []
    invalid = []

    for i, line in enumerate(lines, start=1):
        if valid_account_line(line):
            valid.append(line)
        else:
            invalid.append((i, line))

    if invalid:
        print("[WARN] Invalid account lines found and skipped:")
        for idx, line in invalid:
            print(f"  line={idx}: {line}")

    if not valid:
        raise ValueError("No valid accounts found after validation")

    seen = set()
    duplicates = []
    for i, line in enumerate(valid, start=1):
        if line in seen:
            duplicates.append((i, line))
        seen.add(line)

    if duplicates:
        print("[WARN] Duplicate account lines detected:")
        for idx, line in duplicates:
            print(f"  valid_index={idx}: {line}")

    return valid


def ensure_single_manager():
    if LOCK_FILE.exists():
        try:
            old_pid = LOCK_FILE.read_text(encoding="utf-8").strip()
        except Exception:
            old_pid = "unknown"
        raise RuntimeError(f"Manager already running or stale lock exists: {LOCK_FILE} (pid={old_pid})")

    LOCK_FILE.write_text(str(os.getpid()), encoding="utf-8")


def cleanup_lock():
    try:
        if LOCK_FILE.exists():
            LOCK_FILE.unlink()
    except Exception:
        pass


def build_worker_cmd(python_exe: str, worker_path: Path, accounts_path: Path, mode: str, line_number: int) -> list[str]:
    return [
        python_exe,
        str(worker_path),
        str(accounts_path),
        mode,
        str(line_number),
    ]


def start_worker(python_exe: str, mode: str, line_number: int, accounts_path: Path) -> dict:
    LOGS_DIR.mkdir(parents=True, exist_ok=True)

    log_file = LOGS_DIR / f"{mode}_worker_{line_number}.log"
    err_file = LOGS_DIR / f"{mode}_worker_{line_number}.err.log"

    stdout_f = open(log_file, "a", encoding="utf-8")
    stderr_f = open(err_file, "a", encoding="utf-8")

    cmd = build_worker_cmd(
        python_exe=python_exe,
        worker_path=WORKER_PATH,
        accounts_path=accounts_path,
        mode=mode,
        line_number=line_number,
    )

    proc = subprocess.Popen(
        cmd,
        stdout=stdout_f,
        stderr=stderr_f,
        cwd=str(PROJECT_DIR),
    )

    return {
        "proc": proc,
        "mode": mode,
        "line_number": line_number,
        "stdout_f": stdout_f,
        "stderr_f": stderr_f,
        "log_file": log_file,
        "err_file": err_file,
        "cmd": cmd,
        "restart_count": 0,
        "started_at": time.time(),
    }


def close_process_files(p: dict):
    try:
        p["stdout_f"].close()
    except Exception:
        pass
    try:
        p["stderr_f"].close()
    except Exception:
        pass


def stop_all(processes: list[dict]):
    for p in processes:
        proc = p["proc"]
        if proc.poll() is None:
            try:
                proc.terminate()
            except Exception:
                pass

    time.sleep(2)

    for p in processes:
        proc = p["proc"]
        if proc.poll() is None:
            try:
                proc.kill()
            except Exception:
                pass

    for p in processes:
        close_process_files(p)


def print_plan(plan: list[tuple[str, int]]):
    print("Launch plan:")
    for mode, line_number in plan:
        print(f"  - {mode} line={line_number}")


def build_plan_single_mode(mode: str, workers: int, total_accounts: int) -> list[tuple[str, int]]:
    workers_to_start = min(workers, total_accounts)

    if workers > total_accounts:
        print(
            f"[WARN] Requested {workers} workers for mode={mode}, "
            f"but only {total_accounts} valid accounts available. "
            f"Starting {workers_to_start}."
        )

    return [(mode, i) for i in range(1, workers_to_start + 1)]


def build_plan_mixed(list_workers: int, detail_workers: int, total_accounts: int) -> list[tuple[str, int]]:
    requested = list_workers + detail_workers
    if requested <= 0:
        raise ValueError("Set either --mode/--workers or --list-workers/--detail-workers")

    if requested > total_accounts:
        raise ValueError(
            f"Requested {requested} workers total "
            f"({list_workers} list + {detail_workers} detail), "
            f"but only {total_accounts} valid accounts available"
        )

    plan = []
    offset = 1

    for i in range(offset, offset + list_workers):
        plan.append(("list", i))
    offset += list_workers

    for i in range(offset, offset + detail_workers):
        plan.append(("detail", i))

    return plan


def db_counts() -> dict:
    result = {
        "list_pages": {"pending": 0, "in_progress": 0, "done": 0, "failed": 0},
        "detail_tasks": {"pending": 0, "in_progress": 0, "done": 0, "failed": 0},
    }

    if not DB_PATH.exists():
        return result

    conn = sqlite3.connect(DB_PATH, timeout=10)
    cur = conn.cursor()

    try:
        cur.execute("""
            SELECT status, COUNT(*)
            FROM list_pages
            GROUP BY status
        """)
        for status, cnt in cur.fetchall():
            result["list_pages"][status] = cnt
    except sqlite3.Error:
        pass

    try:
        cur.execute("""
            SELECT status, COUNT(*)
            FROM detail_tasks
            GROUP BY status
        """)
        for status, cnt in cur.fetchall():
            result["detail_tasks"][status] = cnt
    except sqlite3.Error:
        pass

    conn.close()
    return result


def format_runtime(seconds: float) -> str:
    seconds = int(seconds)
    h = seconds // 3600
    m = (seconds % 3600) // 60
    s = seconds % 60
    if h > 0:
        return f"{h:02d}:{m:02d}:{s:02d}"
    return f"{m:02d}:{s:02d}"


def print_stats(processes: list[dict], started_at: float, last_snapshot: dict | None):
    alive = sum(1 for p in processes if p["proc"].poll() is None)
    total = len(processes)
    counts = db_counts()

    runtime = format_runtime(time.time() - started_at)

    print("\n========== MANAGER STATS ==========")
    print(f"runtime: {runtime}")
    print(f"workers alive: {alive}/{total}")

    for p in processes:
        proc = p["proc"]
        state = "alive" if proc.poll() is None else f"exit={proc.poll()}"
        print(
            f"  worker line={p['line_number']} mode={p['mode']} "
            f"pid={proc.pid} state={state} restarts={p['restart_count']}"
        )

    lp = counts["list_pages"]
    dt = counts["detail_tasks"]

    print(
        f"list_pages  | pending={lp.get('pending', 0)} "
        f"in_progress={lp.get('in_progress', 0)} "
        f"done={lp.get('done', 0)} "
        f"failed={lp.get('failed', 0)}"
    )
    print(
        f"detail_tasks| pending={dt.get('pending', 0)} "
        f"in_progress={dt.get('in_progress', 0)} "
        f"done={dt.get('done', 0)} "
        f"failed={dt.get('failed', 0)}"
    )

    if last_snapshot:
        delta_sec = max(1, time.time() - last_snapshot["ts"])

        list_done_delta = lp.get("done", 0) - last_snapshot["list_done"]
        detail_done_delta = dt.get("done", 0) - last_snapshot["detail_done"]

        list_rate = list_done_delta * 60.0 / delta_sec
        detail_rate = detail_done_delta * 60.0 / delta_sec

        print(
            f"speed       | list={list_rate:.2f}/min "
            f"detail={detail_rate:.2f}/min"
        )

    print("===================================\n")

    return {
        "ts": time.time(),
        "list_done": lp.get("done", 0),
        "detail_done": dt.get("done", 0),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--accounts", default=str(DEFAULT_ACCOUNTS_PATH), help="Path to accounts.txt")
    parser.add_argument("--mode", choices=["list", "detail"], help="Run only one mode")
    parser.add_argument("--workers", type=int, default=0, help="Number of workers for --mode")
    parser.add_argument("--list-workers", type=int, default=0, help="Number of list workers")
    parser.add_argument("--detail-workers", type=int, default=0, help="Number of detail workers")
    parser.add_argument("--restart", action="store_true", help="Restart worker if it exits")
    parser.add_argument("--max-restarts", type=int, default=20, help="Maximum restarts per worker when --restart is enabled")
    parser.add_argument("--stagger", type=float, default=3.0, help="Delay between worker starts in seconds")
    parser.add_argument("--monitor-interval", type=float, default=5.0, help="Manager loop sleep interval in seconds")
    parser.add_argument("--stats-interval", type=float, default=15.0, help="How often to print stats in seconds")
    args = parser.parse_args()

    accounts_path = Path(args.accounts).resolve()

    ensure_single_manager()
    atexit.register(cleanup_lock)

    accounts = load_accounts(accounts_path)
    total_accounts = len(accounts)

    python_exe = sys.executable

    print(f"Python: {python_exe}")
    print(f"Worker path: {WORKER_PATH}")
    print(f"Accounts path: {accounts_path}")
    print(f"DB path: {DB_PATH}")
    print(f"Valid accounts: {total_accounts}")

    if args.mode:
        if args.workers <= 0:
            raise ValueError("With --mode you must set --workers > 0")
        plan = build_plan_single_mode(args.mode, args.workers, total_accounts)
    else:
        plan = build_plan_mixed(args.list_workers, args.detail_workers, total_accounts)

    if not plan:
        raise ValueError("No workers to start")

    print_plan(plan)

    processes: list[dict] = []
    started_at = time.time()
    last_stats_print = 0.0
    snapshot = None

    try:
        for mode, line_number in plan:
            p = start_worker(
                python_exe=python_exe,
                mode=mode,
                line_number=line_number,
                accounts_path=accounts_path,
            )
            processes.append(p)
            print(
                f"[STARTED] mode={mode} line={line_number} "
                f"pid={p['proc'].pid} "
                f"log={p['log_file'].name}"
            )
            time.sleep(max(0.0, args.stagger))

        while True:
            time.sleep(max(0.5, args.monitor_interval))

            all_exited = True

            for p in processes:
                proc = p["proc"]
                code = proc.poll()

                if code is None:
                    all_exited = False
                    continue

                print(
                    f"[EXIT] mode={p['mode']} line={p['line_number']} "
                    f"code={code} restarts={p['restart_count']}"
                )
                close_process_files(p)

                if args.restart and p["restart_count"] < args.max_restarts:
                    print(
                        f"[RESTART] mode={p['mode']} line={p['line_number']} "
                        f"attempt={p['restart_count'] + 1}/{args.max_restarts}"
                    )
                    new_p = start_worker(
                        python_exe=python_exe,
                        mode=p["mode"],
                        line_number=p["line_number"],
                        accounts_path=accounts_path,
                    )
                    new_p["restart_count"] = p["restart_count"] + 1
                    p.update(new_p)
                    print(
                        f"[RESTARTED] mode={p['mode']} line={p['line_number']} "
                        f"pid={p['proc'].pid}"
                    )
                    all_exited = False

            now = time.time()
            if now - last_stats_print >= max(1.0, args.stats_interval):
                snapshot = print_stats(processes, started_at, snapshot)
                last_stats_print = now

            if not args.restart and all_exited:
                print("[DONE] All workers exited")
                break

            if args.restart and all(
                p["proc"].poll() is not None and p["restart_count"] >= args.max_restarts
                for p in processes
            ):
                print("[DONE] All workers exited and reached max restarts")
                break

    except KeyboardInterrupt:
        print("\n[STOP] Ctrl+C received, terminating workers...")
        stop_all(processes)
        print("[STOPPED]")


if __name__ == "__main__":
    main()