from pico.evalutor import run_harness_regression_v2
from pico.metrics import (
    run_memory_ablation_v2, 
    run_context_ablation_v2,
    run_recovery_ablation_v2,
    write_benchmark_core_report
)

if __name__=="__main__":
    run_harness_regression_v2()
    run_memory_ablation_v2()
    run_context_ablation_v2()
    run_recovery_ablation_v2()
    write_benchmark_core_report()