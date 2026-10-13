# Agent responsibilities

## Task monitoring

- Delegate monitoring of long builds, tests, benchmarks, and other running jobs
  to a Luna subagent (`gpt-6-luna`) while the primary agent continues useful work.
- Give Luna the command or job identity, working directory, log and result paths,
  expected completion signal, and the conditions that require attention.
- Luna checks process state, log progress, exit status, and available results.
  Use bounded waits and avoid frequent polling or duplicate monitors.
- Prefer shared log and status files for cross-agent monitoring. If an execution
  session is visible only to its owner, the owning agent collects its final exit
  status; Luna reports the evidence available from the shared files.
- Report meaningful progress, completion, failures, suspected stalls, or resource
  exhaustion to the primary agent. Include the relevant log excerpt and paths.
  Silence alone is not proof that a numerical factorization has stalled.
- Monitoring is read-only. Luna must not modify solver code, change benchmark
  parameters, launch competing workloads, or terminate/restart jobs unless the
  primary agent explicitly delegates that action.
- The primary agent owns numerical analysis, implementation, benchmark design,
  recovery decisions, and the final report. Verify monitored results before
  reporting a job as successful.
- Reuse an existing Luna monitor when practical. If Luna is unavailable, report
  that limitation and monitor locally without silently choosing another model.

## Numerical benchmarks

- Use the repository's `.venv` and record the source commit, hardware, package
  versions, selected GPU, CPU thread counts, and exact commands.
- Run timed workloads sequentially to avoid interference. Keep warmup and mesh
  assembly separate from solve timings; include transfers and factorization in
  end-to-end CPU/GPU comparisons.
- Preserve raw timing samples and failures. Check convergence, original-pencil
  residuals, and matched eigenvalues before reporting a speedup.
- GPU access in this environment requires execution outside the sandbox. A
  missing device inside the sandbox does not establish that the host lacks GPUs.
