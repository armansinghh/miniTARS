# llm-memory-os

trying to actually implement the "LLM as an OS" memory management idea from research, from scratch.

the goal: teach an LLM to manage its own memory (what to keep, what to drop, when to recall it) the way an OS manages RAM vs disk. if it works, anyone can build their own long term memory assistant on top of it.

this is a side project, means slow and research driven!

## what it's based on
- [MemGPT: Towards LLMs as Operating Systems](https://arxiv.org/abs/2310.08560) the core idea: virtual memory management for LLMs
- [Mem0: Building Production-Ready AI Agents with Scalable Long-Term Memory](https://arxiv.org/abs/2504.19413)
- [LongMemEval: Benchmarking Chat Assistants on Long-Term Interactive Memory](https://arxiv.org/abs/2410.10813) how "good memory" is even measured
- [A-MEM: Agentic Memory for LLM Agents](https://arxiv.org/abs/2502.12110)

## status
just getting started - reading + scoping the implementation. will update as it actually takes shape.
