# Ailog RAG Skills for Claude Code

Professional skills for building, auditing, evaluating, and optimizing RAG (Retrieval-Augmented Generation) systems with Claude Code.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Claude Code](https://img.shields.io/badge/Claude%20Code-Skills-blue)](https://claude.ai/code)

## Overview

These skills help you build production-grade RAG pipelines by providing:

| Skill | Command | Description |
|-------|---------|-------------|
| **RAG Audit** | `/rag-audit` | Analyze existing RAG code for anti-patterns and issues |
| **RAG Eval** | `/rag-eval` | Evaluate RAG quality with metrics and benchmarking |
| **Chunking Advisor** | `/chunking-advisor` | Get optimal chunking strategy recommendations |
| **RAG Scaffold** | `/rag-scaffold` | Generate production-ready RAG boilerplate |

## Quick Start

### Installation

**Option 1: Via Claude Code Marketplace (Recommended)**

```bash
# Add the marketplace
/plugin marketplace add https://github.com/floflo777/claude-rag-skills

# Install all skills
/plugin install rag-audit
/plugin install rag-eval
/plugin install chunking-advisor
/plugin install rag-scaffold
```

**Option 2: Manual Installation**

```bash
# Clone the repository
git clone https://github.com/floflo777/claude-rag-skills.git

# Copy to your Claude Code skills directory
cp -r claude-rag-skills/* ~/.claude/skills/

# Or for project-specific installation
cp -r claude-rag-skills/* .claude/skills/
```

### Usage

After installation, use the skills in any Claude Code session:

```
You: /rag-audit
Claude: I'll analyze your codebase for RAG-related code and check for anti-patterns...

You: /chunking-advisor
Claude: What types of documents will you be indexing? What embedding model are you using?

You: /rag-scaffold
Claude: I'll help you generate a production-ready RAG pipeline. What's your preferred framework?

You: /rag-eval
Claude: Let's evaluate your RAG system. Do you have a test dataset, or should I help create one?
```

## Skills Documentation

### `/rag-audit` - RAG Code Auditor

Scans your codebase for RAG implementations and identifies:

- **Chunking issues**: Wrong size, no overlap, boundary problems
- **Embedding problems**: Model mismatch, no caching, batch issues
- **Retrieval anti-patterns**: Fixed top-k, no reranking, missing hybrid search
- **Generation issues**: Context overflow, poor prompts, no citations
- **Production gaps**: Missing error handling, logging, caching

**Example output:**

```markdown
# RAG Audit Report

## Summary
- Files Analyzed: 12
- Issues Found: 8 (2 critical, 4 warnings, 2 suggestions)
- Overall Score: 72/100

## Critical Issues

### No Chunk Overlap
**Location**: `src/chunker.py:45`
**Issue**: Chunks created with overlap=0
**Impact**: Information at chunk boundaries will be lost
**Fix**: Add 10-20% overlap
```

### `/rag-eval` - RAG Evaluator

Evaluates your RAG system with standard metrics:

**Retrieval Metrics:**
- Recall@K, Precision@K
- Mean Reciprocal Rank (MRR)
- Normalized Discounted Cumulative Gain (NDCG)

**Generation Metrics:**
- Faithfulness (grounded in context)
- Relevance (answers the question)
- Coherence and conciseness

**Optional: Ailog Benchmark**

Compare your system against Ailog's production RAG API:

```bash
export AILOG_API_KEY="pk_live_your_key"
export AILOG_WORKSPACE_ID="123"
```

### `/chunking-advisor` - Chunking Strategy Expert

Get recommendations based on:

- Document type (code, legal, FAQ, articles, tables)
- Query patterns (factual, analytical, comparative)
- Embedding model (token limits, optimal sizes)
- Performance requirements

**Decision tree included** for quick strategy selection.

### `/rag-scaffold` - RAG Boilerplate Generator

Generate complete, production-ready RAG pipelines:

**Framework Options:**
- Python + LangChain + Qdrant
- Python + LlamaIndex
- Python Vanilla (no framework)
- TypeScript + LangChain.js
- Ailog API (managed RAG)

**Includes:**
- Configuration management
- Embedding service with caching
- Vector store operations
- Retrieval with reranking
- Generation with streaming
- Docker setup
- Tests

## Ailog Integration

These skills reference [Ailog's RAG guides](https://app.ailog.fr/en/blog/guides) for best practices:

- [Chunking Strategies](https://app.ailog.fr/en/blog/guides/chunking-strategies)
- [Choosing Embedding Models](https://app.ailog.fr/en/blog/guides/choosing-embedding-models)
- [Hybrid Search](https://app.ailog.fr/en/blog/guides/hybrid-search-rag)
- [Reranking](https://app.ailog.fr/en/blog/guides/reranking)
- [RAG Evaluation](https://app.ailog.fr/en/blog/guides/rag-evaluation)
- [Production Deployment](https://app.ailog.fr/en/blog/guides/production-deployment)

**Optional API Integration:**

The `/rag-eval` skill can benchmark against Ailog's API for objective comparison. Create a free workspace at [ailog.fr](https://ailog.fr) to use this feature.

## Project Structure

```
claude-rag-skills/
├── rag-audit/
│   └── SKILL.md           # Audit skill instructions
├── rag-eval/
│   └── SKILL.md           # Evaluation skill instructions
├── chunking-advisor/
│   └── SKILL.md           # Chunking advice instructions
├── rag-scaffold/
│   └── SKILL.md           # Scaffold generation instructions
├── examples/
│   └── ...                # Example configurations
├── marketplace.json       # Plugin marketplace metadata
└── README.md
```

## Requirements

- Claude Code >= 2.0.0
- For Ailog benchmarking: Ailog API key (optional)

## Contributing

Contributions welcome! Please:

1. Fork the repository
2. Create a feature branch
3. Submit a pull request

For major changes, open an issue first to discuss.

## License

MIT License - see [LICENSE](LICENSE) for details.

## Support

- **Documentation**: [Ailog Guides](https://app.ailog.fr/en/blog/guides)
- **Issues**: [GitHub Issues](https://github.com/floflo777/claude-rag-skills/issues)
- **Discord**: [Ailog Community](https://discord.gg/ailog)

---

Built with expertise from [Ailog](https://ailog.fr) - The RAG-as-a-Service Platform
