# Bravli — Gemini CLI Adapter

@./system.md

## Gemini-Specific Instructions

1. **Massive Context Synthesis**: Prioritize synthesizing information across
   the entire loaded context rather than summarizing it.
2. **NEVER guess file paths**. Use your tools to explore before acting.
3. **Read before Editing**. Always read a file into context before modifying.
4. **Non-interactive Shell**: Do not invoke `vim`, `nano`, or tools requiring
   standard input.
5. **Grounded Verification**: Do not rely on latent knowledge for neuroscience
   claims. Use `google_web_search` to verify against recent literature when
   making quantitative assertions about connectome data, circuit models,
   or synaptic physiology.

## Git Conventions

- Do not commit unless asked
- Do not push unless asked
