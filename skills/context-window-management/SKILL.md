---
name: context-window-management
description: Guides agents on determining when and how to refresh, clear, or change their context window. Use when agent output quality degrades, memory issues occur, or when switching to an entirely different task.
---

# Context Window Management

## Overview
Context windows are finite and precious. As a conversation grows, older context becomes stale and can confuse the agent or degrade performance. This skill provides a systematic approach for determining exactly when to change or clear the context window, ensuring the agent remains sharp, focused, and free from past hallucinations.

## When to Use
- Agent output quality is noticeably declining (e.g., hallucinating APIs, forgetting recently stated rules).
- The agent is stuck in an error-correction loop, repeatedly failing to fix the same bug.
- You are switching from one major feature or domain to an entirely different one.
- The conversation has grown extremely long (e.g., hundreds of turns) and responses are slowing down.
- **NOT** when you are mid-debugging a specific issue and need the immediate terminal output history.

## Core Process

1. **Evaluate Current Context Quality:**
   - Are recent responses ignoring project rules?
   - Is the agent confusing old task requirements with the new task?
   - Have we shifted domains? (e.g., moving from database schema design to frontend CSS styling).

2. **Determine the Refresh Strategy:**
   - **Soft Refresh:** Summarize the current state and explicitly tell the agent to ignore previous assumptions.
   - **Hard Refresh:** Start a completely new session or clear the conversation history, carrying over only the final summaries or rules.

3. **Execute the Refresh:**
   - Create a summary of "What we have done" and "What is left to do".
   - Save any unresolved questions or pending tasks to a scratchpad or artifact.
   - Initiate the context clear or start a new chat/session.

4. **Re-Initialize Context:**
   - Load the core rules files (e.g., `CLAUDE.md`, `AGENTS.md`).
   - Feed the summary of the previous session.
   - Load only the files strictly necessary for the next task.

## Common Rationalizations

| Rationalization | Reality |
|---|---|
| "I'll just tell the agent to forget the previous instructions." | LLMs struggle to un-see what is in their context window. A fresh session is always more reliable than negative prompting. |
| "Starting a new session takes too much time to set up." | An agent hallucinating due to stale context will cost you hours of debugging. Setting up a new session takes minutes. |
| "But I might need that old context later." | If it's important, summarize it and bring the summary to the new session. Raw conversation logs are rarely needed once a task is done. |

## Red Flags
- The agent repeatedly tries to use a variable or function name that was deleted 20 turns ago.
- The agent apologizes for the same mistake multiple times without fixing it.
- You find yourself writing prompts like "Like I said 5 messages ago..."

## Verification
After executing a context window change, confirm:
- [ ] A new session was started or the context was explicitly cleared.
- [ ] A summary of previous progress was carried over.
- [ ] Core rule files and only the currently relevant source files were re-loaded.
- [ ] The agent's first response in the new context demonstrates correct understanding of the current task without hallucinating old details.
