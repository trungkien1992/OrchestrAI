# Digest Command

## Purpose
Create differential summaries from Claude Code sessions and maintain a RAG (Retrieval-Augmented Generation) knowledge base for context continuity across sessions using the existing Claude RAG system.

## Usage
```bash
# Create digest from current session
/digest --create

# Query existing knowledge base using RAG
/digest --query "keyword or phrase"

# Update with new task completion or current session status
/digest --update

# Show current RAG summary
/digest --show

# Auto-snapshot at session end (automated)
/digest --snapshot

# Update digest command implementation
/digest --update-command
```

## Implementation

### Core Workflow
1. **Session Analysis**: Extract key changes and implementations
2. **RAG Integration**: Use existing Claude RAG at `/Users/admin/AstraTrade-Project/claude-rag/`
3. **Auto-Cleanup**: Replace previous snapshots with current session data
4. **Context Continuity**: Maintain project knowledge across sessions

### Command Actions

#### `/digest --create`
- Analyze current session using TodoRead
- Generate structured summary of implementations
- Save to `claude-rag/digest_resource/session_current.md`
- Index with RAG system via API

#### `/digest --query "term"`
- Query RAG API: `curl -X POST "http://127.0.0.1:8000/search"`
- Return relevant context from indexed sessions
- Format with file references and relevance scores

#### `/digest --update`
- Update session summary with current task status
- Extract completed/pending tasks from TodoRead
- Add timestamp and completion notes
- Re-index with RAG system

#### `/digest --show`
- Display current session summary
- Show RAG system status
- List pending tasks and next steps

#### `/digest --snapshot`
- Create final session summary
- Archive current as previous
- Update RAG index with all changes
- Clean up old detailed diffs

#### `/digest --update-command`
- Read current command implementation
- Allow modifications to digest.md
- Update workflow and functionality
- Re-save command file

### File Structure
```
claude-rag/
├── digest_resource/
│   ├── session_current.md           # Current session details
│   ├── session_previous.md          # Previous session summary
│   └── codebase_snapshot.md         # Current codebase state
└── data/claude_db/                  # ChromaDB storage
```

### RAG Integration
- **Indexing**: All digest content indexed via existing RAG API
- **Queries**: Natural language search across session history
- **Cleanup**: Rolling window - detailed current + compressed previous
- **Context**: Semantic search for implementation patterns

### Update Command Functionality
When `/digest --update-command` is invoked:
1. Read current `/Users/admin/.claude/commands/digest.md`
2. Allow Claude to modify command implementation
3. Update workflow, add features, fix issues
4. Save updated command file
5. Confirm changes and new capabilities

This enables iterative improvement of the digest command based on usage patterns and user needs.

## Benefits
- **Session Continuity**: Context across Claude sessions
- **Automated Snapshots**: End-of-session codebase state
- **Intelligent Queries**: Natural language project search
- **Self-Updating**: Command can evolve with requirements

## Implementation Details

### RAG System Configuration
- **API Endpoint**: `http://127.0.0.1:8001` (Claude RAG API)
- **Collection**: `claude_db` with existing ChromaDB
- **Search Method**: POST `/search` with semantic similarity
- **Index Method**: POST `/index` for project updates

### Session Analysis Process
1. **TodoRead**: Extract current task list and completion status
2. **Git Status**: Capture file changes and modifications
3. **File Analysis**: Identify key implementations and patterns
4. **Context Generation**: Create structured markdown summaries
5. **RAG Indexing**: Store in ChromaDB for future retrieval

### Digest Content Structure
```markdown
# Session [Timestamp] - [Topic]

## Overview
Brief description of session goals and outcomes

## Major Implementations
### 1. [Implementation Name]
**Files**: List of modified files
**Changes**: Detailed change descriptions
**Key Features**: Important functionality added

## Technical Decisions
- Architecture choices
- Design patterns used
- Performance considerations

## Next Steps
- Remaining tasks
- Future enhancements
- Dependencies to resolve

## Code Quality Notes
- Patterns followed
- Testing considerations
- Documentation status
```

### Error Handling
- **RAG Unavailable**: Fallback to local file storage
- **API Errors**: Retry with exponential backoff
- **Invalid Paths**: Clear error messages with suggestions
- **Malformed Queries**: Query validation and formatting

### Performance Optimization
- **Incremental Updates**: Only re-index changed content
- **Batch Operations**: Group multiple updates efficiently
- **Semantic Caching**: Avoid redundant similarity searches
- **Cleanup Automation**: Regular maintenance of old digests

This command provides seamless session continuity for Claude Code users while maintaining an intelligent knowledge base of project history and implementation patterns.