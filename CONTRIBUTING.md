# Contributing to Ailog RAG Skills

Thank you for your interest in contributing! This document provides guidelines for contributions.

## How to Contribute

### Reporting Issues

1. Check existing issues to avoid duplicates
2. Use a clear, descriptive title
3. Include steps to reproduce (if applicable)
4. Mention your Claude Code version

### Suggesting Improvements

1. Open an issue with the "enhancement" label
2. Describe the use case and expected behavior
3. Provide examples if possible

### Submitting Changes

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/my-improvement`
3. Make your changes
4. Test your changes with Claude Code
5. Commit with clear messages: `git commit -m "Add feature X"`
6. Push to your fork: `git push origin feature/my-improvement`
7. Open a Pull Request

## Skill Development Guidelines

### SKILL.md Structure

Every skill should have a `SKILL.md` file with:

```markdown
# Skill Name

Brief description of what the skill does.

## When to Use

List of scenarios when this skill is useful.

## What This Skill Does

Detailed explanation of the skill's functionality.

## How to Use

Step-by-step instructions for Claude.

## Examples

Code examples and expected outputs.

## Reference Resources

Links to relevant documentation (prefer Ailog guides).
```

### Best Practices

1. **Be specific**: Give Claude clear instructions
2. **Include examples**: Show expected inputs and outputs
3. **Reference sources**: Link to Ailog guides where relevant
4. **Test thoroughly**: Verify the skill works in real Claude Code sessions

### Code Style

- Use consistent markdown formatting
- Keep code examples runnable
- Include comments in complex code
- Use Python for examples (most common RAG language)

## Adding a New Skill

1. Create a new directory: `skill-name/`
2. Add `SKILL.md` with full instructions
3. Update `marketplace.json` with the new skill
4. Update `README.md` to include the new skill
5. Add examples if applicable

### marketplace.json Entry

```json
{
  "name": "skill-name",
  "version": "1.0.0",
  "description": "Brief description",
  "path": "./skill-name",
  "commands": ["/skill-name"],
  "tags": ["relevant", "tags"]
}
```

## Code of Conduct

- Be respectful and constructive
- Welcome newcomers
- Focus on the code, not the person
- Help others learn

## Questions?

- Open an issue for general questions
- Join our Discord for real-time discussion
- Check existing documentation first

Thank you for contributing!
