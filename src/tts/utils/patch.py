import re
import uuid

def apply_patch(env, patch, cwd):

    delimiter = (f"PATCH_{uuid.uuid4().hex}")
    patch_string = f"git apply --verbose <<'{delimiter}'\n{patch}\n{delimiter}"

    command_list = [
        patch_string,
        'git config user.email "sweft@anon.com"',
        'git config user.name "sweft"',
        "git commit -am 'Initial commit'",
        "git checkout --orphan new-main",
        "git add .",
        "git commit -m 'Initial commit'",
        "git branch -D main",
        "git branch -m main",
    ]

    for command in command_list:
        _ = env.execute({"command": command}, cwd=cwd)["output"]

    return env

def reverse_diff(diff_text: str) -> str:
    """
    Reverse a git diff, swapping additions and deletions.
    
    Args:
        diff_text: The original diff as a string (can contain escaped newlines)
    
    Returns:
        The reversed diff as a string
    """
    # Handle escaped newlines if present
    if '\\n' in diff_text and '\n' not in diff_text.strip('\'\"'):
        diff_text = diff_text.strip('\'\"').encode().decode('unicode_escape')
    
    lines = diff_text.split('\n')
    result = []
    i = 0
    
    while i < len(lines):
        line = lines[i]
        
        # Swap index hashes: index abc123..def456 -> index def456..abc123
        if line.startswith('index '):
            match = re.match(r'(index )([a-f0-9]+)\.\.([a-f0-9]+)(.*)', line)
            if match:
                line = f"{match.group(1)}{match.group(3)}..{match.group(2)}{match.group(4)}"
            result.append(line)
            i += 1
        
        # Swap hunk header and reorder hunk content
        elif line.startswith('@@'):
            match = re.match(r'@@ -(\d+),?(\d*) \+(\d+),?(\d*) @@(.*)', line)
            if match:
                old_start, old_count, new_start, new_count, rest = match.groups()
                # Build the reversed hunk header
                old_part = f"-{new_start}" + (f",{new_count}" if new_count else "")
                new_part = f"+{old_start}" + (f",{old_count}" if old_count else "")
                result.append(f"@@ {old_part} {new_part} @@{rest}")
            else:
                result.append(line)
            i += 1
            
            # Collect hunk content until next hunk or end
            hunk_lines = []
            while i < len(lines) and not lines[i].startswith('@@') and not lines[i].startswith('diff '):
                hunk_lines.append(lines[i])
                i += 1
            
            # Reverse the hunk with proper ordering
            result.extend(reverse_hunk(hunk_lines))
        
        else:
            result.append(line)
            i += 1
    
    return '\n'.join(result)


def reverse_hunk(hunk_lines: list) -> list:
    """
    Reverse hunk lines with proper ordering for unified diff format.
    
    In a unified diff, change groups (consecutive -/+ lines) need to be 
    reversed as a unit: the + lines become - lines and vice versa,
    but the + lines (now -) must come before the - lines (now +).
    """
    result = []
    i = 0
    
    while i < len(hunk_lines):
        line = hunk_lines[i]
        
        # Context line - keep as is
        if line.startswith(' '):
            result.append(line)
            i += 1
        
        # Start of a change group
        elif line.startswith('-') or line.startswith('+'):
            minus_lines = []
            plus_lines = []
            no_newline_after_minus = False
            no_newline_after_plus = False
            
            # Collect consecutive minus lines
            while i < len(hunk_lines) and hunk_lines[i].startswith('-'):
                minus_lines.append(hunk_lines[i][1:])  # Strip the '-'
                i += 1
                # Check for "no newline" marker after minus
                if i < len(hunk_lines) and hunk_lines[i].startswith('\\'):
                    no_newline_after_minus = True
                    i += 1
            
            # Collect consecutive plus lines
            while i < len(hunk_lines) and hunk_lines[i].startswith('+'):
                plus_lines.append(hunk_lines[i][1:])  # Strip the '+'
                i += 1
                # Check for "no newline" marker after plus
                if i < len(hunk_lines) and hunk_lines[i].startswith('\\'):
                    no_newline_after_plus = True
                    i += 1
            
            # Output reversed: old '+' become '-', old '-' become '+'
            # The new '-' lines (old '+') come first
            for pl in plus_lines:
                result.append('-' + pl)
            if no_newline_after_plus:
                result.append('\\ No newline at end of file')
            
            for ml in minus_lines:
                result.append('+' + ml)
            if no_newline_after_minus:
                result.append('\\ No newline at end of file')
        
        # "No newline" marker outside a change group
        elif line.startswith('\\'):
            result.append(line)
            i += 1
        
        # Empty or other lines - keep as is
        else:
            result.append(line)
            i += 1
    
    return result