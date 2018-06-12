with open('res.md', 'r') as f:
    lines = f.readlines()

with open('res.tex', 'w') as f:
    for line in lines:
        words = line.split('|')[1:-1]
        new_line = '&'.join(words)
        new_line += '\\\\\n'
        f.write(new_line)
