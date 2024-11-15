
file = open("julius_caesar_tiny.tsv","r")

Lines = file.readlines()

for line in Lines:
    content = line.split('\t')
    print(content)
