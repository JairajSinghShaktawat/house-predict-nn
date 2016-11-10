theFile = open('thefile.txt','r')
FILE = theFile.readlines()
theFile.close()
printList = []
for line in FILE:
    if ('TestName' in line) or ('Totals' in line):
         # here you may want to do some splitting/concatenation/formatting to your string
         printList.append(line)

for item in printList:
    print item    # or write it to another file... or whatever