import sys

fileName = sys.argv[1]
out1 = sys.argv[2]
out2 = sys.argv[3]


with open(fileName, "r") as f, open(out1, "w") as o1, open(out2, "w") as o2:
    dimensions = f.readline().split(' ')
    lines = int(dimensions[0])
    currentLine = 0
    file1 = True
    o1.writelines(f"{lines//2} {dimensions[1]} {dimensions[2]}")
    o2.writelines(f"{lines - lines//2} {dimensions[1]} {dimensions[2]}")
    for line in f.readlines():
        currentLine += 1
        if(file1):
            o1.writelines(line)
        else:
            o2.writelines(line)
        if(currentLine == lines // 2):
            file1 = False

# with open(fileName, "r") as f:
#     dimensions = f.readline().split(' ')
#     currentLine = 0
#     dimensions1 = dimensions
#     dimensions2 = dimensions
#     dimensions1[0] = str(int(dimensions1[0]) // 2)
#     dimensions2[0] = str(int(dimensions2[0]) - (int(dimensions1[0]) // 2))
#     with open(out1, "w") as o1:
#         o1.writelines(" ".join(dimensions1))
#         for line in f.readlines():
#             o1.writelines(line)
#             currentLine += 1
#             if(currentLine == (lines // 2)):
#                 break
#     with open(out2, "w") as o2:
#         o2.writelines(" ".join(dimensions2))
#         for line in f.readlines():
#             o2.writelines(line)