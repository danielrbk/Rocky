with open("C:\\Users\\rejabek\\Downloads\\output.csv") as f, open("C:\\Users\\rejabek\\Downloads\\realOutput.csv", 'w') as out:
    lines = f.readlines()
    for i in range(len(lines)):
        lines[i] = lines[i].replace("\"","")
    out.writelines(lines)
