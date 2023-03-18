import os

for name in ['training', 'testing']:
    csvpath = 'mnist/dataset_{}.csv'.format(name)
    with open(csvpath, "w") as csvfile:
        print("Creez {} dataset".format(name))
        csvfile.write("Label, Path\n")
        for i in range(10):
            path = "./mnist/{}/{}".format(name, i)
            path = path.replace("\\", "/")
            print(path)
            for file in os.listdir(path):
                if file.endswith(".jpg"):
                    print()
                    csvfile.write("{},{}".format(int(i), "{}/{}\n".format(path, file)))