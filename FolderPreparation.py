from os import listdir,rename
from os.path import join, isfile
from ConstantDefinitions import CLASS_MAP_PATH, VISIT_PATH

def prepare_folders(folder, max_nest, nested_level=0, new_suffix=".apk"):
    dirs = [join(folder, d) for d in listdir(folder) if not isfile(join(folder, d))]
    if nested_level == max_nest:
        for d in dirs:
            rename(d, d + new_suffix)
    else:
        if nested_level == max_nest-1:
            print folder
        for d in dirs:
            prepare_folders(d, max_nest, nested_level + 1)

def map_file(folder):
    dirs = [d for d in listdir(folder) if d!="Results"]
    s=""
    with open(CLASS_MAP_PATH, 'w') as f:
        for i in range(len(dirs)):
            s += "%s;%s\r\n" % (str(i),dirs[i])
        f.write(s)


map_file(VISIT_PATH)
