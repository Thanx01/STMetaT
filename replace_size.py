
import re
import sys

def replace_size(file_path, support_size, query_size):
    with open(file_path, 'r') as f:
        content = f.read()

    content = re.sub(r'support_size=\d+', f'support_size={support_size}', content)
    content = re.sub(r'query_size=\d+', f'query_size={query_size}', content)

    with open(file_path, 'w') as f:
        f.write(content)

if __name__ == '__main__':
    if len(sys.argv) == 3:
        file_path = '/home/jsj201-11/mount1/xzz/基线/PTrajM-C973_meta/MetaLearningTrajClip.py'
        support_size = int(sys.argv[1])
        query_size = int(sys.argv[2])
    elif len(sys.argv) == 4:
        file_path = sys.argv[1]
        support_size = int(sys.argv[2])
        query_size = int(sys.argv[3])
    else:
        print("Usage: python script.py [file_path] <support_size> <query_size>")
        sys.exit(1)

    replace_size(file_path, support_size, query_size)
    print(f"Successfully replaced support_size and query_size in {file_path}")