import os
import json

def update_json_file(file_path, key, new_value):
    try:
        with open(file_path, 'r') as file:
            data = json.load(file)
        
        if key in data['training']:
            data['training'][key] = new_value
            with open(file_path, 'w') as file:
                json.dump(data, file, indent=4)
            return True
        else:
            print(f"'{key}' not found in 'training' section of {file_path}")
            return False

    except Exception as e:
        print(f"Error processing file {file_path}: {e}")
        return False

def main(base_dir, start_iter=0, end_iter=15, sub_path="00.train/000/input.json"):
    for i in range(start_iter, end_iter + 1):
        iter_dir = f"iter.{i:06d}"
        json_path = os.path.join(base_dir, iter_dir, sub_path)

        if os.path.exists(json_path):
            if update_json_file(json_path, 'stop_batch', 100000):
                print(f"Updated {json_path}")
            else:
                print(f"Failed to update {json_path}")
        else:
            print(f"File not found: {json_path}")

# 使用函数
base_directory = "."  # 更改为你的目录路径
main(base_directory)


