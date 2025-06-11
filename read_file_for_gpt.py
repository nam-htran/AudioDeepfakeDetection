import os

def save_all_source_code(output_path='output.txt', excluded_dirs=None, excluded_files=None):
    if excluded_dirs is None:
        excluded_dirs = {
        }
    if excluded_files is None:
        excluded_files = {
        }

    file_count = 0
    with open(output_path, 'w', encoding='utf-8') as outfile:
        for root, dirs, files in os.walk('.'):
            # Loại trừ thư mục
            dirs[:] = [d for d in dirs if d not in excluded_dirs]

            for file in files:
                if file.endswith('.py') and file not in excluded_files:
                    filepath = os.path.join(root, file)
                    try:
                        with open(filepath, 'r', encoding='utf-8') as f:
                            code = f.read()
                        outfile.write(f"\n===== {filepath} =====\n")
                        outfile.write(code + "\n")
                        file_count += 1
                    except Exception as e:
                        print(f"Lỗi đọc {filepath}: {e}")

        # Ghi phần cây thư mục cuối file
        outfile.write("\n===== DIRECTORY TREE =====\n")
        for dirpath, dirnames, filenames in os.walk('.'):
            # Bỏ qua thư mục bị loại
            dirnames[:] = [d for d in dirnames if d not in excluded_dirs]
            level = dirpath.count(os.sep)
            indent = '    ' * level
            outfile.write(f"{indent}{os.path.basename(dirpath)}/\n")
            for file in filenames:
                if file not in excluded_files:
                    subindent = '    ' * (level + 1)
                    outfile.write(f"{subindent}{file}\n")

    print(f"✅ Đã lưu {file_count} file Python và cây thư mục vào '{output_path}'.")

if __name__ == '__main__':
    save_all_source_code()