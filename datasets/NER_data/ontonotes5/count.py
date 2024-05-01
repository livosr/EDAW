def count_sentences(file_path):
    # 打开文件并读取所有行
    with open(file_path, 'r', encoding='utf-8') as file:
        lines = file.readlines()

    # 初始化句子计数器
    sentence_count = 0

    # 遍历每一行，检查是否为空行
    for line in lines:
        # 如果找到一个空行，表示一个句子的结束
        if line.strip() == '':
            sentence_count += 1

    return sentence_count


def count_B_entities(file_path):
    # 初始化实体计数器
    entity_count = 0

    # 打开文件并逐行读取
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            # 分割每行为单词和标签
            parts = line.strip().split()
            # 检查行是否非空并且是否包含至少两个部分（单词和标签）
            if len(parts) >= 2:
                # 获取标签部分，通常在单词后
                label = parts[-1]
                # 检查标签是否以 'B-' 开始
                if label.startswith('B-'):
                    entity_count += 1

    return entity_count


# 假设文件路径是 'train.txt'
file_path = 'train.txt'
print(f"文件中的B-实体数量为: {count_B_entities(file_path)}")

# 假设文件名为 'train.txt'
file_path = 'train.txt'
print(f"文件中的句子数量为: {count_sentences(file_path)}")