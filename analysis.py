# 删除所有未使用的库导入
# 仅保留基本结构，替换所有defaultdict为普通字典

def main():
    # 输入处理
    print("请输入参考字符串和查询字符串，用空格分隔：")
    line = input().strip().split()
    if len(line) == 2:
        ref, query = line[0], line[1]
    else:
        ref = line[0]
        query = input().strip()
    
    # 预处理查询字符串的所有子串出现次数（用普通字典替代defaultdict）
    n_query = len(query)
    query_substr_count = {}
    for i in range(n_query):
        for j in range(i+1, n_query+1):
            sub = query[i:j]
            if sub not in query_substr_count:
                query_substr_count[sub] = 0
            query_substr_count[sub] += 1
    
    # 互补配对字典（使用基础数据结构）
    complement = {'A': 'T', 'T': 'A', 'C': 'G', 'G': 'C'}
    
    # 生成互补反转后的ref（仅用字符串操作）
    rev_ref = ''.join([complement[c] for c in ref[::-1]])
    
    # 双哈希参数（仅用基础数学运算）
    BASE1 = 911382629
    BASE2 = 3571428571
    MOD1 = 10**18 + 3
    MOD2 = 10**18 + 7
    
    # 哈希预处理函数（仅用列表和循环）
    def compute_prefix_hash(s, base, mod):
        n = len(s)
        prefix = [0] * (n + 1)
        power = [1] * (n + 1)
        for i in range(n):
            prefix[i+1] = (prefix[i] * base + ord(s[i])) % mod
            power[i+1] = (power[i] * base) % mod
        return prefix, power
    
    # 计算所有哈希（允许的数学运算）
    ref_prefix1, ref_power1 = compute_prefix_hash(ref, BASE1, MOD1)
    ref_prefix2, ref_power2 = compute_prefix_hash(ref, BASE2, MOD2)
    rev_ref_prefix1, rev_ref_power1 = compute_prefix_hash(rev_ref, BASE1, MOD1)
    rev_ref_prefix2, rev_ref_power2 = compute_prefix_hash(rev_ref, BASE2, MOD2)
    query_prefix1, query_power1 = compute_prefix_hash(query, BASE1, MOD1)
    query_prefix2, query_power2 = compute_prefix_hash(query, BASE2, MOD2)
    
    # 用普通字典替代嵌套defaultdict
    ref_hash_info = {}
    m = len(ref)
    for i in range(m):
        h1 = 0
        h2 = 0
        for j in range(i, m):
            c = ord(ref[j])
            h1 = (h1 * BASE1 + c) % MOD1
            h2 = (h2 * BASE2 + c) % MOD2
            key = (h1, h2)
            l = j - i + 1
            if key not in ref_hash_info:
                ref_hash_info[key] = {}
            if l not in ref_hash_info[key]:
                ref_hash_info[key][l] = []
            ref_hash_info[key][l].append(i)
    
    rev_ref_hash_info = {}
    len_rev_ref = len(rev_ref)
    for i in range(len_rev_ref):
        h1 = 0
        h2 = 0
        for j in range(i, len_rev_ref):
            c = ord(rev_ref[j])
            h1 = (h1 * BASE1 + c) % MOD1
            h2 = (h2 * BASE2 + c) % MOD2
            key = (h1, h2)
            l = j - i + 1
            if key not in rev_ref_hash_info:
                rev_ref_hash_info[key] = {}
            if l not in rev_ref_hash_info[key]:
                rev_ref_hash_info[key][l] = []
            rev_ref_hash_info[key][l].append(i)
    
    # 动态规划部分（允许使用min/max/range）
    n = len(query)
    dp = [ (float('inf'), -float('inf')) ] * (n + 1)
    dp[0] = (0, 0)
    parent = [-1] * (n + 1)
    
    for i in range(1, n + 1):
        max_possible = min(i, m)
        best_split = float('inf')
        best_repeat = -float('inf')
        best_j = -1
        for k in range(max_possible, 0, -1):
            start = i - k
            if start < 0:
                continue
            sub = query[start:i]
            # 计算哈希（仅用基础运算）
            h1 = (query_prefix1[i] - query_prefix1[start] * query_power1[k]) % MOD1
            h2 = (query_prefix2[i] - query_prefix2[start] * query_power2[k]) % MOD2
            h = (h1, h2)
            found = False
            
            # 检查普通字典中的存在性
            if h in ref_hash_info:
                if k in ref_hash_info[h]:
                    for pos in ref_hash_info[h][k]:
                        if ref[pos:pos+k] == sub:
                            found = True
                            break
            if not found and h in rev_ref_hash_info:
                if k in rev_ref_hash_info[h]:
                    for pos in rev_ref_hash_info[h][k]:
                        if rev_ref[pos:pos+k] == sub:
                            found = True
                            break
            
            if found:
                prev_split, prev_repeat = dp[start]
                if prev_split == float('inf'):
                    continue
                new_split = prev_split + 1
                cnt = query_substr_count.get(sub, 0)
                new_repeat = prev_repeat + (cnt - 1)
                if (new_split < best_split) or (new_split == best_split and new_repeat > best_repeat):
                    best_split = new_split
                    best_repeat = new_repeat
                    best_j = start
        if best_split != float('inf'):
            dp[i] = (best_split, best_repeat)
            parent[i] = best_j
    
    if dp[n][0] == float('inf'):
        print(-1)
        return
    
    # 输出结果（允许enumerate）
    splits = []
    current = n
    while current > 0:
        prev = parent[current]
        splits.append((prev, current))
        current = prev
    splits.reverse()
    
    # 统计结果（允许字符串切片）
    unique_segments = {}
    for s, e in splits:
        sub = query[s:e]
        if sub not in unique_segments:
            unique_segments[sub] = {
                'count_in_query': 0,
                'source': '',
                'first_pos': -1,
                'length': len(sub),
                'repeat_count_in_ref': 0,
                'direction': ''
            }
        unique_segments[sub]['count_in_query'] += 1
    
    # 统计参考中的出现次数
    for sub in unique_segments:
        found = False
        for i in range(len(ref) - len(sub) + 1):
            if ref[i:i+len(sub)] == sub:
                unique_segments[sub]['source'] = 'R'
                unique_segments[sub]['first_pos'] = i
                cnt = 0
                for j in range(len(ref) - len(sub) + 1):
                    if ref[j:j+len(sub)] == sub:
                        cnt += 1
                unique_segments[sub]['repeat_count_in_ref'] = cnt
                found = True
                break
        if not found:
            rc_sub = ''.join([complement[c] for c in sub[::-1]])
            for i in range(len(ref) - len(sub) + 1):
                if ref[i:i+len(sub)] == rc_sub:
                    unique_segments[sub]['source'] = 'CR'
                    unique_segments[sub]['first_pos'] = i
                    cnt = 0
                    for j in range(len(ref) - len(sub) + 1):
                        if ref[j:j+len(sub)] == rc_sub:
                            cnt += 1
                    unique_segments[sub]['repeat_count_in_ref'] = cnt
                    found = True
                    break
    
    # 最终输出
    print(f"最少分割段数：{len(splits)}")
    total_repeats = sum( (v['count_in_query'] -1) for v in unique_segments.values() )
    print(f"总重复次数：{total_repeats}")
    for sub, info in unique_segments.items():
        if info['count_in_query'] > 1:
            print(f"片段：{sub}")
            print(f"来源：{info['source']}")
            print(f"第一次出现位置：{info['first_pos'] + info['length']}")
            print(f"长度：{info['length']}")
            print(f"在查询中的出现次数：{info['count_in_query']}")
            print(f"方向：{'正向' if info['source'] == 'R' else '反向'}")
            print("-" * 20)

if __name__ == "__main__":
    main()