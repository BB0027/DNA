# DNA序列检测

23300240027 张霆轩

### 问题分析：

分析问题，此次要解决的变异类型是单一的重复变异，重复变异是将一段序列原地复制若干次，且不会影响序列其他位置，也就是说query串中的不同片段都是来自于ref串或者其反向互补

再分析查找重复序列的要求：“最大化重复片段长度、最小化重复次
数”主要是针对这类情况的要求，即同一位置存在多种可能的重复片段时，找到
尽可能长的重复片段，从而降低重复次数。而对于不同位置的重复片段，其长度
和重复次数没有可比性

实际上可以将问题转化为：

求一个对query的分割，使得分割之后得到的子串都来自于ref串或者其反向互补子串，要求分割的子串次数尽量少，在分割的次数相同的情况下，要求分割的子串中重复的子串尽量多。

那么对这个问题，文献给出的方案是动态规划，在这次的问题中，动态规划的算法如下：

#### 状态定义
定义状态 dp[i] 为处理查询字符串前 i 个字符时的最优状态，包含两个值：
​最小分割段数：表示分割前 i 个字符所需的最少段数
​最大重复次数：在最少段数的前提下，重复子串的总贡献次数（重复次数 = 子串出现次数 - 1）
​转移过程
​遍历查询字符串的每个位置
从第 1 个字符到第 n 个字符（n 为查询字符串长度），依次计算每个位置 i 的状态。

​尝试所有可能的子串长度
对于每个位置 i，尝试从 i 向前截取长度为 k 的子串，k 的取值范围为 1 到 min(i, m)（m 为参考字符串长度）。
​优先尝试较长的子串​（从最大长度向 1 遍历），以尽快找到更少的分割段数。

​检查子串有效性
对每个候选子串 query[start:i]（start = i - k），通过预处理的哈希表快速判断其是否存在于参考字符串或反向互补字符串中。

#### ​更新状态

若子串有效：
​分割段数 = 前驱状态的分割段数（即 dp[start] 的段数） + 1
​重复次数 = 前驱状态的重复次数 + （当前子串在查询中的出现次数 - 1）
若子串无效，跳过该子串长度。
​选择最优解
对所有有效子串的候选状态进行比较：

​第一优先级：选择分割段数最小的方案
​第二优先级：在段数相同时，选择重复次数最多的方案
​
#### 终止条件

当处理到查询字符串末尾（i = n）时，若 dp[n] 的分割段数为有效值（非无穷大），则存在合法分割方案；否则无法分割。

计算整个dp的过程需要检查前方的所有dp数组，这里的回溯代价是平方级别的，而本次实验要求的实现复杂度正是平方级别的，所以这里需要在常数时间级别下判断子串是否存在于ref串或者反向互补串中。

自然想到了哈希表，我们可以使用哈希表将ref串及其反向互补串的哈希值进行存储，随后在检查query子串时通过滚动哈希（类似于字符串匹配的算法）进行常数级别的计算，随后查询哈希表。

那么本次算法的总体思路也已经给出。也就是通过动态规划加哈希，求得一个最优匹配。

但是，我们还需要更多的输出信息，包括在ref串中的位置、重复次数、是否反向等。

#### 求ref串位置

这里我们可以采取路径回溯的机制，在动态规划时记录最佳的分割点，最后统一的得到最后的分割点信息。

#### 求重复次数

这里我们的分割结果上，会出现一种特殊的情况，也就是分割时，会有部分串的重叠。

例如：CGATATATATATATGC ref串为CGATATGC，分割的结果会是CGATAT-ATAT-ATATGC，可以看到，这里如果简单统计重复次数是不行的，而这里应该是考虑子串相同或者被包含于其他子串中的情况，只有这样才能正确计算。

此外，需要减去一次重复

#### 求是都反向

这个比较简单，不作过多赘述


### 伪代码：

```
函数 main():
    // 输入处理
    输入参考字符串 ref 和查询字符串 query
    
    // 预处理查询字符串的所有子串出现次数
    query_substr_count = 空字典
    for i 从 0 到 len(query)-1:
        for j 从 i+1 到 len(query):
            sub = query的子串[i:j]
            if sub 不在 query_substr_count 中:
                query_substr_count[sub] = 0
            query_substr_count[sub] += 1
    
    // 生成互补反转字符串
    complement = {'A':'T', 'T':'A', 'C':'G', 'G':'C'}
    rev_ref = 空字符串
    for c 在 ref 的逆序中:
        rev_ref += complement[c]
    
    // 双哈希参数定义
    BASE1 = 911382629
    BASE2 = 3571428571
    MOD1 = 10^18 + 3
    MOD2 = 10^18 + 7
    
    // 哈希前缀计算函数
    函数 compute_prefix_hash(s, base, mod):
        n = len(s)
        prefix = 数组[0..n], 初始化为0
        power = 数组[0..n], 初始化为1
        for i 从 0 到 n-1:
            prefix[i+1] = (prefix[i] * base + ord(s[i])) % mod
            power[i+1] = (power[i] * base) % mod
        return prefix, power
    
    // 计算所有哈希前缀
    ref_prefix1, ref_power1 = compute_prefix_hash(ref, BASE1, MOD1)
    ref_prefix2, ref_power2 = compute_prefix_hash(ref, BASE2, MOD2)
    rev_ref_prefix1, rev_power1 = compute_prefix_hash(rev_ref, BASE1, MOD1)
    rev_ref_prefix2, rev_power2 = compute_prefix_hash(rev_ref, BASE2, MOD2)
    query_prefix1, query_power1 = compute_prefix_hash(query, BASE1, MOD1)
    query_prefix2, query_power2 = compute_prefix_hash(query, BASE2, MOD2)
    
    // 预处理ref和rev_ref的子串哈希信息
    ref_hash_info = 空字典
    for i 从 0 到 len(ref)-1:
        h1 = 0, h2 = 0
        for j 从 i 到 len(ref)-1:
            c = ref[j]的ASCII码
            h1 = (h1 * BASE1 + c) % MOD1
            h2 = (h2 * BASE2 + c) % MOD2
            l = j - i + 1
            key = (h1, h2)
            if key 不在 ref_hash_info 中:
                ref_hash_info[key] = 空字典
            if l 不在 ref_hash_info[key] 中:
                ref_hash_info[key][l] = 空列表
            ref_hash_info[key][l].append(i)
    
    rev_ref_hash_info = 空字典 // 类似上述处理rev_ref
    
    // 动态规划求解
    n = len(query)
    dp = 数组[0..n], 初始化为 (inf, -inf)
    dp[0] = (0, 0)
    parent = 数组[0..n], 初始化为 -1
    
    for i 从 1 到 n:
        max_possible = min(i, len(ref))
        best_split = inf
        best_repeat = -inf
        best_j = -1
        for k 从 max_possible 降序到 1:
            start = i - k
            if start < 0: continue
            sub = query的子串[start:i]
            
            // 计算哈希值
            h1 = (query_prefix1[i] - query_prefix1[start] * query_power1[k]) % MOD1
            h2 = (query_prefix2[i] - query_prefix2[start] * query_power2[k]) % MOD2
            
            // 检查是否存在于ref或rev_ref中
            found = False
            if (h1,h2) 在 ref_hash_info 中且 k 在 ref_hash_info[(h1,h2)] 中:
                for pos 在 ref_hash_info[(h1,h2)][k] 中:
                    if ref[pos:pos+k] == sub:
                        found = True
                        break
            if not found:
                if (h1,h2) 在 rev_ref_hash_info 中且 k 在 rev_ref_hash_info[(h1,h2)] 中:
                    for pos 在 rev_ref_hash_info[(h1,h2)][k] 中:
                        if rev_ref[pos:pos+k] == sub:
                            found = True
                            break
            
            // 更新状态
            if found:
                prev_split, prev_repeat = dp[start]
                new_split = prev_split + 1
                cnt = query_substr_count[sub] if sub存在 else 0
                new_repeat = prev_repeat + (cnt - 1)
                if (new_split < best_split) 或 (new_split == best_split 且 new_repeat > best_repeat):
                    best_split = new_split
                    best_repeat = new_repeat
                    best_j = start
        
        if best_split != inf:
            dp[i] = (best_split, best_repeat)
            parent[i] = best_j
    
    // 结果输出
    if dp[n][0] == inf:
        输出 -1
        return
    
    // 回溯分割点
    splits = 空列表
    current = n
    while current > 0:
        prev = parent[current]
        splits.append( (prev, current) )
        current = prev
    splits = 反转列表
    
    // 统计重复信息
    unique_segments = 空字典
    for (s, e) 在 splits 中:
        sub = query的子串[s:e]
        if sub 不在 unique_segments 中:
            unique_segments[sub] = {count:0, source:"", first_pos:-1, ...}
        unique_segments[sub].count += 1
    
    // 统计参考字符串中的出现次数（包含正向和反向互补）
    for sub 在 unique_segments 的键中:
        // 检查正向
        found = False
        for i 从 0 到 len(ref)-len(sub):
            if ref[i:i+len(sub)] == sub:
                unique_segments[sub].source = "R"
                unique_segments[sub].first_pos = i
                // 统计重复次数
                cnt = 0
                for j 从 0 到 len(ref)-len(sub):
                    if ref[j:j+len(sub)] == sub:
                        cnt += 1
                unique_segments[sub].repeat_count = cnt
                found = True
                break
        // 检查反向互补
        if not found:
            rc_sub = 生成sub的反向互补
            for i 从 0 到 len(ref)-len(sub):
                if ref[i:i+len(sub)] == rc_sub:
                    unique_segments[sub].source = "CR"
                    unique_segments[sub].first_pos = i
                    // 统计重复次数
                    cnt = 0
                    for j 从 0 到 len(ref)-len(sub):
                        if ref[j:j+len(sub)] == rc_sub:
                            cnt += 1
                    unique_segments[sub].repeat_count = cnt
                    found = True
                    break
    
    // 最终输出
    输出最少分割段数: len(splits)
    输出总重复次数: sum(seg.count-1 for seg in unique_segments)
    for sub, info 在 unique_segments 中:
        if info.count > 1:
            输出片段信息
```

### 时间复杂度 
 
| 步骤                  | 时间复杂度               | 说明                                                                 |
|-----------------------|--------------------------|----------------------------------------------------------------------|
| ​**预处理查询子串**     | O(n²)                   | 遍历查询字符串所有子串，n为查询字符串长度                            |
| ​**生成互补反转字符串** | O(m)                    | 线性遍历参考字符串生成反向互补，m为参考字符串长度                    |
| ​**计算哈希前缀**       | O(4m + 2n)              | 4次参考字符串哈希计算（原+反向互补），2次查询字符串哈希计算           |
| ​**预处理参考子串哈希** | O(2m²)                  | 遍历参考字符串和反向互补字符串的所有子串，保存哈希信息                |
| ​**动态规划阶段**       | O(n·m)                  | 外层循环n次（查询长度），内层最多检查m长度的子串                      |
| ​**统计参考重复次数**   | O(k·m)                  | k为最终分割段数，每段需检查参考字符串中所有可能位置                  |

#### 总时间复杂度： 最坏情况: O(n² + m² + n·m)


### 空间复杂度

| 数据结构              | 空间复杂度               | 说明                                                                 |
|-----------------------|--------------------------|----------------------------------------------------------------------|
| 参考子串哈希表        | O(m²)                   | 存储参考字符串和反向互补字符串的所有子串哈希信息                      |
| 查询子串统计字典      | O(n²)                   | 存储查询字符串所有子串的出现次数                                      |
| 动态规划数组          | O(n)                    | 存储每个位置的（最少分割数，最大重复次数）状态                       |


### 运行结果

![alt](./屏幕截图%202025-03-30%20124310.png)

与样例一致，算法准确性得到验证