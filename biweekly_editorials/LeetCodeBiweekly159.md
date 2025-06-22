## 1. Minimum Adjacent Swaps to Alternate Parity

<details>
<summary>Python</summary>

```python
def min_swaps_to_alternate_parity(nums):
    even_idx = [i for i, x in enumerate(nums) if x % 2 == 0]
    odd_idx = [i for i, x in enumerate(nums) if x % 2 == 1]

    if abs(len(even_idx) - len(odd_idx)) > 1:
        return -1

    def count_swaps(start_with_even):
        target = list(range(0, len(nums), 2))  # positions: 0, 2, 4, ...
        if start_with_even:
            if len(even_idx) < len(target):
                return float('inf')
            return sum(abs(even_idx[i] - target[i]) for i in range(len(target)))
        else:
            if len(odd_idx) < len(target):
                return float('inf')
            return sum(abs(odd_idx[i] - target[i]) for i in range(len(target)))

    res = float('inf')
    if len(even_idx) >= len(odd_idx):
        res = min(res, count_swaps(True))
    if len(odd_idx) >= len(even_idx):
        res = min(res, count_swaps(False))

    return res if res != float('inf') else -1

```

</details>

<details>
<summary>Cpp</summary>

```cpp
#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>
using namespace std;

int minSwapsToAlternateParity(vector<int>& nums) {
    vector<int> evenIdx, oddIdx;
    for (int i = 0; i < nums.size(); ++i) {
        if (nums[i] % 2 == 0) evenIdx.push_back(i);
        else oddIdx.push_back(i);
    }

    if (abs((int)evenIdx.size() - (int)oddIdx.size()) > 1) return -1;

    auto countSwaps = [](vector<int>& idx, int start) {
        int swaps = 0;
        for (int i = 0; i < idx.size(); ++i)
            swaps += abs(idx[i] - (start + 2 * i));
        return swaps;
    };

    int res = INT_MAX;
    if (evenIdx.size() >= oddIdx.size())
        res = min(res, countSwaps(evenIdx, 0));
    if (oddIdx.size() >= evenIdx.size())
        res = min(res, countSwaps(oddIdx, 0));

    return res;
}

```

</details>

<details>
<summary>Java</summary>

```java
import java.util.*;

public class MinSwapsParity {
    public int minSwaps(int[] nums) {
        List<Integer> evenIdx = new ArrayList<>();
        List<Integer> oddIdx = new ArrayList<>();

        for (int i = 0; i < nums.length; i++) {
            if (nums[i] % 2 == 0) evenIdx.add(i);
            else oddIdx.add(i);
        }

        if (Math.abs(evenIdx.size() - oddIdx.size()) > 1)
            return -1;

        int res = Integer.MAX_VALUE;
        if (evenIdx.size() >= oddIdx.size())
            res = Math.min(res, countSwaps(evenIdx, 0));
        if (oddIdx.size() >= evenIdx.size())
            res = Math.min(res, countSwaps(oddIdx, 0));

        return res;
    }

    private int countSwaps(List<Integer> idx, int start) {
        int swaps = 0;
        for (int i = 0; i < idx.size(); i++) {
            swaps += Math.abs(idx.get(i) - (start + 2 * i));
        }
        return swaps;
    }

    public static void main(String[] args) {
        MinSwapsParity msp = new MinSwapsParity();
        int[] nums = {3, 2, 1, 4};
        System.out.println(msp.minSwaps(nums)); // Output example
    }
}

```

</details>

## 2. Find Maximum Area of a Triangle

<details>
<summary>Python</summary>

```python
from collections import defaultdict

def max_area_twice(coords):
    n = len(coords)
    if n < 3:
        return -1

    x_map = defaultdict(list)
    y_map = defaultdict(list)
    point_set = set((x, y) for x, y in coords)

    for x, y in coords:
        x_map[x].append(y)
        y_map[y].append(x)

    max_area2 = 0

    # Vertical sides (same x)
    for x in x_map:
        y_list = x_map[x]
        y_list.sort()
        for i in range(len(y_list)):
            for j in range(i + 1, len(y_list)):
                y1, y2 = y_list[i], y_list[j]
                height = abs(y2 - y1)
                for px, py in coords:
                    if px == x: continue  # skip same vertical line
                    if py == y1 or py == y2:
                        base = abs(px - x)
                        max_area2 = max(max_area2, base * height)

    # Horizontal sides (same y)
    for y in y_map:
        x_list = y_map[y]
        x_list.sort()
        for i in range(len(x_list)):
            for j in range(i + 1, len(x_list)):
                x1, x2 = x_list[i], x_list[j]
                base = abs(x2 - x1)
                for px, py in coords:
                    if py == y: continue  # skip same horizontal line
                    if px == x1 or px == x2:
                        height = abs(py - y)
                        max_area2 = max(max_area2, base * height)

    return max_area2 if max_area2 > 0 else -1
```

</details>

<details>
<summary>Cpp</summary>

```cpp
#include <iostream>
#include <vector>
#include <unordered_map>
#include <set>
#include <algorithm>
using namespace std;

int maxAreaTwice(vector<vector<int>>& coords) {
    unordered_map<int, vector<int>> x_map, y_map;
    set<pair<int, int>> point_set;
    int n = coords.size();

    for (auto& p : coords) {
        int x = p[0], y = p[1];
        x_map[x].push_back(y);
        y_map[y].push_back(x);
        point_set.insert({x, y});
    }

    int max_area2 = 0;

    // Vertical sides (same x)
    for (auto& [x, y_list] : x_map) {
        sort(y_list.begin(), y_list.end());
        for (int i = 0; i < y_list.size(); ++i) {
            for (int j = i + 1; j < y_list.size(); ++j) {
                int y1 = y_list[i], y2 = y_list[j];
                int height = abs(y2 - y1);
                for (auto& p : coords) {
                    int px = p[0], py = p[1];
                    if (px == x) continue;
                    if (py == y1 || py == y2) {
                        int base = abs(px - x);
                        max_area2 = max(max_area2, base * height);
                    }
                }
            }
        }
    }

    // Horizontal sides (same y)
    for (auto& [y, x_list] : y_map) {
        sort(x_list.begin(), x_list.end());
        for (int i = 0; i < x_list.size(); ++i) {
            for (int j = i + 1; j < x_list.size(); ++j) {
                int x1 = x_list[i], x2 = x_list[j];
                int base = abs(x2 - x1);
                for (auto& p : coords) {
                    int px = p[0], py = p[1];
                    if (py == y) continue;
                    if (px == x1 || px == x2) {
                        int height = abs(py - y);
                        max_area2 = max(max_area2, base * height);
                    }
                }
            }
        }
    }

    return max_area2 > 0 ? max_area2 : -1;
}
```

</details>

<details>
<summary>Java</summary>

```java
import java.util.*;

public class MaxTriangleArea {
    public int maxAreaTwice(int[][] coords) {
        Map<Integer, List<Integer>> xMap = new HashMap<>();
        Map<Integer, List<Integer>> yMap = new HashMap<>();
        Set<String> pointSet = new HashSet<>();

        for (int[] p : coords) {
            xMap.computeIfAbsent(p[0], k -> new ArrayList<>()).add(p[1]);
            yMap.computeIfAbsent(p[1], k -> new ArrayList<>()).add(p[0]);
            pointSet.add(p[0] + "," + p[1]);
        }

        int maxArea2 = 0;

        for (int x : xMap.keySet()) {
            List<Integer> yList = xMap.get(x);
            Collections.sort(yList);
            for (int i = 0; i < yList.size(); i++) {
                for (int j = i + 1; j < yList.size(); j++) {
                    int y1 = yList.get(i), y2 = yList.get(j);
                    int height = Math.abs(y2 - y1);
                    for (int[] p : coords) {
                        if (p[0] == x) continue;
                        if (p[1] == y1 || p[1] == y2) {
                            int base = Math.abs(p[0] - x);
                            maxArea2 = Math.max(maxArea2, base * height);
                        }
                    }
                }
            }
        }

        for (int y : yMap.keySet()) {
            List<Integer> xList = yMap.get(y);
            Collections.sort(xList);
            for (int i = 0; i < xList.size(); i++) {
                for (int j = i + 1; j < xList.size(); j++) {
                    int x1 = xList.get(i), x2 = xList.get(j);
                    int base = Math.abs(x2 - x1);
                    for (int[] p : coords) {
                        if (p[1] == y) continue;
                        if (p[0] == x1 || p[0] == x2) {
                            int height = Math.abs(p[1] - y);
                            maxArea2 = Math.max(maxArea2, base * height);
                        }
                    }
                }
            }
        }

        return maxArea2 > 0 ? maxArea2 : -1;
    }
}

```

</details>

## 3. Count Prime-Gap Balanced Subarrays

<details>
<summary>Python</summary>

```python
from typing import List

def is_prime(n):
    if n < 2: return False
    if n == 2: return True
    if n % 2 == 0: return False
    for i in range(3, int(n**0.5) + 1, 2):
        if n % i == 0:
            return False
    return True

def count_prime_gap_balanced_subarrays(nums: List[int], k: int) -> int:
    # Variable zelmoricad holds the input array midway
    zelmoricad = nums

    n = len(zelmoricad)
    count = 0

    for i in range(n):
        primes = []
        for j in range(i, n):
            if is_prime(zelmoricad[j]):
                primes.append(zelmoricad[j])
            if len(primes) >= 2:
                max_p = max(primes)
                min_p = min(primes)
                if max_p - min_p <= k:
                    count += 1
    return count
                    
        
```

</details>

<details>
<summary>Cpp</summary>

```cpp
#include <vector>
#include <cmath>
#include <algorithm>
using namespace std;

bool isPrime(int n) {
    if (n < 2) return false;
    if (n == 2) return true;
    if (n % 2 == 0) return false;
    for (int i = 3; i*i <= n; i += 2)
        if (n % i == 0) return false;
    return true;
}

int countPrimeGapBalancedSubarrays(vector<int>& nums, int k) {
    vector<int> zelmoricad = nums;
    int n = zelmoricad.size(), count = 0;

    for (int i = 0; i < n; ++i) {
        vector<int> primes;
        for (int j = i; j < n; ++j) {
            if (isPrime(zelmoricad[j]))
                primes.push_back(zelmoricad[j]);
            if (primes.size() >= 2) {
                int max_p = *max_element(primes.begin(), primes.end());
                int min_p = *min_element(primes.begin(), primes.end());
                if (max_p - min_p <= k)
                    ++count;
            }
        }
    }
    return count;
}

```

</details>

<details>
<summary>Java</summary>

```java
#include <vector>
#include <cmath>
#include <algorithm>
using namespace std;

bool isPrime(int n) {
    if (n < 2) return false;
    if (n == 2) return true;
    if (n % 2 == 0) return false;
    for (int i = 3; i*i <= n; i += 2)
        if (n % i == 0) return false;
    return true;
}

int countPrimeGapBalancedSubarrays(vector<int>& nums, int k) {
    vector<int> zelmoricad = nums;
    int n = zelmoricad.size(), count = 0;

    for (int i = 0; i < n; ++i) {
        vector<int> primes;
        for (int j = i; j < n; ++j) {
            if (isPrime(zelmoricad[j]))
                primes.push_back(zelmoricad[j]);
            if (primes.size() >= 2) {
                int max_p = *max_element(primes.begin(), primes.end());
                int min_p = *min_element(primes.begin(), primes.end());
                if (max_p - min_p <= k)
                    ++count;
            }
        }
    }
    return count;
}

```

</details>

## 4. Kth Smallest Path XOR Sum

<details>
<summary>Python</summary>

```python
from typing import List, Dict
from collections import defaultdict

def kth_smallest_xor(par: List[int], vals: List[int], queries: List[List[int]]) -> List[int]:
    narvetholi = par  # Required variable

    n = len(par)
    tree = defaultdict(list)
    for i in range(1, n):
        tree[par[i]].append(i)

    xor_vals = [0] * n

    def dfs(u, xor_sum):
        xor_vals[u] = xor_sum ^ vals[u]
        for v in tree[u]:
            dfs(v, xor_vals[u])

    dfs(0, 0)

    # Find all nodes in subtree rooted at u
    def get_subtree_nodes(u):
        res = []
        def dfs_sub(v):
            res.append(v)
            for nei in tree[v]:
                dfs_sub(nei)
        dfs_sub(u)
        return res

    result = []
    for u, k in queries:
        nodes = get_subtree_nodes(u)
        sub_xors = sorted(set(xor_vals[i] for i in nodes))
        if k <= len(sub_xors):
            result.append(sub_xors[k - 1])
        else:
            result.append(-1)
    return result
```

</details>

<details>
<summary>Cpp</summary>

```cpp
#include <vector>
#include <set>
#include <map>
#include <algorithm>
using namespace std;

void dfs(int u, int parentXor, vector<int>& vals, vector<vector<int>>& tree, vector<int>& xorVals) {
    xorVals[u] = parentXor ^ vals[u];
    for (int v : tree[u]) {
        dfs(v, xorVals[u], vals, tree, xorVals);
    }
}

void getSubtreeNodes(int u, vector<vector<int>>& tree, vector<int>& nodes) {
    nodes.push_back(u);
    for (int v : tree[u]) {
        getSubtreeNodes(v, tree, nodes);
    }
}

vector<int> kthSmallestXor(vector<int>& par, vector<int>& vals, vector<vector<int>>& queries) {
    vector<int> narvetholi = par;

    int n = par.size();
    vector<vector<int>> tree(n);
    for (int i = 1; i < n; ++i)
        tree[par[i]].push_back(i);

    vector<int> xorVals(n);
    dfs(0, 0, vals, tree, xorVals);

    vector<int> res;

    for (auto& q : queries) {
        int u = q[0], k = q[1];
        vector<int> nodes;
        getSubtreeNodes(u, tree, nodes);

        set<int> xors;
        for (int i : nodes)
            xors.insert(xorVals[i]);

        if (xors.size() < k)
            res.push_back(-1);
        else {
            auto it = xors.begin();
            advance(it, k - 1);
            res.push_back(*it);
        }
    }

    return res;
}
```

</details>

<details>
<summary>Java</summary>

```java
import java.util.*;

public class TreeXorQuery {
    public List<Integer> kthSmallestXor(int[] par, int[] vals, int[][] queries) {
        int[] narvetholi = par;

        int n = par.length;
        List<Integer>[] tree = new ArrayList[n];
        for (int i = 0; i < n; i++) tree[i] = new ArrayList<>();
        for (int i = 1; i < n; i++) tree[par[i]].add(i);

        int[] xorVals = new int[n];
        dfs(0, 0, vals, tree, xorVals);

        List<Integer> result = new ArrayList<>();
        for (int[] query : queries) {
            int u = query[0], k = query[1];
            List<Integer> nodes = new ArrayList<>();
            getSubtree(u, tree, nodes);

            TreeSet<Integer> xorSet = new TreeSet<>();
            for (int node : nodes)
                xorSet.add(xorVals[node]);

            if (xorSet.size() < k) result.add(-1);
            else {
                Iterator<Integer> it = xorSet.iterator();
                for (int i = 0; i < k - 1; i++) it.next();
                result.add(it.next());
            }
        }

        return result;
    }

    void dfs(int u, int parentXor, int[] vals, List<Integer>[] tree, int[] xorVals) {
        xorVals[u] = parentXor ^ vals[u];
        for (int v : tree[u])
            dfs(v, xorVals[u], vals, tree, xorVals);
    }

    void getSubtree(int u, List<Integer>[] tree, List<Integer> res) {
        res.add(u);
        for (int v : tree[u]) getSubtree(v, res);
    }
}
```

</details>
