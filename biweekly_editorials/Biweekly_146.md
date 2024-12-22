## Count Subarrays of length three with a Condition

<details>
<summary>Python</summary>

```python
class Solution:
    def countSubarrays(self, nums: List[int]) -> int:
        res = 0
        for i in range(2, len(nums)):
            if nums[i - 1] == (nums[i - 2] + nums[i]) * 2: res += 1
        return res
```

</details>

<details>
<summary>Cpp</summary>

```cpp
class Solution {
public:
    int countSubarrays(vector<int>& nums) {
        int res = 0;
        for (int i = 2; i < nums.size(); i++) {
            if (nums[i - 1] == (nums[i - 2] + nums[i]) * 2) {
                res++;
            }
        }
        return res;
    }
};
```

</details>

<details>
<summary>Java</summary>

```java

class Solution {
    public int countSubarrays(int[] nums) {
        int res = 0;
        for (int i = 2; i < nums.length; i++) {
            if (nums[i - 1] == (nums[i - 2] + nums[i]) * 2) {
                res++;
            }
        }
        return res;
    }
}
```

</details>

## Count Paths with given XOR value

<details>
<summary>Python</summary>

```python
class Solution:
    def countPathsWithXorValue(self, grid: List[List[int]], k: int) -> int:
        m, n = len(grid), len(grid[0])
        MOD = 10 ** 9 + 7
    
        @cache
        def dfs(row: int, col: int, current_xor: int) -> int:
            if row == m - 1 and col == n - 1: return 1 if (current_xor ^ grid[row][col]) == k else 0
            paths = 0
            if row + 1 < m: paths = (paths + dfs(row + 1, col, current_xor ^ grid[row][col])) % MOD
            if col + 1 < n: paths = (paths + dfs(row, col + 1, current_xor ^ grid[row][col])) % MOD
            return paths
    
        return dfs(0, 0, 0)
 
      
```

</details>

<details>
<summary>Cpp</summary>

```cpp

class Solution {
public:
    int countPathsWithXorValue(vector<vector<int>>& grid, int k) {
        int m = grid.size();
        int n = grid[0].size();
        int MOD = 1e9 + 7;

        unordered_map<long long, int> memo; 

        function<int(int, int, int)> dfs = [&](int row, int col, int current_xor) {
            if (row == m - 1 && col == n - 1) {
                return ((current_xor ^ grid[row][col]) == k) ? 1 : 0;
            }

            long long key = ((long long)row << 32) | ((long long)col << 16) | current_xor;
            if (memo.count(key)) {
                return memo[key];
            }

            int paths = 0;
            if (row + 1 < m) {
                paths = (paths + dfs(row + 1, col, current_xor ^ grid[row][col])) % MOD;
            }
            if (col + 1 < n) {
                paths = (paths + dfs(row, col + 1, current_xor ^ grid[row][col])) % MOD;
            }

            memo[key] = paths;
            return paths;
        };

        return dfs(0, 0, 0);
    }
};
```

</details>

<details>
<summary>Java</summary>

```java

class Solution {
    private int m, n, MOD = 1000000007;

    public int countPathsWithXorValue(int[][] grid, int k) {
        m = grid.length;
        n = grid[0].length;
        int max_xor = 1 << 7;
        
        int[][][] dp = new int[m][n][max_xor];

        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                for (int x = 0; x < max_xor; x++) {
                    dp[i][j][x] = -1;
                }
            }
        }
        return dfs(0, 0, 0, grid, k, dp);
    }

    private int dfs(int row, int col, int current_xor, int[][] grid, int k, int[][][] dp) {
        current_xor ^= grid[row][col];
        if (row == m - 1 && col == n - 1) {
            return (current_xor == k) ? 1 : 0;
        }

        if (dp[row][col][current_xor] != -1) {
            return dp[row][col][current_xor];
        }

        int paths = 0;
        if (row + 1 < m) {
            paths = (paths + dfs(row + 1, col, current_xor, grid, k, dp)) % MOD;
        }
        if (col + 1 < n) {
            paths = (paths + dfs(row, col + 1, current_xor, grid, k, dp)) % MOD;
        }

        dp[row][col][current_xor] = paths;
        return paths;
    }
}
```

</details>

## Check if grid can Cut into Pieces


<details>
<summary>Python</summary>

```python

    def checkValidCuts(self, n: int, A: List[List[int]]) -> bool:
        def check(A):
            res = 0
            A.sort()
            pre = A[0][1]
            for a,b in A:
                res += pre <= a
                pre = max(pre, b)
            return res > 1

        return check([a[::2] for a in A]) or check([a[1::2] for a in A])
```

</details>

<details>
<summary>Cpp</summary>

```cpp
bool checkValidCuts(int n, vector<vector<int>>& rectangles) {
        auto check = [](vector<vector<int>> A) -> bool {
            int res = 0;
            sort(A.begin(), A.end());
            int pre = A[0][1];
            for (auto& ab : A) {
                int a = ab[0], b = ab[1];
                res += pre <= a;
                pre = max(pre, b);
            }
            return res > 1;
        };

        vector<vector<int>> X, Y;
        for (auto& r : rectangles) {
            X.push_back({r[0], r[2]});
            Y.push_back({r[1], r[3]});
        }
        return check(X) || check(Y);
    }
```

</details>

<details>
<summary>Java</summary>

```java
public boolean checkValidCuts(int n, int[][] rectangles) {
        int[][] X = new int[rectangles.length][2];
        int[][] Y = new int[rectangles.length][2];
        for (int i = 0; i < rectangles.length; i++) {
            X[i][0] = rectangles[i][0];
            Y[i][0] = rectangles[i][1];
            X[i][1] = rectangles[i][2];
            Y[i][1] = rectangles[i][3];
        }
        return check(X) || check(Y);
    }

    private boolean check(int[][] A) {
        int res = 0;
        Arrays.sort(A, (a, b) -> Integer.compare(a[0], b[0]));
        int pre = A[0][1];
        for (int[] ab : A) {
            int a = ab[0], b = ab[1];
            if (pre <= a) {
                res++;
            }
            pre = Math.max(pre, b);
        }
        return res > 1;
    }
```

</details>

## SubSequences with a Unique middle mode



<details>
<summary>Python</summary>

```python

   class Solution:
    def subsequencesWithMiddleMode(self, nums: List[int]) -> int:
        n = len(nums)
        modulo = 10 ** 9 + 7
        
        @cache
        def comb_2(a):
            return a * (a - 1) // 2

        def calc_2(elem, first_counter, second_counter, first_other_cnt, second_other_cnt):
            # count include
            result = first_other_cnt * comb_2(second_other_cnt)

            for k, v in first_counter.items():
                if k == elem:
                    continue
                vv = second_counter[k]
                # count exclude triples -- [0 1] 1 [0 0]
                result -= v * comb_2(vv)
                # count exclude doubles -- [0 1] 1 [0 2] / [0 1] 1 [2 0]
                result -= v * vv * (second_other_cnt - vv)
            
            for k, v in second_counter.items():
                if k == elem:
                    continue
                # count exclude doubles -- [0 1] 1 [2 2]
                result -= comb_2(v) * (first_other_cnt - first_counter[k])
            
            return result

        result = 0
        left = Counter(nums[i] for i in range(2))
        right = Counter(nums[i] for i in range(2, n))

        for i in range(2, n - 2):
            elem = nums[i]

            # remove elem from right counter
            if right[elem] == 1:
                right.pop(elem)
            else:
                right[elem] -= 1
            
            left_cnt = left[elem]
            right_cnt = right[elem]
            left_other = i - left_cnt
            right_other = n - i - 1 - right_cnt

            result = (
                result
                # [1 1] 1 [1 1] - 5 elements
                + comb_2(left_cnt) * comb_2(right_cnt) % modulo
                # [1 1] 1 [0 1] - 4 elements
                + comb_2(left_cnt) * right_cnt * right_other % modulo
                # [0 1] 1 [1 1] - 4 elements
                + left_cnt * comb_2(right_cnt) * left_other % modulo
                # [1 1] 1 [0 0] - 3 elements
                + comb_2(left_cnt) * comb_2(right_other) % modulo
                # [0 0] 1 [1 1] - 3 elements
                + comb_2(right_cnt) * comb_2(left_other) % modulo
                # [0 1] 1 [0 1] - 3 elements
                + left_cnt * right_cnt * left_other * right_other % modulo
                # [0 1] 1 [0 0] - 2 elements
                + left_cnt * calc_2(elem, left, right, left_other, right_other) % modulo
                # [0 0] 1 [0 1] - 2 elements
                + right_cnt * calc_2(elem, right, left, right_other, left_other) % modulo
            ) % modulo
            
            # add elem to left counter
            left[elem] += 1
        
        return result % modulo
```

</details>

<details>
<summary>Cpp</summary>

```cpp
class Solution {
public:
    int subsequencesWithMiddleMode(vector<int>& nums) {
        int n = nums.size();
        const int MOD = 1e9 + 7;

        // Helper function to calculate combination of 2
        auto comb_2 = [](int a) {
            return (a * (a - 1LL)) / 2;
        };

        // Function to calculate the result for the current element
        auto calc_2 = [&](int elem, unordered_map<int, int>& first_counter, unordered_map<int, int>& second_counter, int first_other_cnt, int second_other_cnt) {
            long long result = first_other_cnt * comb_2(second_other_cnt);

            for (auto& [k, v] : first_counter) {
                if (k == elem) {
                    continue;
                }
                long long vv = second_counter[k];
                result -= v * comb_2(vv);
                result -= v * vv * (second_other_cnt - vv);
            }

            for (auto& [k, v] : second_counter) {
                if (k == elem) {
                    continue;
                }
                result -= comb_2(v) * (first_other_cnt - first_counter[k]);
            }

            return result;
        };

        long long result = 0;
        unordered_map<int, int> left, right;

        // Initialize left and right counters
        for (int i = 0; i < 2; i++) {
            left[nums[i]]++;
        }

        for (int i = 2; i < n; i++) {
            right[nums[i]]++;
        }

        for (int i = 2; i < n - 2; i++) {
            int elem = nums[i];

            // Remove the current element from the right counter
            if (right[elem] == 1) {
                right.erase(elem);
            } else {
                right[elem]--;
            }

            int left_cnt = left[elem];
            int right_cnt = right[elem];
            int left_other = i - left_cnt;
            int right_other = n - i - 1 - right_cnt;

            result = (result
                      + comb_2(left_cnt) * comb_2(right_cnt) % MOD
                      + comb_2(left_cnt) * right_cnt * right_other % MOD
                      + left_cnt * comb_2(right_cnt) * left_other % MOD
                      + comb_2(left_cnt) * comb_2(right_other) % MOD
                      + comb_2(right_cnt) * comb_2(left_other) % MOD
                      + left_cnt * right_cnt * left_other * right_other % MOD
                      + left_cnt * calc_2(elem, left, right, left_other, right_other) % MOD
                      + right_cnt * calc_2(elem, right, left, right_other, left_other) % MOD) % MOD;

            // Add the current element to the left counter
            left[elem]++;
        }

        return result % MOD;
    }
};
```

</details>

<details>
<summary>Java</summary>

```java
class Solution {
    private static final int MODULO = 1000000007;

    // Helper function to calculate combination of 2 (a choose 2)
    private long comb2(long a) {
        return a * (a - 1) / 2;
    }

    // Helper function to calculate the result for the second counter calculation
    private long calc2(int elem, Map<Integer, Long> firstCounter, Map<Integer, Long> secondCounter, 
                       long firstOtherCnt, long secondOtherCnt) {
        long result = firstOtherCnt * comb2(secondOtherCnt);

        for (Map.Entry<Integer, Long> entry : firstCounter.entrySet()) {
            int k = entry.getKey();
            long v = entry.getValue();
            if (k == elem) {
                continue;
            }
            long vv = secondCounter.getOrDefault(k, 0L);
            // count exclude triples -- [0 1] 1 [0 0]
            result -= v * comb2(vv);
            // count exclude doubles -- [0 1] 1 [0 2] / [0 1] 1 [2 0]
            result -= v * vv * (secondOtherCnt - vv);
        }

        for (Map.Entry<Integer, Long> entry : secondCounter.entrySet()) {
            int k = entry.getKey();
            long v = entry.getValue();
            if (k == elem) {
                continue;
            }
            // count exclude doubles -- [0 1] 1 [2 2]
            result -= comb2(v) * (firstOtherCnt - firstCounter.getOrDefault(k, 0L));
        }

        return result;
    }

    public int subsequencesWithMiddleMode(int[] nums) {
        int n = nums.length;
        Map<Integer, Long> left = new HashMap<>();
        Map<Integer, Long> right = new HashMap<>();

        // Initialize left and right counters
        for (int i = 0; i < 2; i++) {
            left.put(nums[i], left.getOrDefault(nums[i], 0L) + 1);
        }
        for (int i = 2; i < n; i++) {
            right.put(nums[i], right.getOrDefault(nums[i], 0L) + 1);
        }

        long result = 0;

        for (int i = 2; i < n - 2; i++) {
            int elem = nums[i];

            // Remove elem from right counter
            if (right.get(elem) == 1) {
                right.remove(elem);
            } else {
                right.put(elem, right.get(elem) - 1);
            }

            long leftCnt = left.getOrDefault(elem, 0L);
            long rightCnt = right.getOrDefault(elem, 0L);
            long leftOther = i - leftCnt;
            long rightOther = n - i - 1 - rightCnt;

            result = (result 
                + comb2(leftCnt) * comb2(rightCnt) % MODULO
                + comb2(leftCnt) * rightCnt * rightOther % MODULO
                + leftCnt * comb2(rightCnt) * leftOther % MODULO
                + comb2(leftCnt) * comb2(rightOther) % MODULO
                + comb2(rightCnt) * comb2(leftOther) % MODULO
                + leftCnt * rightCnt * leftOther * rightOther % MODULO
                + leftCnt * calc2(elem, left, right, leftOther, rightOther) % MODULO
                + rightCnt * calc2(elem, right, left, rightOther, leftOther) % MODULO
            ) % MODULO;

            // Add elem to left counter
            left.put(elem, left.getOrDefault(elem, 0L) + 1);
        }

        return (int) (result % MODULO);
    }
    
}
```

</details>
